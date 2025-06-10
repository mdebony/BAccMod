# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: base_fit_acceptance_map_creator.py
# Purpose: Abstract base class for “fit”‐style acceptance.  Implements everything needed for fitting background data
#          Subclasses must implement create_acceptance_map().
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


import logging
from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from gammapy.maps import MapAxis
from iminuit import Minuit

from .grid3d_acceptance_map_creator import Grid3DAcceptanceMapCreator
from .modeling import FIT_FUNCTION, log_factorial, log_poisson

logger = logging.getLogger(__name__)


class BaseFitAcceptanceMapCreator(Grid3DAcceptanceMapCreator, ABC):
    """
    Abstract base for any “fit”‐style 3D‐acceptance creator.  Inherits from Grid3DAcceptanceMapCreator
    to have access to:
      - self._create_base_computation_map()
      - self._transform_exclusion_region_to_camera_frame()
      - etc.
    Implements `fit_background()` for a single energy slice.  Subclasses must implement
    `create_acceptance_map()` to define how to assemble all energies + spatial layout.
    """

    def __init__(
        self,
        energy_axis: MapAxis,
        offset_axis: MapAxis,
        oversample_map: int = 10,
        exclude_regions=None,
        cos_zenith_binning_method: str = 'min_livetime',
        cos_zenith_binning_parameter_value: int = 3600,
        initial_cos_zenith_binning: float = 0.01,
        max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
        zenith_binning_run_splitting: bool = False,
        max_fraction_pixel_rotation_fov: float = 0.5,
        time_resolution: u.Quantity = 0.1 * u.s,
        use_mini_irf_computation: bool = False,
        mini_irf_time_resolution: u.Quantity = 1. * u.min,
        interpolation_type: str = 'linear',
        activate_interpolation_cleaning: bool = False,
        interpolation_cleaning_energy_relative_threshold: float = 1e-4,
        interpolation_cleaning_spatial_relative_threshold: float = 1e-2,
        fit_fnc='gaussian2d',
        fit_seeds=None,
        fit_bounds=None,
    ) -> None:
        """
        Abstract base class for fitting.  All the “core‐Poisson‐fit” logic lives here.

        Parameters
        ----------
        energy_axis, offset_axis, oversample_map, etc.  (all the same spatial parameters
        as Grid3DAcceptanceMapCreator).
        fit_fnc : str or callable
            If str, must be a key in FIT_FUNCTION, otherwise a custom function with signature
            f(x, y, *params) → counts_density.
        fit_seeds : dict, optional
            Initial seeds for the fit parameters (excluding “size” which is always re‐derived).
        fit_bounds : dict, optional
            Bounds on fit parameters.  “size” bound is overridden by seeds logic.
        """
        # Call the “stack”‐only constructor in Grid3D, to set up geometry, offset axes, etc.
        super().__init__(
            energy_axis=energy_axis,
            offset_axis=offset_axis,
            oversample_map=oversample_map,
            exclude_regions=exclude_regions,
            cos_zenith_binning_method=cos_zenith_binning_method,
            cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
            initial_cos_zenith_binning=initial_cos_zenith_binning,
            max_angular_separation_wobble=max_angular_separation_wobble,
            zenith_binning_run_splitting=zenith_binning_run_splitting,
            max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
            time_resolution=time_resolution,
            use_mini_irf_computation=use_mini_irf_computation,
            mini_irf_time_resolution=mini_irf_time_resolution,
            interpolation_type=interpolation_type,
            activate_interpolation_cleaning=activate_interpolation_cleaning,
            interpolation_cleaning_energy_relative_threshold=interpolation_cleaning_energy_relative_threshold,
            interpolation_cleaning_spatial_relative_threshold=interpolation_cleaning_spatial_relative_threshold,
        )

        # Store fitting‐specific members
        self.fit_fnc = fit_fnc
        self.fit_seeds = fit_seeds or {}
        self.fit_bounds = fit_bounds or {}
        self.sq_rel_residuals = {"mean": [], "std": []}

    @abstractmethod
    def create_acceptance_map(self, observations):
        """
        Subclasses must implement: run self._create_base_computation_map(), then
        call self.fit_background() for each energy slice, and finally pack into Background3D.
        """
        pass

    def fit_background(self, count_map: np.ndarray, exp_map_total: np.ndarray, exp_map: np.ndarray) -> np.ndarray:
        """
        Perform a 2D Poisson fit on one [x,y] slice of `count_map`, given
        the fine‐binned exposure (exp_map) and total‐exposure (exp_map_total),
        returning a smooth “counts model” on the same [x,y] grid as `count_map`.

        Parameters
        ----------
        count_map : 2D np.ndarray (counts in each pixel, integer)
        exp_map_total : 2D np.ndarray (exposure WITHOUT exclusion regions)
        exp_map : 2D np.ndarray (exposure CORRECTED for exclusion regions)

        Returns
        -------
        fitted_counts_model : 2D np.ndarray
            The best‐fit model of counts (on the same pixel grid as count_map).
        """
        # 1) prepare x,y mesh from the offset axis centers
        centers = self.offset_axis.center.to_value(u.deg)
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        x, y = np.meshgrid(centers, centers)

        # 2) choose fit function
        if isinstance(self.fit_fnc, str):
            try:
                fnc = FIT_FUNCTION[self.fit_fnc]
            except KeyError:
                logger.error(f"Invalid built‐in fit_fnc: {self.fit_fnc}.")
                raise
            raw_seeds = fnc.default_seeds.copy()
            bounds = fnc.default_bounds.copy()
        else:
            fnc = self.fit_fnc
            raw_seeds = {}
            bounds = {}

        # 3) override seeds/bounds if provided by user
        raw_seeds.update(self.fit_seeds)
        bounds.update(self.fit_bounds)

        # 4) seed “size” from total counts / mean exposure
        mask = exp_map > 0
        raw_seeds["size"] = np.sum(count_map[mask] * exp_map_total[mask] / exp_map[mask]) / np.mean(mask)
        bounds["size"] = (raw_seeds["size"] * 0.1, raw_seeds["size"] * 10)

        # 5) re‐order seeds to match fnc signature (fnc(x,y,*params))
        param_names = list(fnc.__code__.co_varnames[: fnc.__code__.co_argcount])
        param_names.remove("x")
        param_names.remove("y")
        seeds_ordered = {k: raw_seeds[k] for k in param_names}

        # 6) build Poisson‐log‐likelihood to minimize (Minuit likes to minimize)
        log_fact = log_factorial(count_map)

        def neg_logL(*args):
            model_counts = fnc(x, y, *args) * exp_map / exp_map_total
            return -np.sum(log_poisson(count_map, model_counts, log_fact))

        # 7) run Minuit
        m = Minuit(neg_logL, name=seeds_ordered.keys(), *seeds_ordered.values())
        for key, bnd in bounds.items():
            if bnd is None:
                m.fixed[key] = True
            else:
                m.limits[key] = bnd
        m.errordef = Minuit.LIKELIHOOD
        m.simplex().migrad()
        if not m.valid:
            logger.warning("Fit invalid for this energy/zenith bin.")

        # 8) collect results & (optionally) track residuals
        if logger.getEffectiveLevel() <= logging.INFO:
            fitted_model = fnc(x, y, **m.values.to_dict()) * exp_map / exp_map_total
            fitted_model[exp_map == 0] = 1.0
            rel_resid = 100 * (count_map - fitted_model) / fitted_model
            sq_rel_resid = (count_map - fitted_model) / np.sqrt(fitted_model)
            self.sq_rel_residuals["mean"].append(np.mean(sq_rel_resid))
            self.sq_rel_residuals["std"].append(np.std(sq_rel_resid))
            logger.info(f"Fit results ({fnc.__name__}): {m.values.to_dict()}")
            logger.debug(
                f"  Avg rel residual: {np.mean(rel_resid):.1f}%,  Std = {np.std(rel_resid):.2f}%\n"
            )

        return fnc(x, y, **m.values.to_dict())
