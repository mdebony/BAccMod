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
from typing import List

import astropy.units as u
import numpy as np
from astropy.modeling import Model
from gammapy.maps import MapAxis

from .fitting import PoissonFitter
from .grid3d_acceptance_map_creator import Grid3DAcceptanceMapCreator

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
        list_name_normalisation_parameter: List[str] = None,
    ) -> None:
        """
        Abstract base class for fitting.  All the “core‐Poisson‐fit” logic lives here.

        Parameters
        ----------
        energy_axis : MapAxis
            The energy axis for the acceptance model
        offset_axis : MapAxis
            The offset axis for the acceptance model
        oversample_map : int, optional
            Oversample in number of pixel of the spatial axis used for the calculation
        exclude_regions : list of regions.SkyRegion, optional
            Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
        cos_zenith_binning_method : str, optional
            The method used for cos zenith binning: 'min_livetime','min_n_observation'
        cos_zenith_binning_parameter_value : int, optional
            Minimum livetime (in seconds) or number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation_wobble : u.Quantity, optional
            The maximum angular separation between identified wobbles, in degrees
        zenith_binning_run_splitting : float, optional
            If true, will split each run to match zenith binning for the base model computation
            Could be computationally expensive, especially at high zenith with a high resolution zenith binning
        max_fraction_pixel_rotation_fov : bool, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV and cut as function of the zenith bins
        use_mini_irf_computation : bool, optional
            If true, in case the case of zenith interpolation or binning, each run will be divided in small subrun (the slicing is based on time).
            A model will be computed for each sub run before averaging them to obtain the final model for the run.
            Should improve the accuracy of the model, especially at high zenith angle.
        mini_irf_time_resolution : astropy.units.Quantity, optional
            Time resolution to use for mini irf used for computation of the final background model
        interpolation_type: str, optional
            Select the type of interpolation to be used, could be either "log" or "linear", log tend to provided better results be could more easily create artefact that will cause issue
        activate_interpolation_cleaning: bool, optional
            If true, will activate the cleaning step after interpolation, it should help to eliminate artefact caused by interpolation
        interpolation_cleaning_energy_relative_threshold: float, optional
            To be considered value, the bin in energy need at least one adjacent bin with a relative difference within this range
        interpolation_cleaning_spatial_relative_threshold: float, optional
            To be considered value, the bin in space need at least one adjacent bin with a relative difference within this range
        list_name_normalisation_parameter: list of string, optional
            All the parameters contained in this list in the model will be automatically normalised based on overall counts at the start of the fit, normalisation correction is done with hypothesis of addition of components, therefore they will be all corrected by the same factor
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

        self.sq_rel_residuals = {"mean": [], "std": []}
        self.list_name_normalisation_parameter = list_name_normalisation_parameter

    @abstractmethod
    def create_acceptance_map(self, observations):
        """
        Subclasses must implement: run self._create_base_computation_map(), then
        call self.fit_background() for each energy slice, and finally pack into Background3D.
        """
        pass

    def fit_background(self, model: Model, *coords: List[np.ndarray], count_map: np.ndarray, exp_map_total: np.ndarray, exp_map: np.ndarray) -> np.ndarray:
        """
        Perform a Poisson fit on the given map (could be 1D, 2D or 3D), given
        the fine‐binned exposure (exp_map) and total‐exposure (exp_map_total),
        returning a smooth “counts model” on the same grid as `count_map`.

        Parameters
        ----------
        model : astropy.modeling.model
            the model to fit to the data
        coords : List of np.ndarray
            the list of coordinate
        count_map : np.ndarray
            counts in each pixel, integer
        exp_map_total : np.ndarray
            exposure WITHOUT exclusion regions
        exp_map : np.ndarray
            exposure CORRECTED for exclusion regions

        Returns
        -------
        fitted_counts_model : 2D np.ndarray
            The best‐fit model of counts (on the same pixel grid as count_map).
        """

        # Compute exposure correction
        mask = exp_map_total > 0
        exp_map_total_safe = exp_map_total.copy()
        exp_map_total_safe[mask] = 1
        exp_correction = exp_map / exp_map_total_safe

        # Initialise the model
        model_init = model.copy()

        # Correct normalisation of the model
        if self.list_name_normalisation_parameter is not None and len(self.list_name_normalisation_parameter) > 0:
            # Compute correction
            init_count_model = np.sum(model_init(*coords)*exp_correction)
            correction_norm = np.sum(count_map)/init_count_model

            # build list of free parameters
            tied = model_init.tied or {}
            all_params = list(model_init.param_names)
            free_params = [p for p in all_params if (p not in tied) or (not tied[p]) and (not model_init.fixed.get(p, False))]

            # Apply correction
            for p in free_params:
                if p in self.list_name_normalisation_parameter:
                    setattr(model_init, p, getattr(model_init, p)*correction_norm)

        # Fit the model
        fitter = PoissonFitter()
        best_model = fitter(model_init, *coords, data=count_map, exposure_correction=exp_correction)

        # Collect results & (optionally) track residuals
        if logger.getEffectiveLevel() <= logging.INFO:
            fitted_model = best_model(*coords) * exp_correction
            fitted_model[exp_map == 0] = 1.0
            rel_resid = 100 * (count_map - fitted_model) / fitted_model
            sq_rel_resid = (count_map - fitted_model) / np.sqrt(fitted_model)
            self.sq_rel_residuals["mean"].append(np.mean(sq_rel_resid))
            self.sq_rel_residuals["std"].append(np.std(sq_rel_resid))
            param_dict = dict(zip(best_model.param_names, best_model.parameters))
            logger.info(f"Fit results ({model.__name__}): {param_dict}")
            logger.debug(
                f"  Avg rel residual: {np.mean(rel_resid):.1f}%,  Std = {np.std(rel_resid):.2f}%\n"
            )

        return best_model(*coords)
