# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: base_fit_acceptance_map_creator.py
# Purpose: Class for creation of model by fitting on a 3D grid (spatial and energy).
#          It supports 3D models, 2D spatial models (fit per energy bin), and 1D spectral models (fit per spatial bin).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


import logging
from typing import List

import numpy as np
import astropy.units as u
from astropy.modeling import Model, FittableModel
from astropy.modeling.functional_models import Gaussian2D

from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis

from .fitting import poisson_fitter
from .grid3d_acceptance_map_creator import Grid3DAcceptanceMapCreator
from .logging import MOREINFO

logger = logging.getLogger(__name__)


class FitAcceptanceMapCreator(Grid3DAcceptanceMapCreator):

    def __init__(
        self,
        energy_axis: MapAxis,
        offset_axis: MapAxis,
        model_to_fit: FittableModel = Gaussian2D(),
        list_name_normalisation_parameter: List[str] = ['amplitude'],
        maxiter: int = 1000,
        **kwargs
    ) -> None:
        """
        Class for fitting a 3D acceptance model.

        Parameters
        ----------
        model_to_fit: FittableModel, optional
            The model to fit to the data. Can be 1, 2 or 3 dimensionnal.
        list_name_normalisation_parameter: list of string, optional
            All the parameters contained in this list in the model will be automatically normalised based on overall
            counts at the start of the fit, normalisation correction is done with hypothesis of addition of components,
            therefore they will be all corrected by the same factor
        maxiter : int
            maximum number of iteration for fitting, as the fitting is performed in two steps,
            could be the double of this value in practice
        **kwargs
            Additional arguments for controlling background creation,
            see documentation of BaseAcceptanceMapCreator and Grid3DAcceptanceMapCreator for more details
        """

        # Call the “stack”‐only constructor in Grid3D, to set up geometry, offset axes, etc.
        super().__init__(energy_axis, offset_axis, **kwargs)

        self.sq_rel_residuals = {"mean": [], "std": []}
        self.model_to_fit = model_to_fit
        self.list_name_normalisation_parameter = list_name_normalisation_parameter
        self.maxiter = maxiter

    def create_model(self, observations) -> Background3D:
        """
        Calculate a 3D grid acceptance map by fitting a model, either globally, or per energy or spatial slices.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background3D
        """

        # 1) gather fine-binned counts + exposures + livetime
        (count_background,
         exp_map_background,
         exp_map_background_total,
         livetime) = self._create_base_computation_map(observations)

        # 2) downsample exposures
        exp_ds = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_total_ds = exp_map_background_total.downsample(self.oversample_map, preserve_counts=True)

        # 3) build final offset axes (matching Grid3DAcceptanceMapCreator)
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name="fov_lon")
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name="fov_lat")

        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], extended_offset_axis_x.nbin, axis=1)
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], extended_offset_axis_y.nbin, axis=0)
        solid_angle = 4.0 * (np.sin(bin_width_x / 2) * np.sin(bin_width_y / 2)) * u.steradian

        energy_bin_width = self.energy_axis_computation.bin_width

        # 4) fit function on counts → “corrected counts”
        predicted_counts = np.empty(count_background.shape)
        self.sq_rel_residuals = {"mean": [], "std": []}

        # coordinates system for the fit, based on the model dimension
        # order should be (energy, lon, lat) in the model for the fit to be correct
        coordinates = []
        centers = self.offset_axis.center.to_value('deg')
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        if self.model_to_fit.n_inputs != 2:
            coordinates.append(self.energy_axis_computation.center.to_value('TeV'))
        if self.model_to_fit.n_inputs > 1:
            coordinates.append(centers)
            coordinates.append(centers)
        coords = np.meshgrid(*coordinates, indexing='ij')

        # perform the fit, looping over non fitted axes
        if self.model_to_fit.n_inputs == 3:
            bin_size = (solid_angle[np.newaxis, :, :] * energy_bin_width[:, np.newaxis, np.newaxis]).value
            logger.info(
                "Fitting background with a 3D model."
            )
            predicted_counts = self._fit_background(
                self.model_to_fit,
                *coords,
                count_map=count_background.astype(int),
                exp_map_total=exp_total_ds.data,
                exp_map=exp_ds.data,
                bin_size=bin_size
            )
        elif self.model_to_fit.n_inputs == 2:
            bin_size = solid_angle.value
            logger.info(
                "Fitting background per enery bin"
                )
            for e in range(count_background.shape[0]):
                logger.log(MOREINFO,
                    "Fitting energy bin [%.2f, %.2f] TeV",
                    self.energy_axis_computation.edges[e].to_value('TeV'),
                    self.energy_axis_computation.edges[e + 1].to_value('TeV')
                )
                predicted_counts[e] = self._fit_background(
                    self.model_to_fit,
                    *coords,
                    count_map=count_background[e].astype(int),
                    exp_map_total=exp_total_ds.data[e],
                    exp_map=exp_ds.data[e],
                    bin_size=bin_size
                )
        elif self.model_to_fit.n_inputs == 1:
            bin_size = energy_bin_width.value
            logger.info(
                "Fitting background per spatial bin"
                )
            for x in range(count_background.shape[1]):
                for y in range(count_background.shape[2]):
                    logger.log(MOREINFO,
                        "Fitting spatial bin %.2f, %.2f deg", centers[x], centers[y]
                    )
                    predicted_counts[:,x,y] = self._fit_background(
                        self.model_to_fit,
                        *coords,
                        count_map=count_background[:,x,y].astype(int),
                        exp_map_total=exp_total_ds.data[:,x,y],
                        exp_map=exp_ds.data[:,x,y],
                        bin_size=bin_size
                    )
        else:
            raise RuntimeError(f"The provided model dimension is incorrect : {self.model_to_fit.n_inputs}")

        logger.info(
            "Average event sqrt‐residuals per energy: %s, std = %s",
            np.array_str(np.round(self.sq_rel_residuals['mean'], 2)),
            np.array_str(np.round(self.sq_rel_residuals['std'], 2))
        )

        # 5) normalize to flux units
        data_background = (
            predicted_counts
            / solid_angle[np.newaxis, :, :]
            / self.energy_axis_computation.bin_width[:, np.newaxis, np.newaxis]
            / livetime
        )
        data_background = self._interpolate_bkg_to_energy_axis(data_background)

        # 6) instantiate Background3D
        acceptance_map = Background3D(axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                      data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),
                                      fov_alignment=FoVAlignment.ALTAZ)
        return acceptance_map

    def _fit_background(self, model: Model, *coords: np.ndarray, count_map: np.ndarray,
                        exp_map_total: np.ndarray, exp_map: np.ndarray, bin_size: np.ndarray) -> np.ndarray:
        """
        Perform a Poisson fit on the given map (could be 1D, 2D or 3D), given
        the fine‐binned exposure (exp_map) and total‐exposure (exp_map_total),
        returning a smooth “counts model” on the same grid as `count_map`.

        Parameters
        ----------
        model : astropy.modeling.model
            the model to fit to the data
        coords : np.ndarray
            the list of coordinates
        count_map : np.ndarray
            counts in each bin, integer
        exp_map_total : np.ndarray
            exposure WITHOUT exclusion regions
        exp_map : np.ndarray
            exposure CORRECTED for exclusion regions
        bin_size : np.ndarray
            integral size of each bin

        Returns
        -------
        fitted_counts_model : np.ndarray
            The best‐fit model of counts (on the same bin grid as count_map).
        """
        total_counts = np.sum(count_map)
        # Skip the fit if there are no events
        if total_counts == 0:
            logger.info("No events")
            self.sq_rel_residuals["mean"].append(np.nan)
            self.sq_rel_residuals["std"].append(np.nan)
            return count_map

        # Compute exposure correction, avoiding dividing by 0
        mask = exp_map_total > 0
        exp_map_total_safe = exp_map_total.copy()
        exp_map_total_safe[~mask] = 1
        exp_correction = exp_map / exp_map_total_safe

        # Initialise the model
        model_init = model.copy()

        # Correct normalisation of the model
        if self.list_name_normalisation_parameter is not None and len(self.list_name_normalisation_parameter) > 0:
            # Compute correction
            init_count_model = np.sum(model_init(*coords)*exp_correction*bin_size)
            correction_norm = total_counts/init_count_model

            # build list of free parameters
            tied = model_init.tied
            all_params = list(model_init.param_names)
            indep_params = [p for p in all_params if not tied[p]]

            # Apply correction
            for p in self.list_name_normalisation_parameter:
                if p in indep_params:
                    setattr(model_init, p, getattr(model_init, p)*correction_norm)
                else:
                    logger.warning('%s is not a parameters of the model and therefore normalisation was not adjusted'
                                   'for this parameter', p)

        # Fit the model
        best_model = poisson_fitter(model_init, *coords, data=count_map, exposure_correction=exp_correction,
                                    mask=mask, maxiter=self.maxiter, bin_size=bin_size)
        # Collect results & (optionally) track residuals
        if logger.getEffectiveLevel() <= logging.INFO:
            fitted_model = best_model(*coords) * exp_correction * bin_size
            fitted_model[exp_map == 0] = 1.0
            rel_resid = 100 * (count_map - fitted_model) / fitted_model
            sq_rel_resid = (count_map - fitted_model) / np.sqrt(fitted_model)
            self.sq_rel_residuals["mean"].append(np.mean(sq_rel_resid))
            self.sq_rel_residuals["std"].append(np.std(sq_rel_resid))
            param_dict = dict(zip(best_model.param_names, best_model.parameters))
            logger.log(MOREINFO, "Fit results (%s): %s", type(best_model).__name__, str(param_dict))
            logger.debug(
                "Avg rel residual: %.1f,  Std = %.2f\n", np.mean(rel_resid), np.std(rel_resid)
            )

        return best_model(*coords) * bin_size
