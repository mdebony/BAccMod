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

import numpy as np
from astropy.modeling import Model

from .fitting import poisson_fitter
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
        list_name_normalisation_parameter: List[str] = ['amplitude'],
        **kwargs
    ) -> None:
        """
        Abstract base class for fitting.  All the “core‐Poisson‐fit” logic lives here.

        Parameters
        ----------
        list_name_normalisation_parameter: list of string, optional
            All the parameters contained in this list in the model will be automatically normalised based on overall
            counts at the start of the fit, normalisation correction is done with hypothesis of addition of components,
            therefore they will be all corrected by the same factor
        **kwargs
            Additional arguments for controlling background creation,
            see documentation of BaseAcceptanceMapCreator and Grid3DAcceptanceMapCreator for more details
        """

        # Call the “stack”‐only constructor in Grid3D, to set up geometry, offset axes, etc.
        super().__init__(**kwargs)

        self.sq_rel_residuals = {"mean": [], "std": []}
        self.list_name_normalisation_parameter = list_name_normalisation_parameter

    @abstractmethod
    def create_model(self, observations):
        """
        Subclasses must implement: run self._create_base_computation_map(), then
        call self.fit_background() for each slice (energy, spatial or all depending on the implementation), and finally pack into Background3D.
        """
        pass

    def _fit_background(self, model: Model, *coords: List[np.ndarray], count_map: np.ndarray, exp_map_total: np.ndarray, exp_map: np.ndarray) -> np.ndarray:
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

        total_counts = np.sum(count_map)
        # Skip the fit if there are no events
        if total_counts == 0:
            return count_map

        # Compute exposure correction
        mask = exp_map_total > 0
        exp_map_total_safe = exp_map_total.copy()
        exp_map_total_safe[~mask] = 1
        exp_correction = exp_map / exp_map_total_safe

        # Initialise the model
        model_init = model.copy()

        # Correct normalisation of the model
        if self.list_name_normalisation_parameter is not None and len(self.list_name_normalisation_parameter) > 0:
            # Compute correction
            init_count_model = np.sum(model_init(*coords)*exp_correction)
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
                    logger.warning(f'{p} is not a parameters of the model and therefore normalisation was not adjusted for this parameter')

        # Fit the model
        if total_counts>0:
            best_model = poisson_fitter(model_init, *coords, data=count_map, exposure_correction=exp_correction, mask=mask)
            # Collect results & (optionally) track residuals
            if logger.getEffectiveLevel() <= logging.INFO:
                fitted_model = best_model(*coords) * exp_correction
                fitted_model[exp_map == 0] = 1.0
                rel_resid = 100 * (count_map - fitted_model) / fitted_model
                sq_rel_resid = (count_map - fitted_model) / np.sqrt(fitted_model)
                self.sq_rel_residuals["mean"].append(np.mean(sq_rel_resid))
                self.sq_rel_residuals["std"].append(np.std(sq_rel_resid))
                param_dict = dict(zip(best_model.param_names, best_model.parameters))
                logger.info(f"Fit results ({type(best_model).__name__}): {param_dict}")
                logger.debug(
                    f"  Avg rel residual: {np.mean(rel_resid):.1f}%,  Std = {np.std(rel_resid):.2f}%\n"
                )
        else:
            best_model = model_init
            self.sq_rel_residuals["mean"].append(np.nan)
            self.sq_rel_residuals["std"].append(np.nan)


        return best_model(*coords)
