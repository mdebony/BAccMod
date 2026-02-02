# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: spatial_fit_acceptance_map_creator.py
# Purpose: Concrete class implementing “fit” for a spatial 3D grid acceptance model
# ---------------------------------------------------------------------

import logging

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.functional_models import Gaussian2D
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis

from .base_fit_acceptance_map_creator import BaseFitAcceptanceMapCreator

logger = logging.getLogger(__name__)


class SpatialFitAcceptanceMapCreator(BaseFitAcceptanceMapCreator):
    """
    Implements the “fit” method for a 3D grid acceptance model.
    Relies on BaseFitAcceptanceMapCreator.fit_background(...) for each energy slice.
    """

    def __init__(
        self,
        model_to_fit: Fittable2DModel = Gaussian2D(),
        **kwargs
    ) -> None:
        """
        Spatial “fit” acceptance creator: splits each energy slice, calls BaseFitAcceptanceMapCreator.fit_background(),
        then normalizes into [counts / (MeV‐sr s)] and builds a Background3D.

        Parameters
        ----------
        model_to_fit: Fittable2DModel, optional
            The model to fit to the data
        **kwargs
            Additional arguments for controlling background creation,
            see documentation of BaseAcceptanceMapCreator, Grid3DAcceptanceMapCreator and BaseFitAcceptanceMapCreator
        """
        super().__init__(**kwargs)

        self.model_to_fit = model_to_fit

    def create_model(self, observations) -> Background3D:
        """
        Calculate a 3D grid acceptance map by fitting 2D model in each energy slice

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background3D
        """

        # 1) gather fine-binned counts + exposures + livetime
        count_background, exp_map_background, exp_map_background_total, livetime, energy_axis_computation  = self._create_base_computation_map(
            observations
        )

        # 2) downsample exposures
        exp_ds = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_total_ds = exp_map_background_total.downsample(self.oversample_map, preserve_counts=True)

        # 3) for each energy slice, fit 2D function → “corrected counts” on [x,y] grid
        n_energy = count_background.shape[0]
        corrected_counts = np.empty(count_background.shape)
        self.sq_rel_residuals = {"mean": [], "std": []}

        # 2D coordinates system for the fit
        centers = self.offset_axis.center.to_value(u.deg)
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        x, y = np.meshgrid(centers, centers)

        for e in range(n_energy):
            logger.info(
                f"Fitting background, energy bin [{energy_axis_computation.edges[e]:.2f}, "
                f"{energy_axis_computation.edges[e + 1]:.2f}]"
            )
            corrected_counts[e] = self._fit_background(
                self.model_to_fit,
                x, y,
                count_map=count_background[e].astype(int),
                exp_map_total=exp_total_ds.data[e],
                exp_map=exp_ds.data[e],
            )

        logger.info(
            "Average event sqrt‐residuals per energy: "
            f"{np.round(self.sq_rel_residuals['mean'], 2)}, std = {np.round(self.sq_rel_residuals['std'], 2)}"
        )

        # 4) build final offset axes (matching Grid3DAcceptanceMapCreator)
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name="fov_lon")
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name="fov_lat")

        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], extended_offset_axis_x.nbin, axis=1)
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], extended_offset_axis_y.nbin, axis=0)
        solid_angle = 4.0 * (np.sin(bin_width_x / 2) * np.sin(bin_width_y / 2)) * u.steradian

        # 5) normalize: data_background[e, :, :] = corrected_counts[e] / (solid_angle * energy_bin_width * livetime)
        data_background = (
            corrected_counts
            / solid_angle[np.newaxis, :, :]
            / energy_axis_computation.bin_width[:, np.newaxis, np.newaxis]
            / livetime
        )
        data_background = self._interpolate_bkg_to_energy_axis(data_background, energy_axis_computation)

        # 6) instantiate Background3D
        acceptance_map = Background3D(axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                      data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),
                                      fov_alignment=FoVAlignment.ALTAZ)
        return acceptance_map
