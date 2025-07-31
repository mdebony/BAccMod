# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: spatial_fit_acceptance_map_creator.py
# Purpose: Concrete class implementing “fit” for a spatial 3D grid acceptance model
# ---------------------------------------------------------------------

import logging
from typing import List

import astropy.units as u
import gammapy
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
        model_to_fit: Fittable2DModel = Gaussian2D()
    ) -> None:
        """
        Spatial “fit” acceptance creator: splits each energy slice, calls BaseFitAcceptanceMapCreator.fit_background(),
        then normalizes into [counts / (MeV‐sr s)] and builds a Background3D.

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
        model_to_fit: Fittable2DModel, optional
            The model to fit to the data
        """
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
            list_name_normalisation_parameter=list_name_normalisation_parameter
        )

        self.model_to_fit = model_to_fit

    def create_acceptance_map(self, observations) -> Background3D:
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
        count_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
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
                f"Fitting background, energy bin [{self.energy_axis.edges[e]:.2f}, "
                f"{self.energy_axis.edges[e + 1]:.2f}]"
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
        ext_off_x = MapAxis.from_edges(extended_edges, name="fov_lon")
        ext_off_y = MapAxis.from_edges(extended_edges, name="fov_lat")

        bin_width_x = np.repeat(ext_off_x.bin_width[:, np.newaxis], ext_off_x.nbin, axis=1)
        bin_width_y = np.repeat(ext_off_y.bin_width[np.newaxis, :], ext_off_y.nbin, axis=0)
        solid_angle = 4.0 * (np.sin(bin_width_x / 2) * np.sin(bin_width_y / 2)) * u.steradian

        # 5) normalize: data_background[e, :, :] = corrected_counts[e] / (solid_angle * energy_bin_width * livetime)
        data_background = (
            corrected_counts
            / solid_angle[np.newaxis, :, :]
            / self.energy_axis.bin_width[:, np.newaxis, np.newaxis]
            / livetime
        )

        # 6) instantiate Background3D
        gammapy_major = int(gammapy.__version__.split(".")[0])
        gammapy_minor = int(gammapy.__version__.split(".")[1])
        if gammapy_major == 1 and gammapy_minor >= 3:
            data_final = np.flip(data_background.to(u.Unit("s-1 MeV-1 sr-1")), axis=1)
            acceptance_map = Background3D(
                axes=[self.energy_axis, ext_off_x, ext_off_y],
                data=data_final,
                fov_alignment=FoVAlignment.ALTAZ,
            )
        else:
            acceptance_map = Background3D(
                axes=[self.energy_axis, ext_off_x, ext_off_y],
                data=data_background.to(u.Unit("s-1 MeV-1 sr-1")),
                fov_alignment=FoVAlignment.ALTAZ,
            )

        return acceptance_map
