# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: radial_acceptance_map_creator.py
# Purpose: Class for creating background model with spatial circular symmetry rotation
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from gammapy.data import Observations
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D
from gammapy.maps import MapAxis, WcsNDMap, WcsGeom
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, SkyRegion

from .base_acceptance_map_creator import BaseAcceptanceMapCreator


class RadialAcceptanceMapCreator(BaseAcceptanceMapCreator):

    def __init__(self,
                 energy_axis: MapAxis,
                 offset_axis: MapAxis,
                 oversample_map: int = 10,
                 energy_axis_computation: MapAxis = None,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 dynamic_energy_axis: bool = False,
                 dynamic_energy_axis_target_statistics: int = 500,
                 dynamic_energy_axis_maximum_wideness_bin: float = 0.5,
                 dynamic_energy_axis_merge_zeros_low_energy: bool = False,
                 dynamic_energy_axis_merge_zeros_high_energy: bool = False,
                 cos_zenith_binning_method: str = 'min_livetime',
                 cos_zenith_binning_parameter_value: int = 3600,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
                 zenith_binning_run_splitting: bool = False,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution: u.Quantity = 0.1 * u.s,
                 use_mini_irf_computation: bool = False,
                 mini_irf_time_resolution: u.Quantity = 1. * u.min,
                 azimuth_east_west_splitting = False,
                 interpolation_zenith_type: str = 'linear',
                 activate_interpolation_zenith_cleaning: bool = False,
                 interpolation_cleaning_energy_relative_threshold: float = 1e-4,
                 interpolation_cleaning_spatial_relative_threshold: float = 1e-2) -> None:
        """
        Create the class for calculating radial acceptance model
        This class should be use when strict 2D model is good enough

        Parameters
        ----------
        energy_axis : MapAxis
            The energy axis for the acceptance model
        offset_axis : MapAxis
            The offset axis for the acceptance model
        oversample_map : int, optional
            Oversample in number of pixel of the spatial axis used for the calculation
        energy_axis_computation : gammapy.maps.geom.MapAxis
            The energy axis used for computation of the models, the model will then be reinterpolated on energy axis, if None, energy_axis will be used
        exclude_regions : list of regions.SkyRegion, optional
            Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
        dynamic_energy_axis: bool
            if True, the energy axis will for computation will be determined independently for each model, the algorithm will use the energy_axis_computation and grouped bin in order to reach the target statisctics
        dynamic_energy_axis_target_statistics: int
            the target statistics per spatial and energy bin, for spatial, it is computed based on an average and therefore doesn't guaranty is is meet in every bin
        dynamic_energy_axis_maximum_wideness_bin: float
            energy bin will not be merged if the resulting bin will be wider (in logorarithmic space) than this value
        dynamic_energy_axis_merge_zeros_low_energy: bool
            decides if empty energy bins at low energy are merged during the dynamic binning
        dynamic_energy_axis_merge_zeros_high_energy: bool
            decides if empty energy bins at high energy are merged during the dynamic binning
        cos_zenith_binning_method : str, optional
            The method used for cos zenith binning: 'min_livetime','min_n_observation'
        cos_zenith_binning_parameter_value : int, optional
            Minimum livetime (in seconds) or number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation_wobble : u.Quantity, optional
            The maximum angular separation between identified wobbles, in degrees
        zenith_binning_run_splitting : bool, optional
            If true, will split each run to match zenith binning for the base model computation
            Could be computationally expensive, especially at high zenith with a high resolution zenith binning
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV and cut as function of the zenith bins
        use_mini_irf_computation : bool, optional
            If true, in case the case of zenith interpolation or binning, each run will be divided in small subrun (the slicing is based on time).
            A model will be computed for each sub run before averaging them to obtain the final model for the run.
            Should improve the accuracy of the model, especially at high zenith angle.
        mini_irf_time_resolution : astropy.units.Quantity, optional
            Time resolution to use for mini irf used for computation of the final background model
        azimuth_east_west_splitting: bool, optional
            if true will make a separate model of east oriented and west oriented data
        interpolation_zenith_type: str, optional
            Select the type of interpolation to be used, could be either "log" or "linear", log tend to provided better results be could more easily create artefact that will cause issue
        activate_interpolation_zenith_cleaning: bool, optional
            If true, will activate the cleaning step after interpolation, it should help to eliminate artefact caused by interpolation
        interpolation_cleaning_energy_relative_threshold: float, optional
            To be considered value, the bin in energy need at least one adjacent bin with a relative difference within this range
        interpolation_cleaning_spatial_relative_threshold: float, optional
            To be considered value, the bin in space need at least one adjacent bin with a relative difference within this range
        """

        # Compute parameters for internal map
        self.offset_axis = offset_axis
        self.oversample_map = oversample_map
        spatial_resolution = np.min(
            np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1])) / self.oversample_map
        max_offset = np.max(self.offset_axis.edges)

        # Initiate upper instance
        super().__init__(energy_axis=energy_axis,
                         max_offset=max_offset,
                         spatial_resolution=spatial_resolution,
                         energy_axis_computation=energy_axis_computation,
                         exclude_regions=exclude_regions,
                         dynamic_energy_axis=dynamic_energy_axis,
                         dynamic_energy_axis_target_statistics=dynamic_energy_axis_target_statistics,
                         dynamic_energy_axis_maximum_wideness_bin=dynamic_energy_axis_maximum_wideness_bin,
                         dynamic_energy_axis_merge_zeros_low_energy=dynamic_energy_axis_merge_zeros_low_energy,
                         dynamic_energy_axis_merge_zeros_high_energy=dynamic_energy_axis_merge_zeros_high_energy,
                         cos_zenith_binning_method=cos_zenith_binning_method,
                         cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
                         initial_cos_zenith_binning=initial_cos_zenith_binning,
                         max_angular_separation_wobble=max_angular_separation_wobble,
                         zenith_binning_run_splitting=zenith_binning_run_splitting,
                         max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
                         time_resolution=time_resolution,
                         use_mini_irf_computation=use_mini_irf_computation,
                         mini_irf_time_resolution=mini_irf_time_resolution,
                         azimuth_east_west_splitting=azimuth_east_west_splitting,
                         interpolation_zenith_type=interpolation_zenith_type,
                         activate_interpolation_zenith_cleaning=activate_interpolation_zenith_cleaning,
                         interpolation_cleaning_energy_relative_threshold=interpolation_cleaning_energy_relative_threshold,
                         interpolation_cleaning_spatial_relative_threshold=interpolation_cleaning_spatial_relative_threshold)

    def create_model(self, observations: Observations) -> Background2D:
        """
        Calculate a radial acceptance map

        Parameters
        ----------
        observations : Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : Background2D
        """
        count_map_background, exp_map_background, exp_map_background_total, livetime, energy_axis_computation = self._create_base_computation_map(
            observations)
        geom = self._get_geom(energy_axis_computation)

        data_background = np.zeros((energy_axis_computation.nbin, self.offset_axis.nbin)) * u.Unit('s-1 MeV-1 sr-1')
        for i in range(self.offset_axis.nbin):
            if np.isclose(0. * u.deg, self.offset_axis.edges[i]):
                selection_region = CircleSkyRegion(center=self.center_map, radius=self.offset_axis.edges[i + 1])
            else:
                selection_region = CircleAnnulusSkyRegion(center=self.center_map,
                                                          inner_radius=self.offset_axis.edges[i],
                                                          outer_radius=self.offset_axis.edges[i + 1])
            selection_map = geom.to_image().region_mask([selection_region])
            for j in range(energy_axis_computation.nbin):
                value = u.dimensionless_unscaled * np.sum(count_map_background.data[j, :, :] * selection_map)
                value *= np.sum(exp_map_background_total.data[j, :, :] * selection_map) / np.sum(
                    exp_map_background.data[j, :, :] * selection_map)

                value /= (energy_axis_computation.edges[j + 1] - energy_axis_computation.edges[j])
                value /= 2. * np.pi * (
                            np.cos(self.offset_axis.edges[i]) - np.cos(self.offset_axis.edges[i + 1])) * u.steradian
                value /= livetime
                data_background[j, i] = value

        acceptance_map = Background2D(axes=[self.energy_axis, self.offset_axis], data=self._interpolate_bkg_to_energy_axis(data_background, energy_axis_computation))

        return acceptance_map

    def _create_base_computation_map(self, observations: Observations) -> Tuple[np.ndarray, WcsNDMap, WcsNDMap, u.Quantity, MapAxis]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a
        model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map
        exp_map_background : gammapy.map.WcsNDMap
            The exposure map corrected for exclusion regions
        exp_map_background_total : gammapy.map.WcsNDMap
            The exposure map without correction for exclusion regions
        livetime : astropy.unit.Quantity
            The total exposure time for the model
        energy_axis : gammapy.maps.MapAxis
            The energy axis used for the computation
        """

        energy_axis_computation = self.energy_axis_computation.copy()
        if self.dynamic_energy_axis:
            data_energy_distribution = np.zeros(self.energy_axis_computation.nbin, dtype=np.int64)
            for obs in observations:
                mask_event = obs.events.offset <= self.max_offset
                distrib, _ = np.histogram(obs.events.energy[mask_event], energy_axis_computation.edges)
                data_energy_distribution += distrib
            energy_axis_computation = self._compute_dynamic_energy_axis(energy_axis_computation, data_energy_distribution, self.offset_axis.nbin)

        geom = self._get_geom(energy_axis_computation)
        count_map_background = WcsNDMap(geom=geom)
        exp_map_background = WcsNDMap(geom=geom, unit=u.s)
        exp_map_background_total = WcsNDMap(geom=geom, unit=u.s)
        livetime = 0. * u.s

        for obs in observations:
            geom = WcsGeom.create(skydir=obs.pointing.fixed_icrs, npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size, frame="icrs", axes=[energy_axis_computation])
            count_map_obs, exclusion_mask = self._create_map(obs, geom, self.exclude_regions, add_bkg=False)

            exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs.counts.data = obs.observation_live_time_duration.value
            exp_map_obs_total.counts.data = obs.observation_live_time_duration.value

            for i in range(count_map_obs.counts.data.shape[0]):
                count_map_obs.counts.data[i, :, :] = count_map_obs.counts.data[i, :, :] * exclusion_mask
                exp_map_obs.counts.data[i, :, :] = exp_map_obs.counts.data[i, :, :] * exclusion_mask

            count_map_background.data += count_map_obs.counts.data
            exp_map_background.data += exp_map_obs.counts.data
            exp_map_background_total.data += exp_map_obs_total.counts.data
            livetime += obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime, energy_axis_computation
