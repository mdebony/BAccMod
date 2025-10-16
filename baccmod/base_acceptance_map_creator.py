# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: base_acceptance_map_creator.py
# Purpose: Base class for common functionalities in background model creation
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, Any, Dict

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D, Background3D
from gammapy.irf.background import BackgroundIRF
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, Map, MapAxis
from regions import CircleSkyRegion, EllipseSkyRegion, SkyRegion
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from .bkg_collection import BackgroundCollectionZenith, BackgroundCollection, BackgroundCollectionZenithSplitAzimuth
from .exception import BackgroundModelFormatException
from .toolbox import (compute_rotation_speed_fov,
                      get_unique_wobble_pointings,
                      get_time_mini_irf,
                      generate_irf_from_mini_irf)

logger = logging.getLogger(__name__)


class BaseAcceptanceMapCreator(ABC):

    def __init__(self,
                 energy_axis: MapAxis,
                 max_offset: u.Quantity,
                 spatial_resolution: u.Quantity,
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
        Create the class for calculating radial acceptance model.

        Parameters
        ----------
        energy_axis : gammapy.maps.geom.MapAxis
            The energy axis for the acceptance model
        max_offset : astropy.units.Quantity
            The offset corresponding to the edge of the model
        spatial_resolution : astropy.units.Quantity
            The spatial resolution of the finely binned map used for computation
        energy_axis_computation : gammapy.maps.geom.MapAxis
            The energy axis used for computation of the models, the model will then be reinterpolated on energy axis, if None, energy_axis will be used
        exclude_regions : list of regions.SkyRegion, optional
            Regions with known or putative gamma-ray emission, will be excluded from the calculation of the acceptance map
        dynamic_energy_axis: bool
            if True, the energy axis will for computation will be determined independently for each model, the algorithm will use the energy_axis_computation and grouped bin in order to reach the target statisctics
        dynamic_energy_axis_target_statistics: int
            the target statistics per spatial and energy bin, for spatial, it is computed based on an average and therefore doesn't guaranty is is meet in every bin
        dynamic_energy_axis_maximum_wideness_bin: float
            energy bin will not be merged if the resulting bin will be wider (in logarithmic space) than this value
        dynamic_energy_axis_merge_zeros_low_energy: bool
            decides if empty energy bins at low energy are merged during the dynamic binning
        dynamic_energy_axis_merge_zeros_high_energy: bool
            decides if empty energy bins at high energy are merged during the dynamic binning
        cos_zenith_binning_method : str, optional
            The method used for cos zenith binning: 'min_livetime', 'min_livetime_per_wobble', 'min_n_observation', 'min_n_observation_per_wobble'
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

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Store base parameter
        self.energy_axis = energy_axis
        self.max_offset = max_offset
        self.exclude_regions = exclude_regions

        # Store energy axis computation information
        self.energy_axis_computation = self.energy_axis if energy_axis_computation is None else energy_axis_computation
        self.dynamic_energy_axis = dynamic_energy_axis
        self.dynamic_energy_axis_target_statistics = dynamic_energy_axis_target_statistics
        self.dynamic_energy_axis_maximum_wideness_bin = dynamic_energy_axis_maximum_wideness_bin
        self.dynamic_energy_axis_merge_zeros_low_energy = dynamic_energy_axis_merge_zeros_low_energy
        self. dynamic_energy_axis_merge_zeros_high_energy = dynamic_energy_axis_merge_zeros_high_energy

        # Calculate map parameter
        self.n_bins_map = 2 * int(np.rint((self.max_offset / spatial_resolution).to(u.dimensionless_unscaled)))
        self.spatial_bin_size = self.max_offset / (self.n_bins_map / 2)
        self.center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')
        logger.info(
            'Computation will be made with a bin size of {:.3f} arcmin'.format(
                self.spatial_bin_size.to_value(u.arcmin)))

        # Store computation parameters for run splitting
        self.max_fraction_pixel_rotation_fov = max_fraction_pixel_rotation_fov
        self.time_resolution = time_resolution
        self.zenith_binning_run_splitting = zenith_binning_run_splitting

        # Store azimuth splitting data
        self.azimuth_east_west_splitting = azimuth_east_west_splitting

        # Store zenith binning parameters
        self.cos_zenith_binning_method = cos_zenith_binning_method
        self.cos_zenith_binning_parameter_value = cos_zenith_binning_parameter_value
        self.initial_cos_zenith_binning = initial_cos_zenith_binning
        self.max_angular_separation_wobble = max_angular_separation_wobble

        # Store interpolation parameters
        self.threshold_value_log_interpolation = np.finfo(np.float64).tiny
        self.interpolation_type = interpolation_zenith_type

        # Store cleaning parameters for models created from interpolation
        self.activate_interpolation_cleaning = activate_interpolation_zenith_cleaning
        self.interpolation_cleaning_energy_relative_threshold = interpolation_cleaning_energy_relative_threshold
        self.interpolation_cleaning_spatial_relative_threshold = interpolation_cleaning_spatial_relative_threshold
        self.max_cleaning_iteration = 50

        # Store mini irf computation parameters
        self.use_mini_irf_computation = use_mini_irf_computation
        self.mini_irf_time_resolution = mini_irf_time_resolution

    @abstractmethod
    def create_model(self, observations: Observations) -> BackgroundIRF:
        """
        Abstract method to calculate an acceptance map from a list of observations.

        Subclasses must implement this method to provide the specific algorithm for calculating the acceptance map.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to create the acceptance map.

        Returns
        -------
        model : gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            The background IRF calculated using the specific algorithm implemented by the subclass.
        """
        pass

    @abstractmethod
    def _create_base_computation_map(self, observations: Observation) -> Tuple[np.ndarray, WcsNDMap, WcsNDMap, u.Quantity, MapAxis]:
        """
        Abstract method to calculate maps used in acceptance computation from a list of observations.

        Subclasses must implement this method to provide the data used for calculating the acceptance map.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to create the acceptance map.

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
        pass

    def _get_geom(self, energy_axis: MapAxis = None) -> WcsGeom:
        """
        Return the gammapy geometry for maps based on the provided energy axis (if none is provided, use the default axis for computation)

        Parameters
        ----------
        energy_axis : MapAxis
            The energy axis that will be associated with spatial information to make the full geometry

        Returns
        -------
        geom: WcsGeom
            The geometry object to use to create gammapy maps

        """
        if energy_axis is None:
            energy_axis = self.energy_axis_computation
        return WcsGeom.create(skydir=self.center_map, npix=(self.n_bins_map, self.n_bins_map),
                              binsz=self.spatial_bin_size, frame="icrs", axes=[energy_axis])

    @staticmethod
    def _get_events_in_camera_frame(obs: Observation) -> SkyCoord:
        """
        Transform events and pointing of an obs from a sky frame to camera frame

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
           The observation to transform

        Returns
        -------
        events_camera_frame : astropy.coordinates.SkyCoord
           The events coordinates for reference in camera frame
        """

        if len(obs.events.time) == 0:
            # Handling the case with zero event in the observation
            camera_frame = SkyOffsetFrame(origin=AltAz(alt=obs.get_pointing_altaz(obs.tmid).alt,
                                                       az=obs.get_pointing_altaz(obs.tmid).az,
                                                       obstime=obs.tmid,
                                                       location=obs.observatory_earth_location),
                                          rotation=[0., ] * u.deg)
            return SkyCoord(lon=[]*u.deg, lat=[]*u.deg, frame=camera_frame)

        else:
            # Transform to altaz frame
            altaz_frame = AltAz(obstime=obs.events.time,
                                location=obs.observatory_earth_location)
            events_altaz = obs.events.radec.transform_to(altaz_frame)
            pointing_altaz = obs.get_pointing_icrs(obs.events.time).transform_to(altaz_frame)

            # Rotation to transform to camera frame
            camera_frame = SkyOffsetFrame(origin=AltAz(alt=pointing_altaz.alt,
                                                       az=pointing_altaz.az,
                                                       obstime=obs.events.time,
                                                       location=obs.observatory_earth_location),
                                          rotation=[0., ] * len(obs.events.time) * u.deg)

            return events_altaz.transform_to(camera_frame)

    @staticmethod
    def _transform_obs_to_camera_frame(obs: Observation) -> Observation:
        """
        Transform events and pointing of an obs from a sky frame to camera frame

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation to transform

        Returns
        -------
        obs_camera_frame : gammapy.data.observations.Observation
            The observation transformed for reference in camera frame
        """

        # Transform to altaz frame
        altaz_frame = AltAz(obstime=obs.events.time,
                            location=obs.observatory_earth_location)
        events_altaz = obs.events.radec.transform_to(altaz_frame)
        pointing_altaz = obs.get_pointing_icrs(obs.events.time).transform_to(altaz_frame)

        # Rotation to transform to camera frame
        camera_frame = SkyOffsetFrame(origin=AltAz(alt=pointing_altaz.alt,
                                                   az=pointing_altaz.az,
                                                   obstime=obs.events.time,
                                                   location=obs.observatory_earth_location),
                                      rotation=[0., ] * len(obs.events.time) * u.deg)
        events_camera_frame = events_altaz.transform_to(camera_frame)

        # Formatting data for the output
        camera_frame_events = obs.events.copy()
        camera_frame_events.table['RA'] = events_camera_frame.lon
        camera_frame_events.table['DEC'] = events_camera_frame.lat
        camera_frame_obs_info = copy.deepcopy(obs.obs_info)
        camera_frame_obs_info['RA_PNT'] = 0.
        camera_frame_obs_info['DEC_PNT'] = 0.
        obs_camera_frame = Observation(obs_id=obs.obs_id,
                                       obs_info=camera_frame_obs_info,
                                       events=camera_frame_events,
                                       gti=obs.gti,
                                       aeff=obs.aeff)
        obs_camera_frame._location = obs.observatory_earth_location

        return obs_camera_frame

    def _transform_exclusion_region_to_camera_frame(self, pointing_altaz: AltAz) -> List[SkyRegion]:
        """
        Transform the list of exclusion regions in sky frame into a list in camera frame.

        Parameters
        ----------
        pointing_altaz : astropy.coordinates.AltAz
            The pointing position of the telescope.

        Returns
        -------
        exclusion_region_camera_frame : list of regions.SkyRegion
            The list of exclusion regions in camera frame.

        Raises
        ------
        Exception
            If the region type is not supported.
        """

        camera_frame = SkyOffsetFrame(origin=pointing_altaz,
                                      rotation=[90., ] * u.deg)
        exclude_region_camera_frame = []
        for region in self.exclude_regions:
            if isinstance(region, CircleSkyRegion):
                center_coordinate = region.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(camera_frame)
                center_coordinate_camera_frame_arb = SkyCoord(ra=center_coordinate_camera_frame.lon[0],
                                                              dec=-center_coordinate_camera_frame.lat[0])
                exclude_region_camera_frame.append(CircleSkyRegion(center=center_coordinate_camera_frame_arb,
                                                                   radius=region.radius))
            elif isinstance(region, EllipseSkyRegion):
                center_coordinate = region.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(camera_frame)
                width_coordinate = center_coordinate.directional_offset_by(region.angle, region.width)
                width_coordinate_altaz = width_coordinate.transform_to(pointing_altaz)
                width_coordinate_camera_frame = width_coordinate_altaz.transform_to(camera_frame)
                center_coordinate_camera_frame_arb = SkyCoord(ra=center_coordinate_camera_frame.lon[0],
                                                              dec=-center_coordinate_camera_frame.lat[0])
                width_coordinate_camera_frame_arb = SkyCoord(ra=width_coordinate_camera_frame.lon,
                                                             dec=-width_coordinate_camera_frame.lat)
                angle_camera_frame_arb = center_coordinate_camera_frame_arb.position_angle(width_coordinate_camera_frame_arb).to(u.deg)[0]
                exclude_region_camera_frame.append(EllipseSkyRegion(center=center_coordinate_camera_frame_arb,
                                                                    width=region.width, height=region.height,
                                                                    angle=angle_camera_frame_arb))
            else:
                raise Exception(f'{type(region)} region type not supported')

        return exclude_region_camera_frame

    @staticmethod
    def _get_circumscribed_circle_for_region(region: SkyRegion)->Tuple[SkyCoord, u.Quantity]:
        """
        For a given region return the circumscribed circle for this region
        Parameters
        ----------
        region : SkyRegion
            The region for which to compute the circumscribed circle

        Returns
        -------
        center: SkyCoord
            The center of the circumscribed circle
        radius: u.Quantity
            The radius of the circumscribed circle
        """
        if isinstance(region, CircleSkyRegion):
            return region.center, region.radius
        elif isinstance(region, EllipseSkyRegion):
            return region.center, max(region.width, region.height)
        else:
            raise Exception(f'{type(region)} region type not supported')

    def _get_exclusion_regions_for_obs(self, obs:Observation)->List[SkyRegion]:
        """
        Return the list of exclusion region relevant to this observation
        Parameters
        ----------
        obs : Observation
            The region for which to compute the circumscribed circle

        Returns
        -------
        final_exclusion_list: CircleSkyRegion

        """
        final_exclusion_list = []
        for region in self.exclude_regions:
            center, radius = self._get_circumscribed_circle_for_region(region)
            if center.separation(obs.get_pointing_icrs(obs.tmid)) < (radius + self.max_offset*np.sqrt(2))*1.1:
                final_exclusion_list.append(region)

        return final_exclusion_list

    def _create_map(self,
                    obs: Observation,
                    geom: WcsGeom,
                    exclude_regions: List[SkyRegion],
                    add_bkg: bool = False
                    ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create a map and the associated exclusion mask based on the given geometry and exclusion region.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        geom : gammapy.maps.WcsGeom
            The geometry for the maps.
        exclude_regions : list of regions.SkyRegion
            The list of exclusion regions.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=self.max_offset)

        geom_image = geom.to_image()
        exclusion_mask = ~geom_image.region_mask(exclude_regions) if len(exclude_regions) > 0 else ~Map.from_geom(
            geom_image)

        map_obs = maker.run(MapDataset.create(geom=geom), obs)
        map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def _create_sky_map(self,
                        obs: Observation,
                        add_bkg: bool = False
                        ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create the sky map and the associated exclusion mask based on the observation and the exclusion regions.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        geom_obs = WcsGeom.create(skydir=obs.get_pointing_icrs(obs.tmid),
                                  npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size,
                                  frame="icrs",
                                  axes=[self.energy_axis_computation])
        map_obs, exclusion_mask = self._create_map(obs, geom_obs, self.exclude_regions, add_bkg=add_bkg)

        return map_obs, exclusion_mask

    def _compute_time_intervals_based_on_fov_rotation(self, obs: Observation) -> Time:
        """
        Calculate time intervals based on the rotation of the Field of View (FoV).

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Determine time interval for cutting the obs as function of the rotation of the Fov
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution).to_value(u.dimensionless_unscaled))))
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)
        rotation_speed_fov = compute_rotation_speed_fov(time_axis, obs.get_pointing_icrs(obs.tmid),
                                                        obs.observatory_earth_location)
        rotation_fov = cumulative_trapezoid(x=time_axis.unix_tai,
                                            y=rotation_speed_fov.to_value(u.rad / u.s),
                                            initial=0.) * u.rad
        distance_rotation_fov = rotation_fov.to_value(u.rad) * np.pi * self.max_offset
        node_obs = distance_rotation_fov // (self.spatial_bin_size * self.max_fraction_pixel_rotation_fov)
        change_node = node_obs[2:] != node_obs[1:-1]
        time_interval = Time([obs.tstart, ] + [time_axis[1:-1][change_node], ] + [obs.tstop, ])

        return time_interval

    def _compute_time_intervals_based_on_zenith_bin(self, obs: Observation, edge_zenith_bin: u.Quantity) -> Time:
        """
        Calculate time intervals based on an input zenith binning

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.
        edge_zenith_bin : astropy.units.Quantity
            The edge of the bins used for zenith binning

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Create the time axis
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution).to_value(u.dimensionless_unscaled))))
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)

        # Compute the zenith for each evaluation time
        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            altaz_coordinates = obs.get_pointing_altaz(time_axis)
            zenith_values = altaz_coordinates.zen
            if np.any(zenith_values < np.min(edge_zenith_bin)) or np.any(zenith_values > np.max(edge_zenith_bin)):
                logger.error('Run with zenith value outside of the considered range for zenith binning')

        # Split the time interval to transition between zenith bin
        id_bin = np.digitize(zenith_values, edge_zenith_bin)
        bin_transition = id_bin[2:] != id_bin[1:-1]
        time_interval = Time([obs.tstart, ] + [time_axis[1:-1][bin_transition], ] + [obs.tstop, ])

        return time_interval

    def _compute_dynamic_energy_axis(self, base_energy_axis: MapAxis, data: np.ndarray, nb_spatial_bin: int) -> MapAxis:
        """
        Compute a new energy axis from the base one to better have a more uniform number of event in each bin

        Parameters
        ----------
        base_energy_axis: gammapy.maps.MapAxis
            the base energy axis
        data : np.ndarray
            the count distribution only binned in energy (using base_energy_axis binning)
        nb_spatial_bin : int
            the number of spatial bin covering the data_energy_distribution

        Returns
        -------
        final_energy_axis: gammapy.maps.MapAxis
            the optimal energy axis
        """

        # Relative numerical tolerance for the energy bin width limit
        r_tol = 1 + 1e-7

        edges_energy_axis = base_energy_axis.edges

        cumsumdata = np.cumsum(data)/nb_spatial_bin
        rev_cumsumdata = np.cumsum(data[::-1])[::-1]/nb_spatial_bin
        # Evaluate the bins with only zeros at the start and end of the distribution
        zeros_lowE = np.nonzero(cumsumdata==0)[0]
        zeros_highE = np.nonzero(rev_cumsumdata==0)[0]

        log_edges = np.log10(edges_energy_axis.to_value(u.TeV))


        min_i = len(zeros_lowE)
        i0 = len(edges_energy_axis) - len(zeros_highE)-1
        i = i0
        indexes_edges = [len(edges_energy_axis)-1]

        # Evaluate bin edges fulfilling the dynamic criteria
        while i > min_i:
            # Index for the mean count criteria
            j_counts = np.sum(rev_cumsumdata >= self.dynamic_energy_axis_target_statistics)-1
            # Index for the max bin width criteria
            j_maxw = np.sum((log_edges[i] - log_edges[:i-1]) > self.dynamic_energy_axis_maximum_wideness_bin * r_tol)
            if j_counts < min_i and j_maxw < min_i:
                # Handle the last bin before the lower zeros, adding its lower edge and eventually merging with the previous bin
                if i!=i0 and log_edges[indexes_edges[-2]] - log_edges[min_i] <= self.dynamic_energy_axis_maximum_wideness_bin * r_tol:
                    indexes_edges.pop(-1)
                indexes_edges.append(min_i)
                i = min_i-1
            else:
                # Pick the first fulfilled criteria
                if j_counts >= j_maxw:
                    j = j_counts
                else:
                    j = j_maxw
                    logger.warning(
                        'Dynamic energy binning is unable to reach target statistics due to bin maximum bin wideness')
                indexes_edges.append(j)
                rev_cumsumdata -= rev_cumsumdata[j]
                i = j
        # Add empty bins
        if self.dynamic_energy_axis_merge_zeros_low_energy and len(zeros_lowE)>0:
            zeros_lowE = np.array([0], dtype=int)
        if self.dynamic_energy_axis_merge_zeros_high_energy and len(zeros_highE)>0:
            zeros_highE = np.array([i0], dtype=int)
        indexes_edges=np.sort(np.concatenate([zeros_highE, indexes_edges, zeros_lowE], dtype=int))
        return MapAxis.from_energy_edges(edges_energy_axis[np.array(indexes_edges, dtype=int)], name='energy')

    @staticmethod
    def _split_observations_azimuth(observations: Observations) -> Tuple[Observations, Observations, Dict[int, Dict[str, Any]]]:
        """
        Split observations between east and west pointing ones, if a given observation cross the line, split it into two observations

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The observations to process

        Returns
        -------
        east_observations : gammapy.data.observations.Observations
            Observations pointing east
        west_observations : gammapy.data.observations.Observations
            Observations pointing west
        split_obs : Dict[int, Dict[str, Any]]
            Each entry store the time of each split (east and west)
        """

        east_observations = Observations()
        west_observations = Observations()
        split_obs = {}

        for obs in observations:
            az_start = obs.get_pointing_altaz(obs.tstart).az
            az_end = obs.get_pointing_altaz(obs.tstop).az
            if az_start < 180.*u.deg and az_end < 180.*u.deg:
                east_observations.append(obs)
            elif az_start > 180.*u.deg and az_end > 180.*u.deg:
                west_observations.append(obs)
            # Case were the observation pass across the line and need to be split
            else:
                # Determine if the observations start east or west
                if az_start < 180.*u.deg:
                    east_west = True
                else:
                    east_west = False

                # Determine if the line is crossed at az=0 or az=180
                time_eval = np.linspace(obs.tstart, obs.tstop, num=100)
                az_eval = obs.get_pointing_altaz(time_eval).az
                if np.min(np.abs(az_eval-180.*u.deg)) < np.min(az_eval):
                    az_split = 180.*u.deg
                else:
                    az_split = 0.*u.deg

                # Search time for the split
                def wrap_angle(angle):
                    mask = angle > 270.*u.deg
                    angle_final = angle.copy()
                    angle_final[mask] -= 360*u.deg
                    return angle_final
                def root_function(t):
                    tt = Time(t, format = 'unix')
                    return (wrap_angle(obs.get_pointing_altaz(tt).az)-az_split).to_value(u.deg)
                res = root_scalar(root_function, method='brentq', bracket=[obs.tstart.unix, obs.tstop.unix])
                t_split = Time(res.root, format='unix')
                if east_west:
                    east_observations.append(obs.select_time([obs.tstart, t_split]))
                    west_observations.append(obs.select_time([t_split, obs.tstop]))
                    split_obs[obs.obs_id] = {'east_time': t_split-obs.tstart, 'west_time': obs.tstop-t_split}
                else:
                    west_observations.append(obs.select_time([obs.tstart, t_split]))
                    east_observations.append(obs.select_time([t_split, obs.tstop]))
                    split_obs[obs.obs_id] = {'west_time': t_split-obs.tstart, 'east_time': obs.tstop-t_split}

        return east_observations, west_observations, split_obs

    def _cluster_observations(self,
                              observations: Observations
                              ) -> Tuple[dict[str, Observations], Dict[int, Dict[str, Any]]]:
        """
        Perform observation splitting and gather them into named subsets.
        Parameters
        ----------
        observations: gammapy.data.observations.Observations
            Observations to be clustered based on required criteria such as azimuth splitting
        Returns
        -------
        observations_dict: dict of str:Observations
            Subset of observations identified with a keyword
        split_obs: dict
            Meta information potentially used to merge observations later
        """
        if self.azimuth_east_west_splitting:
            east_observations, west_observations, split_obs = self._split_observations_azimuth(observations)
            observations_dict = {'east':east_observations, 'west':west_observations}
        else:
            observations_dict={'all':observations}
            split_obs = None
        return observations_dict, split_obs

    def _create_model_cos_zenith_binned(self,
                                        observations: Observations
                                        ) -> BackgroundCollectionZenith:
        """
        Calculate a model for each cos zenith bin

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the background model

        Returns
        -------
        background : BackgroundCollectionZenith
            The collection of background model with the zenith associated to each model

        """

        # Determine binning method. Convention : per_wobble methods have negative values
        methods = {'min_livetime': 1, 'min_livetime_per_wobble': -1, 'min_n_observation': 2,
                   'min_n_observation_per_wobble': -2}
        try:
            i_method = methods[self.cos_zenith_binning_method]
        except KeyError:
            logger.error(f" KeyError : {self.cos_zenith_binning_method} not a valid zenith binning method.\nValid "
                         f"methods are {[*methods]}")
            raise
        per_wobble = i_method < 0

        # Initial bins edges
        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        zenith_bin = np.rad2deg(np.arccos(cos_zenith_bin)) * u.deg

        # Cut observations if requested
        if self.zenith_binning_run_splitting:
            if abs(i_method) == 2:
                logger.warning('Using a zenith binning requirement based on n_observation while using run splitting '
                               'is not recommended and could lead to poor models. We recommend switching to a binning '
                               'requirement based on livetime.')
            compute_observations = Observations()
            for obs in observations:
                time_interval = self._compute_time_intervals_based_on_zenith_bin(obs, zenith_bin)
                for i in range(len(time_interval) - 1):
                    compute_observations.append(obs.select_time(Time([time_interval[i], time_interval[i + 1]])))
        else:
            compute_observations = observations

        # Determine initial bins values
        cos_zenith_observations = np.array(
            [np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in compute_observations])
        livetime_observations = np.array(
            [obs.observation_live_time_duration.to_value(u.s) for obs in compute_observations])

        # Select the quantity used to count observations
        if i_method in [-1, 1]:
            cut_variable_weights = livetime_observations
        elif i_method in [-2, 2]:
            cut_variable_weights = np.ones(len(cos_zenith_observations), dtype=int)

        # Gather runs per separation angle or all together. Define the minimum multiplicity (-1) to create a zenith bin.
        if per_wobble:
            wobble_observations = np.array(
                get_unique_wobble_pointings(compute_observations, self.max_angular_separation_wobble))
            multiplicity_wob = 1
        else:
            wobble_observations = np.full(len(cos_zenith_observations), 'any', dtype=np.object_)
            multiplicity_wob = 0

        cumsum_variable = {}
        for wobble in np.unique(wobble_observations):
            # Create an array of cumulative weight of the selected variable vs cos(zenith)
            cumsum_variable[wobble] = np.cumsum(np.histogram(cos_zenith_observations[wobble_observations == wobble],
                                                             bins=cos_zenith_bin,
                                                             weights=cut_variable_weights[
                                                                 wobble_observations == wobble])[0])
        # Initiate the list of index of selected zenith bin edges
        zenith_selected = [0]

        i = 0
        n = len(cos_zenith_bin) - 2

        while i < n:
            # For each wobble, find the index of the first zenith which fulfills the zd binning criteria if any
            # Then concatenate and sort the results for all wobbles
            candidate_i_per_wobble = [np.nonzero(cum_cut_variable >= self.cos_zenith_binning_parameter_value)[0][:1]
                                      for cum_cut_variable in cumsum_variable.values()]  # nonzero is assumed sorted
            candidate_i = np.sort(np.concatenate(candidate_i_per_wobble))

            if len(candidate_i) > multiplicity_wob:
                # If the criteria is fulfilled save the correct index.
                # The first and only candidate_i in the non-per_wobble case and the second in the per_wobble case.
                i = candidate_i[multiplicity_wob]
                zenith_selected.append(i + 1)
                for wobble in np.unique(wobble_observations):
                    # Reduce the cumulative sum by the value at the selected index for the next iteration
                    cumsum_variable[wobble] -= cumsum_variable[wobble][i]
            else:
                # The zenith bin creation criteria is not fulfilled, the last bin edge is set to the end of the
                # cos(zenith) array
                if i == 0:
                    zenith_selected.append(n + 1)
                else:
                    zenith_selected[-1] = n + 1
                i = n
        cos_zenith_bin = cos_zenith_bin[zenith_selected]

        # Associate each observation to the correct bin
        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in compute_observations:
            binned_observations[np.digitize(np.cos(obs.get_pointing_altaz(obs.tmid).zen), cos_zenith_bin) - 1].append(
                obs)

        # Determine the center of the bin (weighted as function of the livetime of each observation)
        bin_center = []
        for i in range(len(binned_observations)):
            weighted_cos_zenith_bin_per_obs = []
            livetime_per_obs = []
            for obs in binned_observations[i]:
                weighted_cos_zenith_bin_per_obs.append(
                    obs.observation_live_time_duration * np.cos(obs.get_pointing_altaz(obs.tmid).zen))
                livetime_per_obs.append(obs.observation_live_time_duration)
            bin_center.append(np.sum([wcos.value for wcos in weighted_cos_zenith_bin_per_obs]) / np.sum(
                [livet.value for livet in livetime_per_obs]))

        logger.info(f"cos zenith bin edges: {list(np.round(cos_zenith_bin, 2))}")
        logger.info(f"cos zenith bin centers: {list(np.round(bin_center, 2))}")
        logger.info(f"observation per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin)[0])}")
        logger.info(f"livetime per bin [s]: " +
                    f"{list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=livetime_observations)[0].astype(int))}")
        if per_wobble:
            wobble_observations_bool_arr = [(np.array(wobble_observations.tolist()) == wobble) for wobble in
                                            np.unique(np.array(wobble_observations))]
            livetime_observations_and_wobble = [np.array(livetime_observations) * wobble_bool for wobble_bool in
                                                wobble_observations_bool_arr]
            for i, wobble in enumerate(np.unique(np.array(wobble_observations))):
                logger.info(
                    f"{wobble} observation per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=1 * wobble_observations_bool_arr[i])[0])}")
                logger.info(
                    f"{wobble} livetime per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=livetime_observations_and_wobble[i])[0].astype(int))}")

        # Compute the model for each bin
        binned_model = []
        for i, binned_obs in enumerate(binned_observations):
            logger.info(f"Creating model for the bin at cos zenith = {np.round(bin_center[i], 2)}°")
            binned_model.append(self.create_model(binned_obs))

        dict_binned_model = {}
        for i in range(len(binned_model)):
            dict_binned_model[np.rad2deg(np.arccos(bin_center[i]))] = binned_model[i]
        # Create the dict for output of the function
        collection_binned_model = BackgroundCollectionZenith(bkg_dict=dict_binned_model,
                                                             interpolation_type=self.interpolation_type,
                                                             threshold_value_log_interpolation=self.threshold_value_log_interpolation,
                                                             activate_interpolation_cleaning=self.activate_interpolation_cleaning,
                                                             interpolation_cleaning_energy_relative_threshold=self.interpolation_cleaning_energy_relative_threshold,
                                                             interpolation_cleaning_spatial_relative_threshold=self.interpolation_cleaning_spatial_relative_threshold)
        return collection_binned_model

    def _create_acceptance_model(self,
                                 off_observations: dict[str, Observations],
                                 zenith_binning: bool = False,
                                 zenith_interpolation: bool = False
                                 ) -> Union[BackgroundIRF,BackgroundCollectionZenith]:
        """
        A function to create a combined background model for observations previously separated in subsets.
        Parameters
        ----------
        off_observations: gammapy.data.observations.Observations
        zenith_binning: bool
        zenith_interpolation: bool

        Returns
        -------
        model : BackgroundIRF or BackgroundCollection
        """
        models={}
        for key in off_observations.keys():
            if zenith_interpolation or zenith_binning:
                models[key] = self._create_model_cos_zenith_binned(observations=off_observations[key])
            else:
                models[key] = self.create_model(observations=off_observations[key])
        if self.azimuth_east_west_splitting:
            model=BackgroundCollectionZenithSplitAzimuth(bkg_east=models['east'], bkg_west=models['west'],
                                                         interpolation_type=self.interpolation_type,
                                                         threshold_value_log_interpolation=self.threshold_value_log_interpolation,
                                                         activate_interpolation_cleaning=self.activate_interpolation_cleaning,
                                                         interpolation_cleaning_energy_relative_threshold=self.interpolation_cleaning_energy_relative_threshold,
                                                         interpolation_cleaning_spatial_relative_threshold=self.interpolation_cleaning_spatial_relative_threshold
                                                         )
        elif len(models) == 1:
            model=models['all']
        else:
            raise Exception('Supports only Azimuth splitting. '
                            'keys of the Observations dict should be "all" or "east" and "west"')
        return model

    def _apply_mini_irf(self, obs, model, interp):
        """
        Creates the background irf for an observation by averaging the IRF evaluated in smaller time intervals.
        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation to which the acceptance model will be applied
        model : BackgroundCollectionZenith or BackgroundCollectionZenithSplitAzimuth
            The method will use this model for the acceptance map
        interp : bool
            Determine if we use IRF interpolation or the closest model at each time.

        Returns
        -------
        background : gammapy.irf.background.BackgroundIRF
            The background IRF for the observation.
        """
        # Determine model type and axes
        type_model = model.type_model
        axes_model = model.axes_model
        unit_model = model.unit_model
        shape_model = model.shape_model

        if not model.consistent_bkg:
            raise Exception('Inconsistent background IRF incompatible with the use of mini-IRF computation')
        evaluation_time, observation_time = get_time_mini_irf(obs, self.mini_irf_time_resolution)

        data_obs_all = np.zeros(tuple([len(evaluation_time), ]+list(shape_model))) * unit_model
        for i in range(len(evaluation_time)):
            if not interp:
                selected_model_bin = model.get_closest_model(zenith=obs.get_pointing_altaz(evaluation_time[i]).zen,
                                                             azimuth=obs.get_pointing_altaz(evaluation_time[i]).az)
            else:
                selected_model_bin = model.get_interpolated_model(zenith=obs.get_pointing_altaz(evaluation_time[i]).zen,
                                                                  azimuth=obs.get_pointing_altaz(evaluation_time[i]).az)
            data_obs_all[i] = selected_model_bin.data * selected_model_bin.unit

        data_obs = generate_irf_from_mini_irf(data_obs_all, observation_time)

        if type_model is Background2D:
            return Background2D(axes=axes_model, data=data_obs)
        elif type_model is Background3D:
            return Background3D(axes=axes_model, data=data_obs, fov_alignment=model.fov_alignment)
        else:
            raise Exception('Unknown background format')

    @staticmethod
    def _apply_model_all_run(observations: Observations,
                             model: Union[BackgroundIRF, BackgroundCollectionZenithSplitAzimuth]
                             ) -> dict[int, BackgroundIRF]:
        """
        Method to calculate an acceptance map associated to each observation from a list using a single model.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations requiring a background IRF.
        model : gammapy.irf.background.BackgroundIRF or BackgroundCollectionZenithSplitAzimuth
            The method will use this model for the acceptance map. It is possible to provide one model for each of east
            and west pointing using a BackgroundCollectionZenithSplitAzimuth

        Returns
        -------
        acceptance_map : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            The acceptance map for each run.
        """

        acceptance_map = {}
        for obs in observations:
            if isinstance(model, BackgroundIRF):
                acceptance_map[obs.obs_id] = model
            elif isinstance(model, BackgroundCollectionZenithSplitAzimuth):
                acceptance_map[obs.obs_id] = model.get_closest_model(zenith=45 * u.deg,
                                                                     azimuth=obs.get_pointing_altaz(obs.tmid).az)
            else:
                error_message = 'The models should be provided as a BackgroundIRF or BackgroundCollection object'
                logger.error(error_message)
                raise BackgroundModelFormatException(error_message)
        return acceptance_map

    def _apply_model_cos_zenith_binned(
            self,
            observations: Observations,
            model: Union[BackgroundCollectionZenith, BackgroundCollectionZenithSplitAzimuth],
    ) -> dict[int, BackgroundIRF]:
        """
        Calculate an acceptance map per run using cos zenith binned model.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations to which the acceptance model will be applied
        model : BackgroundCollectionZenith or BackgroundCollectionZenithSplitAzimuth
            The method will use this model for the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dictionary with the background IRF for each observation id.

        """

        if not isinstance(model, BackgroundCollection):
            error_message = 'The models should be provided as a BackgroundCollection object'
            logger.error(error_message)
            raise BackgroundModelFormatException(error_message)

        # Find the closest model for each observation and associate it to each observation
        acceptance_map = {}
        for obs in observations:
            if self.use_mini_irf_computation:
                acceptance_map[obs.obs_id] = self._apply_mini_irf(obs, model, interp=False)
            else:
                acceptance_map[obs.obs_id] = model.get_closest_model(zenith=obs.get_pointing_altaz(obs.tmid).zen,
                                                                     azimuth=obs.get_pointing_altaz(obs.tmid).az)
        return acceptance_map

    def _apply_model_cos_zenith_interpolated(self,
                                             observations: Observations,
                                             model: BackgroundCollectionZenith
                                             ) -> dict[int, BackgroundIRF]:
        """
        Calculate an acceptance map per run using a cos zenith binned model and interpolation

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations to which the acceptance model will be applied
        model : BackgroundCollectionZenith
            The method will use this model for the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dictionary with the background IRF for each observation id.

        """

        # If needed produce the zenith binned model
        if not isinstance(model, BackgroundCollection):
            error_message = 'The models should be provided as a BackgroundCollection object'
            logger.error(error_message)
            raise BackgroundModelFormatException(error_message)

        # Interpolate the model for each observation
        acceptance_map = {}
        # Perform the interpolation
        for obs in observations:
            if self.use_mini_irf_computation:
                acceptance_map[obs.obs_id] = self._apply_mini_irf(obs, model, interp=True)
            else:
                acceptance_map[obs.obs_id] = model.get_interpolated_model(zenith=obs.get_pointing_altaz(obs.tmid).zen,
                                                                          azimuth=obs.get_pointing_altaz(obs.tmid).az)

        return acceptance_map

    def _apply_model_per_observation(self,
                                     observations: Observations,
                                     model: Union[BackgroundCollectionZenith, BackgroundIRF],
                                     zenith_binning: bool = False,
                                     zenith_interpolation: bool = False,
                                     ) -> dict[int, BackgroundIRF]:
        """
        Apply an existing model to observations. Similar to giving a base_model to create_acceptance_map_per_observation
        but with reduced fonctionnalities.

        Parameters
        ----------
        observations: gammapy.data.observations.Observations
            The collection of observations to which the acceptance model will be applied
        model: BackgroundIRF or BackgroundCollectionZenith
        zenith_binning: bool, optional
        zenith_interpolation: bool, optional

        Returns
        -------
        acceptance_map: dict
            A dictionary with the background IRF for each observation id.
        """
        if zenith_interpolation:
            acceptance_map = self._apply_model_cos_zenith_interpolated(observations=observations, model=model)
        elif zenith_binning:
            acceptance_map = self._apply_model_cos_zenith_binned(observations=observations, model=model)
        else:
            acceptance_map = self._apply_model_all_run(observations=observations, model=model)
        return acceptance_map

    def _interpolate_bkg_to_energy_axis(self, data_bkg: u.Quantity, energy_axis_computation: MapAxis):
        """
            Compute the final background model from the provided data by interpolating the energy axis used for
            computation to the ones for the model

            Parameters
            ----------
            data_bkg : u.Quantity
                The data cube of the background model. The energy axis needs to be the first one.
            energy_axis_computation : gammapy.maps.geom.MapAxis
                The energy axis used for computation

            Returns
            -------
            final_data_bkg : u.Quantity
                The background data cube interpolated to the energy axis of the output model
        """

        # Return the provided bkg data if energy axis are matching
        if len(energy_axis_computation.edges) == len(self.energy_axis.edges) and np.all(energy_axis_computation.edges == self.energy_axis.edges):
            logger.info('Identical computation energy axis and model energy axis, no interpolation required')
            return data_bkg

        logger.info('Interpolating from computation energy axis to model energy axis')

        unit = data_bkg.unit
        raw_log_data = np.log10(data_bkg.value+self.threshold_value_log_interpolation)
        mask_zero_input = np.isclose(0., data_bkg.value, atol=100.*self.threshold_value_log_interpolation)
        min_value = np.min(data_bkg.value[~mask_zero_input])
        max_value = np.max(data_bkg.value[~mask_zero_input])

        interp_func = interp1d(x=np.log10(energy_axis_computation.center.to_value(u.TeV)),
                               y=raw_log_data,
                               axis=0,
                               fill_value='extrapolate')
        raw_log_final_data_bkg = interp_func(np.log10(self.energy_axis.center.to_value(u.TeV)))
        raw_final_data_bkg = 10**raw_log_final_data_bkg
        mask_zero_final = raw_log_final_data_bkg < (np.log10(min_value) - max(2, np.log10(max_value)-np.log10(min_value))) # All values that are smaller than 2 orders of magnitude than the minimum non zero input data (or the difference in order of magnitude between minimum and maximum) will be considered as zero
        raw_final_data_bkg[mask_zero_final] = 0.

        return raw_final_data_bkg * unit

    @staticmethod
    def _merge_model_azimuth(east_models: Dict[int, BackgroundIRF], west_models: Dict[int, BackgroundIRF], split_obs: Dict[int, Dict[str, Any]]) -> Dict[int, BackgroundIRF]:
        """
        Merge model, if a given observation cross the line, split it into two observations

        Parameters
        ----------
        east_models : Dict[int, BackgroundIRF]
            The collection of models for east pointing observations, keys are obs id
        west_models : Dict[int, BackgroundIRF]
            The collection of models for west pointing observations, keys are obs id
        split_obs : Dict[int, Dict[str, Any]]
            Each dictionary store an entry for an obs split in two (east and west)

        Returns
        -------
        merged_models: Dict[int, BackgroundIRF]
            The merged collection of models, keys are obs id
        """

        merged_models = {}
        list_split_obs = [k for k in split_obs.keys()]

        for k in east_models:
            if k in list_split_obs:
                data = (east_models[k].data * split_obs[k]['east_time'] + west_models[k].data * split_obs[k]['west_time'])/(split_obs[k]['east_time'] + split_obs[k]['west_time'])
                if type(east_models[k]) is Background2D:
                    merged_models[k] = Background2D(axes=east_models[k].axes, data=data.to_value(u.dimensionless_unscaled)*east_models[k].unit)
                elif type(east_models[k]) is Background3D:
                    merged_models[k] = Background3D(axes=east_models[k].axes, data=data.to_value(u.dimensionless_unscaled)*east_models[k].unit)
                else:
                    raise Exception('Unknown background IRF')
            else:
                merged_models[k] = east_models[k]
        for k in west_models:
            if k in list_split_obs:
                # Already merged nothing more is needed
                continue
            else:
                merged_models[k] = west_models[k]

        return merged_models

    def _normalise_model_per_run(self,
                                 observations: Observations,
                                 acceptance_map: dict[int, BackgroundIRF]) -> dict[int, BackgroundIRF]:
        """
        Normalises the acceptance model associated to each run to the events rates in the run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations
        acceptance_map :dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and an associated background model for each observation.
            These models will be normalised.
        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dictionary with the re-normalised background IRF for each observation id.
        """

        normalised_acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = acceptance_map[id_observation]

            # Fit the background model
            logger.info('Fit to model to run ' + str(id_observation))
            map_obs, exclusion_mask = self._create_sky_map(modified_observation, add_bkg=True)
            maker_FoV_background = FoVBackgroundMaker(method='fit', exclusion_mask=exclusion_mask)
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters['name'] == 'norm']['value'][0]

            if norm_background < 0.:
                logger.error(
                    'Invalid normalisation value for run ' + str(id_observation) + ' : ' + str(
                        norm_background) + ', normalisation set back to 1')
                norm_background = 1.
            elif norm_background > 1.5 or norm_background < 0.5:
                logger.warning(
                    'High correction of the background normalisation for run ' + str(id_observation) + ' : ' + str(
                        norm_background))

            # Apply normalisation to the background model
            normalised_acceptance_map[id_observation] = copy.deepcopy(acceptance_map[id_observation])
            normalised_acceptance_map[id_observation].data = normalised_acceptance_map[
                                                                 id_observation].data * norm_background

        return normalised_acceptance_map

    def create_acceptance_model(self,
                                off_observations: Observations,
                                zenith_binning: bool = False
                                ) -> Union[BackgroundIRF,BackgroundCollectionZenith]:
        """
        A high level function to create a background model from observations.
        Parameters
        ----------
        off_observations: gammapy.data.observations.Observations
        zenith_binning: bool

        Returns
        -------
        model : BackgroundIRF or BackgroundCollection
        """

        off_observations_sets, _ = self._cluster_observations(off_observations)
        return self._create_acceptance_model(off_observations_sets, zenith_binning)


    def create_acceptance_map_per_observation(self,
                                              observations: Observations,
                                              zenith_binning: bool = False,
                                              zenith_interpolation: bool = False,
                                              runwise_normalisation: bool = True,
                                              off_observations: Observations = None,
                                              base_model: Union[BackgroundCollectionZenith, BackgroundIRF] = None,
                                              ) -> dict[int, BackgroundIRF]:
        """
        Calculate an acceptance map with the norm adjusted for each run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations to which the acceptance model will be applied
        zenith_binning : bool, optional
            If true the acceptance maps will be generated using zenith binning
        zenith_interpolation : bool, optional
            If true the acceptance maps will be generated using zenith binning and interpolation
        runwise_normalisation : bool, optional
            If true the acceptance maps will be normalised runwise to the observations
        off_observations : gammapy.data.observations.Observations
            The collection of observations used to generate the acceptance map, if None will be the observations provided as target
            Will be ignored if a base_model parameter is provided
        base_model : gammapy.irf.background.BackgroundIRF or BackgroundCollectionZenith
            If you have already a precomputed model, the method will use this model as base for the acceptance map instead of computing it from the data
            In the case of a zenith dependant model, you should provide a BackgroundCollectionZenith object

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """
        if base_model is not None and off_observations is not None:
            logger.warning('The off observations provided will be ignored as a base model has been provided.')
            off_observations = None
        observations_dict, split_obs = self._cluster_observations(observations)
        if off_observations is not None:
            off_observations_dict, _ = self._cluster_observations(off_observations)
        else:
            off_observations_dict = observations_dict

        model = base_model or self._create_acceptance_model(off_observations=off_observations_dict,
                                                            zenith_binning=zenith_binning,
                                                            zenith_interpolation=zenith_interpolation)
        acceptance_map_set = {}
        for key in observations_dict.keys():
            acceptance_map_set[key] = self._apply_model_per_observation(observations=observations_dict[key],
                                                                        zenith_binning=zenith_binning,
                                                                        zenith_interpolation=zenith_interpolation,
                                                                        model=model)
        if self.azimuth_east_west_splitting:
            acceptance_map = self._merge_model_azimuth(acceptance_map_set['east'],
                                                       acceptance_map_set['west'],
                                                       split_obs)
        else:
            acceptance_map = acceptance_map_set['all']

        if runwise_normalisation:
            acceptance_map = self._normalise_model_per_run(observations, acceptance_map)

        return acceptance_map
