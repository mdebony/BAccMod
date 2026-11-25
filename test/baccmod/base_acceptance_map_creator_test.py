# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: base_acceptance_map_creator_test.py
# Purpose: Unit test for the base_acceptance_map_creator.py
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from gammapy.data import Observations
from gammapy.maps import MapAxis

from baccmod import BaseAcceptanceMapCreator


@pytest.mark.usefixtures("offline_iers")
class TestSplitObservationsAzimuth:
    TIME_ATOL = 1e-3  # Seconds.
    TRANSIT = Time("2020-01-01T00:00:00", scale="utc")


    # Source are chosen to cross the border between east and west at TRANSIT time
    # RA, Dec, site, azimuth at the crossing, side before transit, pytest ID.
    ORDINARY_TARGETS = (
        (81.272328037, 80.745322727, "ctao_north", 0 * u.deg, "east", "longitude-wrapping-north-upper-east-to-west"),
        (262.574349275, 79.253512362, "ctao_north", 0 * u.deg, "west", "longitude-wrapping-north-lower-west-to-east"),
        (116.899059158, -80.223517116, "hess", 180 * u.deg, "east", "south-upper-east-to-west"),
        (295.854713683, -79.775506378, "hess", 180 * u.deg, "west", "south-lower-west-to-east"),
    )

    # Source are chosen to cross the border between east and west at TRANSIT time
    # RA, Dec, site, azimuth at the crossing, side before transit, pytest ID.
    NEAR_TARGETS = (
        (81.907183001, 28.647100279, "ctao_north", 180 * u.deg, "east", "north-zenith"),
        (116.400565072, -23.122597137, "hess", 0 * u.deg, "east", "south-zenith"),
    )

    @pytest.mark.parametrize(
        "ra,dec,location,boundary,first_side,start_offset,stop_offset",
        [
            pytest.param(*target[:5], -120, 240, id=target[5])
            for target in ORDINARY_TARGETS + NEAR_TARGETS
        ] + [
            pytest.param(
                *target[:5], -1800, 1800,
                id=target[5] + "-one-hour",
            )
            for target in NEAR_TARGETS
        ],
    )
    def test_split_observation(
        self, ra, dec, location, boundary, first_side, start_offset, stop_offset,
        make_observation, observatory_locations,
    ):
        """Check that a transit crossing produces contiguous east and west pieces.

        Cover both crossing directions at 0 and 180 degrees, including tracks near the zenith. The crossing happen at transit time

        Parameters
        ----------
        ra : float
            ICRS right ascension of the target, in degrees.
        dec : float
            ICRS declination of the target, in degrees.
        location : str
            Observatory key in ``observatory_locations``
        boundary : astropy.units.Quantity
            Expected azimuth crossing (``0 * u.deg`` or ``180 * u.deg``), supplied as
            scenario metadata; not used directly by the assertions.
        first_side : str
            Expected side before transit, either ``"east"`` or ``"west"``.
        start_offset : int
            Observation start time relative to ``TRANSIT``, in seconds.
        stop_offset : int
            Observation stop time relative to ``TRANSIT``, in seconds.
        make_observation : callable
            Pytest fixture providing a factory for in-memory fixed-sky
            tracking observations with absolute start and stop times.
        observatory_locations : dict[str, astropy.coordinates.EarthLocation]
            Pytest fixture mapping observatory keys to Earth locations.
        """
        observation = make_observation(
            ra, dec, observatory_locations[location],
            self.TRANSIT + start_offset * u.s,
            self.TRANSIT + stop_offset * u.s,
            obs_id=1,
            deadtime_fraction=0.2,
        )
        second_side = "west" if first_side == "east" else "east"
        east, west, split = BaseAcceptanceMapCreator._split_observations_azimuth(
            Observations([observation])
        )
        assert isinstance(east, Observations)
        assert isinstance(west, Observations)
        assert [item.obs_id for item in east] == [observation.obs_id]
        assert [item.obs_id for item in west] == [observation.obs_id]
        assert set(split) == {observation.obs_id}

        pieces = {"east": east[0], "west": west[0]}
        first_piece = pieces[first_side]
        last_piece = pieces[second_side]
        assert first_piece.tstart == observation.tstart
        assert last_piece.tstop == observation.tstop
        assert first_piece.tstop == last_piece.tstart
        assert abs((first_piece.tstop - self.TRANSIT).to_value(u.s)) <= self.TIME_ATOL
        assert first_piece.tstop > first_piece.tstart
        assert last_piece.tstop > last_piece.tstart
        for side, piece in pieces.items():
            azimuth = piece.get_pointing_altaz(piece.tmid).az.to_value(u.deg)
            if side == "east":
                assert azimuth < 180
            else:
                assert azimuth > 180

    @pytest.mark.parametrize(
        "ra,dec,location",
        [
            pytest.param(*target[:3], id=target[5])
            for target in (ORDINARY_TARGETS[0], ORDINARY_TARGETS[2])
        ],
    )
    @pytest.mark.parametrize(
        "start_offset,stop_offset,side",
        [(-300, -1, "east"), (1, 300, "west")],
        ids=["east-only", "west-only"],
    )
    def test_no_split_observation(
        self, ra, dec, location, start_offset, stop_offset, side,
        make_observation, observatory_locations,
    ):
        """Check an observation on one side of transit stay intact in its east or west group.

        Parameters
        ----------
        ra : float
            ICRS right ascension of the fixed tracking target, in degrees.
        dec : float
            ICRS declination of the fixed tracking target, in degrees.
        location : str
            Observatory key in ``observatory_locations``: ``"ctao_north"``
            or ``"hess"``.
        start_offset : int
            Observation start time relative to ``TRANSIT``, in seconds.
        stop_offset : int
            Observation stop time relative to ``TRANSIT``, in seconds.
        side : str
            Expected group for the entire observation, either ``"east"``
            or ``"west"``.
        make_observation : callable
            Pytest fixture providing a factory for in-memory fixed-sky
            tracking observations with absolute start and stop times.
        observatory_locations : dict[str, astropy.coordinates.EarthLocation]
            Pytest fixture mapping observatory keys to Earth locations.
        """
        observation = make_observation(
            ra, dec, observatory_locations[location],
            self.TRANSIT + start_offset * u.s,
            self.TRANSIT + stop_offset * u.s,
            obs_id=1,
        )
        east, west, split = BaseAcceptanceMapCreator._split_observations_azimuth(
            Observations([observation])
        )
        actual, other = (east, west) if side == "east" else (west, east)
        assert isinstance(east, Observations)
        assert isinstance(west, Observations)
        assert len(actual) == 1 and actual[0] is observation
        assert len(other) == 0
        assert split == {}

class DummyBaseAcceptanceMapCreator(BaseAcceptanceMapCreator):
    """ Dummy class to test base functionnalities of BaseAcceptanceMapCreator. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_base_computation_map(self):
        return None

    def create_model(self):
        return None

class TestDynamicEnergyAxis():

    numerical_r_tolerance = 0
    numerical_a_tolerance = 1e-8

    out_energy_axis = MapAxis.from_energy_bounds(100.*u.GeV, 10.*u.TeV, nbin=15, per_decade=False, name='energy')

    bamc = DummyBaseAcceptanceMapCreator(energy_axis_out=out_energy_axis,
                                         max_offset=1*u.deg,
                                         spatial_resolution=0.5*u.deg,
                                         use_dynamic_energy_axis=True)

    def test_simple_calls(self):
        """ Test `BaseAcceptanceMapCreator._compute_dynamic_energy_axis` in a simple case with known outputs """
        errors_msg=''
        errors=[]

        bamc=self.bamc

        # basic run with known output
        data=[0,0,2,20,15,4,6,3,2,1,1,0,0,0,0] # counts per energy bin
        nb_spatial_bin = 4
        bamc.dynamic_energy_axis_target_statistics = 3
        bamc.dynamic_energy_axis_maximum_wideness_bin = 2 # remove the maximum bin wideness criteria
        # expected output : more than 3X4=12 counts per energy bin which would be [0,0,22,19,13,0,0,0,0] counts
        # Number of merged bins : [0,0,2,2,4,0,0,0,0]
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)

        try:
            assert len(bamc.energy_axis_computation.edges) == 10
            assert np.all(np.isclose(bamc.energy_axis_computation.edges,
                                     bamc.energy_axis_base.edges[[0,1,2,4,6,11,12,13,14,15]],
                                     rtol=self.numerical_r_tolerance, atol=self.numerical_a_tolerance))
        except AssertionError as e:
            e.add_note('Error in the usage without maximum bin wideness.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges[[0,1,2,4,6,11,12,13,14,15]]))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Error in the usage without maximum bin wideness.\n'

        bamc.dynamic_energy_axis_maximum_wideness_bin = 0.5 # default maximum bin wideness criteria: limiting once
        # we have 15 bin for 2 decades so this corresponds to a limit of 3.75 bins merged at most
        # expected output : [0,0,22,15,13,4,0,0,0,0] counts per energy bins
        # Number of merged bins : [0,0,2,1,3,3,0,0,0,0]
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)


        try:
            assert len(bamc.energy_axis_computation.edges) == 11
            assert np.all(np.isclose(bamc.energy_axis_computation.edges,
                                     bamc.energy_axis_base.edges[[0,1,2,4,5,8,11,12,13,14,15]],
                                     rtol=self.numerical_r_tolerance, atol=self.numerical_a_tolerance))
        except AssertionError as e:
            e.add_note('Error in the usage with maximum bin wideness.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       +str(bamc.energy_axis_base.edges[[0,1,2,4,5,8,11,12,13,14,15]]))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Error in the usage with maximum bin wideness.\n'

        if len(errors)>0:
            raise ExceptionGroup('Errors in the test of BaseAcceptanceMapCreator._compute_dynamic_energy_axis:\n' +
                                 errors_msg, errors)
    def test_merging_zeros(self):
        """ Test the dynamic_energy_axis_merge_zeros_high/low_energy options of the `BaseAcceptanceMapCreator` """
        errors_msg=''
        errors=[]

        bamc=self.bamc

        # 1) Test all merging combination when there are zeros at the edges

        data=[0,0,1,1,1,1,1,1,1,1,1,0,0,0,0] # counts per energy bin
        nb_spatial_bin = 1
        bamc.dynamic_energy_axis_target_statistics = 1
        bamc.dynamic_energy_axis_maximum_wideness_bin = 2 # remove the maximum bin wideness criteria
        # All non-empty bins will be kept. Handling of the empty bins will depend on the
        # dynamic_energy_axis_merge_zeros_high/low_energy options.
        # Both are False by default so all bins should be kept
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)

        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges)
            assert np.all(bamc.energy_axis_computation.edges == bamc.energy_axis_base.edges)
        except AssertionError as e:
            e.add_note('Error when not merging zeros.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Multi zeros case, error when not merging.\n'

        bamc.dynamic_energy_axis_merge_zeros_high_energy = True # Now merge the 4 empty bins at high energy
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges) - 3
            assert np.all(np.isclose(bamc.energy_axis_computation.edges,
                                     bamc.energy_axis_base.edges[[0,1,2,3,4,5,6,7,8,9,10,11,15]],
                                     rtol=self.numerical_r_tolerance, atol=self.numerical_a_tolerance))
        except AssertionError as e:
            e.add_note('Error when merging zeros at high energy.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges[[0,1,2,3,4,5,6,7,8,9,10,11,15]]))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Multi zeros case, error when merging at high energy.\n'

        bamc.dynamic_energy_axis_merge_zeros_low_energy = True # Now merge the 4/2 empty bins at high/low energy
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges) - 3 - 1
            assert np.all(np.isclose(bamc.energy_axis_computation.edges,
                                     bamc.energy_axis_base.edges[[0,2,3,4,5,6,7,8,9,10,11,15]],
                                     rtol=self.numerical_r_tolerance, atol=self.numerical_a_tolerance))
        except AssertionError as e:
            e.add_note('Error when merging zeros at low and high energy.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges[[0,2,3,4,5,6,7,8,9,10,11,15]]))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Multi zeros case, error when merging at low and high energy.\n'

        bamc.dynamic_energy_axis_merge_zeros_high_energy = False # Now merge the 2 empty bins at low energy
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges) - 1
            assert np.all(np.isclose(bamc.energy_axis_computation.edges,
                                     bamc.energy_axis_base.edges[[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],
                                     rtol=self.numerical_r_tolerance, atol=self.numerical_a_tolerance))
        except AssertionError as e:
            e.add_note('Error when merging zeros at low energy.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges[[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]))
            errors.append(e)
            errors_msg+=f'{len(errors)}: Multi zeros case, error when merging at low energy.\n'

        # 2) Test merging when there are no zeros at the edges
        data=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # counts per energy bin
        bamc.dynamic_energy_axis_merge_zeros_low_energy = False
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges)
        except AssertionError as e:
            e.add_note('Error when not merging zeros.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges))
            errors.append(e)
            errors_msg+=f'{len(errors)}: No zero case, error when not merging zeros.\n'

        bamc.dynamic_energy_axis_merge_zeros_low_energy = True
        bamc.dynamic_energy_axis_merge_zeros_high_energy = True
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges)
        except AssertionError as e:
            e.add_note('Error when merging zeros.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges))
            errors.append(e)
            errors_msg+=f'{len(errors)}: No zero case, error when merging zeros.\n'


        # 3) Test merging when there is exactly 1 zero at the edges
        data=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0] # counts per energy bin
        bamc.dynamic_energy_axis_merge_zeros_low_energy = False
        bamc.dynamic_energy_axis_merge_zeros_high_energy = False
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges)
        except AssertionError as e:
            e.add_note('Error when not merging zeros.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges))
            errors.append(e)
            errors_msg+=f'{len(errors)}: One zero case, error when not merging zeros.\n'

        bamc.dynamic_energy_axis_merge_zeros_low_energy = True
        bamc.dynamic_energy_axis_merge_zeros_high_energy = True
        bamc._compute_dynamic_energy_axis(data, nb_spatial_bin)
        try:
            assert len(bamc.energy_axis_computation.edges) == len(bamc.energy_axis_base.edges)
        except AssertionError as e:
            e.add_note('Error when merging zeros.\n Output: '
                       + str(bamc.energy_axis_computation.edges) + '\nExpected: '
                       + str(bamc.energy_axis_base.edges))
            errors.append(e)
            errors_msg+=f'{len(errors)}: One zero case, error when merging zeros.\n'


        if len(errors)>0:
            raise ExceptionGroup('Errors in the test of dynamic_energy_axis_merge_zeros_high/low_energy:\n' +
                                 errors_msg, errors)
