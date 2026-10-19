# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: conftest.py
# Purpose: Generic fixtures for test
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation
from astropy.utils import iers
from gammapy.data import FixedPointingInfo, Observation


@pytest.fixture
def offline_iers():
    """Use bundled IERS_B data and restore Astropy settings after the test."""
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(
        iers.IERS_B.open(iers.IERS_B_FILE)
    ):
        yield


@pytest.fixture
def observatory_locations():
    """Explicit observatory locations without a network-dependent site registry."""
    return {
        "ctao_north": EarthLocation.from_geodetic(
            lon=-17.892005 * u.deg, lat=28.762164 * u.deg, height=2200 * u.m
        ),
        "hess": EarthLocation.from_geodetic(
            lon=16.50004902672975 * u.deg,
            lat=-23.271584051253615 * u.deg,
            height=1800 * u.m,
        ),
    }


@pytest.fixture
def make_observation():
    """Factory for in-memory celestial tracking observations with absolute times."""
    def create(ra, dec, location, tstart, tstop, obs_id, deadtime_fraction=0.0):
        pointing = FixedPointingInfo.from_fits_header(
            {"OBS_MODE": "POINTING", "RA_PNT": ra, "DEC_PNT": dec}
        )
        return Observation.create(
            pointing=pointing,
            location=location,
            obs_id=obs_id,
            tstart=tstart,
            tstop=tstop,
            irfs={},
            deadtime_fraction=deadtime_fraction,
        )

    return create
