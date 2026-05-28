"""Tests for SatelliteConstellation. Migrated from testSatConstelation in SRC/test_main.py."""

from datetime import UTC, datetime
import os

import pytest

from object_definitions.Constants_def import Constants
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite

START_TIME = datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC)


def _orbit_args(altitude_km):
    return [
        51.6374,
        0.0009376,
        1.5935,
        121.8064,
        341.7727,
        altitude_km * 1e3 + Constants.R_EARTH_MEAN,
        START_TIME,
    ]


def test_constellation_from_satellite_array_has_correct_size():
    satellites = [
        Satellite(1, _orbit_args(421)),
        Satellite(2, _orbit_args(441)),
        Satellite(3, _orbit_args(461)),
        Satellite(4, _orbit_args(481)),
    ]
    constellation = SatelliteConstellation(satellites)
    assert isinstance(constellation, SatelliteConstellation)
    assert constellation.no_satellites == 4


def test_constellation_from_tle_file_loads():
    """SatelliteConstellation can be constructed from a TLE file directly."""
    galileo_path = os.path.join(Constants.tle_folderpath, "GALILEO.tle")
    if not os.path.exists(galileo_path):
        pytest.skip(f"GALILEO.tle missing at {galileo_path}.")
    constellation = SatelliteConstellation(tle_file=galileo_path)
    assert isinstance(constellation, SatelliteConstellation)
    assert constellation.no_satellites > 0
