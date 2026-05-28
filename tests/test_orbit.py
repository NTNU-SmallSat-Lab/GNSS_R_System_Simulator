"""Tests for Orbit_specification. Migrated from testObjOrbit in SRC/test_main.py."""

from datetime import UTC, datetime

from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit_specification

APOGEE_M = 421e3 + Constants.R_EARTH_MEAN
ORBIT_ARGS = (51.6374, 0.0009376, 1.5935, 121.8064, 341.7727, APOGEE_M)
START_TIME = datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC)


class TestOrbitSpecification:
    def test_constructed_without_epoch_has_none_epoch_str(self):
        orbit = Orbit_specification(*ORBIT_ARGS)
        assert isinstance(orbit, Orbit_specification)
        assert orbit.epoch_str is None

    def test_constructed_with_epoch_has_expected_tle_epoch_string(self):
        orbit = Orbit_specification(*ORBIT_ARGS, START_TIME)
        assert isinstance(orbit, Orbit_specification)
        # TLE epoch format: YYDDD.DDDDDDDD for 2024-10-31 05:05:57 UTC
        assert orbit.epoch_str == "24305.21246528"
