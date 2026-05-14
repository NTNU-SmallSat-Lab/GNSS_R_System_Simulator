"""Tests for ReadWriteSatData.read_tle_data. Migrated from testReadWriteData in SRC/test_main.py.

The expected values are snapshots of the TLE files committed to the repo. If those files are
refreshed (TLEs age), the snapshot values must be updated to match — this is the natural
brittleness of testing parsed TLE content against ground truth.
"""

import math
import os
from datetime import UTC, datetime

import pytest

from object_definitions.Constants_def import Constants
from object_definitions.ReadWriteSatData import ReadWriteSatData


GALILEO_PATH = os.path.join(Constants.tle_folderpath, "GALILEO.tle")
ISS_PATH = os.path.join(Constants.tle_folderpath, "ISS.tle")


@pytest.fixture(scope="module")
def galileo_satellites():
    if not os.path.exists(GALILEO_PATH):
        pytest.skip(f"GALILEO.tle missing at {GALILEO_PATH}.")
    return ReadWriteSatData.read_tle_data(GALILEO_PATH)


@pytest.fixture(scope="module")
def iss_satellites():
    if not os.path.exists(ISS_PATH):
        pytest.skip(f"ISS.tle missing at {ISS_PATH}.")
    return ReadWriteSatData.read_tle_data(ISS_PATH)


class TestGalileoTle:
    def test_returns_list_of_31_satellites(self, galileo_satellites):
        assert isinstance(galileo_satellites, list)
        assert len(galileo_satellites) == 31

    def test_arg_of_perigee_snapshot(self, galileo_satellites):
        # Snapshot of the 11th satellite's argOfPerigee at the time the TLE file was committed.
        assert galileo_satellites[10].argOfPerigee == 12.6669


class TestIssTle:
    def test_returns_list_of_one_satellite(self, iss_satellites):
        assert isinstance(iss_satellites, list)
        assert len(iss_satellites) == 1

    def test_iss_orbital_elements_snapshot(self, iss_satellites):
        # https://www.heavens-above.com/orbit_n.aspx?satid=25544 — values frozen with the TLE file.
        iss = iss_satellites[0]
        assert iss.satCatalogNumber == 25544
        assert iss.inclination == 51.638
        assert iss.eccentricity == 0.0008444
        assert iss.raan == 301.8966
        assert iss.argOfPerigee == 165.9795
        assert iss.meanAnomaly == 194.1431
        assert iss.meanMotion == 15.51387213
        assert iss.epoch == datetime(2024, 11, 12, 5, 47, 25, 801440, tzinfo=UTC)

    def test_iss_apogee_perigee_height_match_snapshot(self, iss_satellites):
        iss = iss_satellites[0]
        # Derived quantities from the snapshot orbital elements.
        semi_major_axis = (Constants.GM_EARTH_TT / (iss.meanMotion * 2 * math.pi / Constants.SEC_PER_DAY) ** 2) ** (1 / 3)
        apogee_above_earth = semi_major_axis * (1 + iss.eccentricity) - Constants.R_EARTH_MEAN
        perigee_above_earth = semi_major_axis * (1 - iss.eccentricity) - Constants.R_EARTH_MEAN
        assert apogee_above_earth == pytest.approx(425.5460901848506e3, abs=1.0)
        assert perigee_above_earth == pytest.approx(414.07776699984353e3, abs=1.0)
