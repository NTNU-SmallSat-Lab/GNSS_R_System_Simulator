"""Tests for Satellite construction and basic state. Migrated from testObjSatellite
(and testFreeFlyerConnection, which was misnamed and only smoke-tested write_tle_file)
in SRC/test_main.py."""

import contextlib
from datetime import UTC, datetime
import io
import os
import warnings

import pytest

from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit_specification
from object_definitions.Satellite_def import Satellite

APOGEE_M = 421e3 + Constants.R_EARTH_MEAN
ORBIT_ARGS = (51.6374, 0.0009376, 1.5935, 121.8064, 341.7727, APOGEE_M)
START_TIME = datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC)


@pytest.fixture
def orbit_no_epoch():
    return Orbit_specification(*ORBIT_ARGS)


@pytest.fixture
def orbit_with_epoch():
    return Orbit_specification(*ORBIT_ARGS, START_TIME)


class TestSatelliteConstruction:
    def test_satellite_with_no_orbit_has_orbit_n_none(self):
        sat = Satellite(1)
        assert isinstance(sat, Satellite)
        assert sat.orbit_n is None

    def test_satellite_with_orbit_without_epoch_does_not_have_tle(self, orbit_no_epoch):
        sat = Satellite(2, orbit_no_epoch)
        assert isinstance(sat, Satellite)
        assert sat.orbit_n is orbit_no_epoch
        assert sat.tle_exists() is False

    def test_satellite_with_orbit_with_epoch_has_tle(self, orbit_with_epoch):
        sat = Satellite(3, orbit_with_epoch)
        assert isinstance(sat, Satellite)
        assert sat.orbit_n is orbit_with_epoch
        assert sat.tle_exists() is True

    def test_satellite_from_orbit_list_with_epoch_generates_tle(self):
        sat = Satellite(4, [*ORBIT_ARGS, START_TIME])
        assert isinstance(sat, Satellite)
        assert sat.orbit_n.epoch_str == "24305.21246528"
        assert sat.tle_exists() is True

    def test_satellite_from_orbit_list_without_epoch_has_no_tle(self):
        sat = Satellite(5, list(ORBIT_ARGS), satName="newName")
        assert isinstance(sat, Satellite)
        assert sat.orbit_n.epoch_str is None
        assert sat.tle_exists() is False

    def test_satellite_with_custom_name_only(self):
        sat = Satellite(6, satName="newName")
        assert isinstance(sat, Satellite)
        assert sat.satName == "newName"


class TestSatelliteClassification:
    def test_set_classification_assigns_value(self):
        sat = Satellite(7, list(ORBIT_ARGS), satName="classified_sat")
        sat.set_classification("C")
        assert sat.classification == "C"


class TestSatelliteWriteTleFile:
    def test_write_tle_warns_when_epoch_missing(self):
        """write_tle_file emits a warning and leaves tle_file_path unset if the orbit has no epoch."""
        sat = Satellite(8, list(ORBIT_ARGS), satName="no_epoch_sat")
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            sat.write_tle_file()
        assert any(
            "Epoch was not specified when this Orbit was created" in str(w.message)
            for w in caught
        )
        assert sat.tle_exists() is False

    def test_write_tle_smoke(self, tmp_path, monkeypatch):
        """Constructing a Satellite from a list that includes start_time should write a TLE.
        Same coverage as the legacy testFreeFlyerConnection (which was misnamed — it never
        connected to FreeFlyer, only smoke-tested write_tle_file)."""
        monkeypatch.setattr(Constants, "tle_eph_folder_path", str(tmp_path))
        sat = Satellite(9, [*ORBIT_ARGS, START_TIME])
        # The constructor already calls write_tle_file when the list includes an epoch.
        # Calling it again here exercises the second-write codepath as well.
        sat.write_tle_file()
        assert sat.tle_exists() is True
        assert os.path.isfile(sat.tle_file_path)
        with open(sat.tle_file_path, encoding="ascii") as tle_file:
            assert tle_file.read().strip() == sat.tle_str

    def test_set_epoch_writes_loadable_tle(self, orbit_no_epoch, tmp_path, monkeypatch):
        monkeypatch.setattr(Constants, "tle_eph_folder_path", str(tmp_path))
        sat = Satellite(11, orbit_no_epoch, satName="epoch_later")

        sat.set_epoch(START_TIME)

        assert sat.tle_exists()
        assert sat.skf is not None


class TestSatelliteFromTleFile:
    def test_load_iss_from_tle_file(self):
        """Construct a Satellite from a TLE file, then re-export it via write_tle_file."""
        iss_path = os.path.join(Constants.tle_folderpath, "ISS.tle")
        if not os.path.exists(iss_path):
            pytest.skip(f"ISS.tle missing at {iss_path}; cannot run TLE-load test.")
        sat = Satellite(10, tle_file_path=iss_path)
        assert isinstance(sat, Satellite)
        sat.write_tle_file()
        assert sat.tle_exists() is True

    def test_multi_satellite_index_keeps_metadata_and_skyfield_record_aligned(self):
        galileo_path = os.path.join(Constants.tle_folderpath, "GALILEO.tle")
        if not os.path.exists(galileo_path):
            pytest.skip(f"GALILEO.tle missing at {galileo_path}.")

        sat = Satellite(0, tle_file_path=galileo_path, satNo_in_tle=10)

        assert sat.satName == sat.skf.name
        assert sat.satID == sat.skf.model.satnum

    def test_multi_satellite_name_selects_matching_skyfield_record(self):
        galileo_path = os.path.join(Constants.tle_folderpath, "GALILEO.tle")
        if not os.path.exists(galileo_path):
            pytest.skip(f"GALILEO.tle missing at {galileo_path}.")
        selected_name = "GSAT0208 (GALILEO 11)"

        sat = Satellite(0, tle_file_path=galileo_path, satName=selected_name)

        assert sat.satName == selected_name
        assert sat.skf.name == selected_name
