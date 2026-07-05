"""Tests for ReadWriteSatData.read_tle_data. Migrated from testReadWriteData in SRC/test_main.py.

The expected values are snapshots of the TLE files committed to the repo. If those files are
refreshed (TLEs age), the snapshot values must be updated to match — this is the natural
brittleness of testing parsed TLE content against ground truth.
"""

from datetime import UTC, datetime
import math
import os

import numpy as np
import pytest
from skyfield.api import load

from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit_specification
from object_definitions.ReadWriteSatData import (
    ReadWriteSatData,
    _mean_anomaly_to_true_anomaly,
    _state_from_classical_elements,
    _true_anomaly_to_mean_anomaly,
)
from object_definitions.Satellite_def import Satellite

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


class TestTrueAnomalyTleGeneration:
    """The 5th classical element is the TRUE anomaly; the generated TLE must reproduce the
    osculating state built from it (not from a mean anomaly of the same numeric value)."""

    EPOCH = datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC)
    # Eccentric orbit so true and mean anomaly differ by a few degrees (~hundreds of km).
    I_DEG, E, RAAN_DEG, ARGP_DEG, NU_DEG, A_M = 45.0, 0.05, 30.0, 60.0, 200.0, 7000e3

    def _target_state(self, nu_rad):
        elements = {
            "a": self.A_M, "e": self.E, "i": np.radians(self.I_DEG), "raan": np.radians(self.RAAN_DEG),
            "arg_perigee": np.radians(self.ARGP_DEG), "true_anomaly": nu_rad,
        }
        return _state_from_classical_elements(elements)

    def test_generated_tle_reproduces_true_anomaly_state(self):
        orbit = Orbit_specification(self.I_DEG, self.E, self.RAAN_DEG, self.ARGP_DEG, self.NU_DEG, self.A_M, self.EPOCH)
        sat = Satellite(12345, orbit)

        r_true, _ = self._target_state(np.radians(self.NU_DEG))  # state defined by the true anomaly
        r_asmean, _ = self._target_state(_mean_anomaly_to_true_anomaly(np.radians(self.NU_DEG), self.E))  # old interpretation

        geocentric = sat.skf.at(load.timescale().from_datetime(self.EPOCH))
        r_skf = geocentric.position.m

        # The propagated TLE matches the true-anomaly state to within the SGP4 fit residual,
        # and is far from the (wrong) mean-anomaly state.
        assert np.linalg.norm(r_skf - r_true) < 50.0
        assert np.linalg.norm(r_true - r_asmean) > 1e4

    def test_mean_true_anomaly_roundtrip(self):
        for mean_anomaly_deg in [0.0, 45.0, 123.4, 359.0]:
            nu_deg = ReadWriteSatData.mean_to_true_anomaly_deg(mean_anomaly_deg, self.E)
            mean_back_deg = np.degrees(_true_anomaly_to_mean_anomaly(np.radians(nu_deg), self.E))
            assert (mean_back_deg - mean_anomaly_deg + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)


class TestTleFitWeightsFromYaml:
    """The residual normalization weights are engine settings read strictly from simulator_settings.yaml."""

    def test_returns_committed_weights(self):
        import __init__ as sim_init

        assert sim_init.tle_fit_weights_from_yaml() == (100.0, 0.1)

    def test_missing_section_raises(self, monkeypatch):
        import __init__ as sim_init

        monkeypatch.setattr(sim_init, "load_simulator_settings", lambda: {"simulation": {"step_size_s": 20}})
        with pytest.raises(KeyError, match="tle_fit"):
            sim_init.tle_fit_weights_from_yaml()

    def test_missing_key_raises(self, monkeypatch):
        import __init__ as sim_init

        monkeypatch.setattr(sim_init, "load_simulator_settings", lambda: {"tle_fit": {"position_weight_m": 100.0}})
        with pytest.raises(KeyError, match="velocity_weight_m_s"):
            sim_init.tle_fit_weights_from_yaml()

    def test_nonpositive_weight_raises(self, monkeypatch):
        import __init__ as sim_init

        monkeypatch.setattr(sim_init, "load_simulator_settings", lambda: {"tle_fit": {"position_weight_m": 0.0, "velocity_weight_m_s": 0.1}})
        with pytest.raises(ValueError, match="finite and > 0"):
            sim_init.tle_fit_weights_from_yaml()
