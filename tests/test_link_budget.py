"""Unit tests for functions/link_budget.py::compute_rx_look_direction_antenna_frame."""

from _link_budget_settings import link_budget_settings
import numpy as np
import pytest

from functions.link_budget import compute_bistatic_link_budget, compute_rx_look_direction_antenna_frame
from functions.rotations import eci_to_antenna_rotation_from_state
from object_definitions.Constants_def import Constants


class _FakeAntenna:
    """Duck-typed antenna providing only the attribute the function reads."""

    def __init__(self, R_antenna_to_body):
        self.R_antenna_to_body = R_antenna_to_body


def _equatorial_state():
    """Return (sat_pos, sat_vel, refl_pos) for a well-conditioned equatorial geometry."""
    sat_pos = np.array([7e6, 0.0, 0.0])
    sat_vel = np.array([0.0, 7e3, 0.0])
    refl_pos = np.array([6.371e6, 0.0, 0.0])
    return sat_pos, sat_vel, refl_pos


class TestComputeRxLookDirectionAntennaFrame:
    def test_antenna_none_returns_nan(self):
        sat_pos, sat_vel, refl_pos = _equatorial_state()
        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, refl_pos, antenna=None)
        assert np.isnan(result).all()

    def test_shape_mismatch_returns_nan(self):
        antenna = _FakeAntenna(np.eye(3))
        sat_pos = np.array([7e6, 0.0])  # wrong shape
        sat_vel = np.array([0.0, 7e3, 0.0])
        refl_pos = np.array([6.371e6, 0.0, 0.0])
        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, refl_pos, antenna)
        assert np.isnan(result).all()

    def test_nan_input_returns_nan(self):
        antenna = _FakeAntenna(np.eye(3))
        sat_pos, sat_vel, refl_pos = _equatorial_state()
        sat_pos = sat_pos.copy()
        sat_pos[0] = np.nan
        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, refl_pos, antenna)
        assert np.isnan(result).all()

    def test_zero_slant_range_returns_nan(self):
        antenna = _FakeAntenna(np.eye(3))
        sat_pos, sat_vel, _ = _equatorial_state()
        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, sat_pos.copy(), antenna)
        assert np.isnan(result).all()

    def test_degenerate_vnc_returns_nan(self):
        # Zero velocity → build_vnc_frame returns None → function must return NaNs.
        antenna = _FakeAntenna(np.eye(3))
        sat_pos, _, refl_pos = _equatorial_state()
        result = compute_rx_look_direction_antenna_frame(sat_pos, np.zeros(3), refl_pos, antenna)
        assert np.isnan(result).all()

    def test_happy_path_returns_unit_vector(self):
        antenna = _FakeAntenna(np.eye(3))
        sat_pos, sat_vel, refl_pos = _equatorial_state()
        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, refl_pos, antenna)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_matches_manual_chain(self):
        # The function must equal (R_eci_to_antenna @ n_ar_eci) normalised.
        antenna = _FakeAntenna(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        sat_pos = np.array([7e6, 1e5, 3e5])
        sat_vel = np.array([1e2, 7e3, -1e2])
        refl_pos = np.array([6.371e6, 0.0, 0.0])
        q = np.array([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])

        result = compute_rx_look_direction_antenna_frame(sat_pos, sat_vel, refl_pos, antenna, q)

        R = eci_to_antenna_rotation_from_state(sat_pos, sat_vel, antenna, q)
        n_ar_eci = (refl_pos - sat_pos) / np.linalg.norm(refl_pos - sat_pos)
        expected = R @ n_ar_eci
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestComputeBistaticLinkBudgetNoise:
    def _budget(self, *, noise_bandwidth_hz=2.0e6, system_noise_temperature_k=290.0):
        return compute_bistatic_link_budget(
            range_tx_m=np.array([20.0e6]),
            range_rx_m=np.array([720.0e3]),
            rx_gain_dbi=np.array([5.0]),
            rcs_m2=np.array([1.0e6]),
            frequency_hz=1.57542e9,
            tx_eirp_dbw=27.0,
            settings=link_budget_settings(
                noise_bandwidth_hz=noise_bandwidth_hz,
                system_noise_temperature_k=system_noise_temperature_k,
            ),
        )

    def test_noise_power_uses_k_t_b(self):
        lb = self._budget(noise_bandwidth_hz=2.0e6, system_noise_temperature_k=290.0)

        expected_density = Constants.BOLTZMANN_CONSTANT * 290.0
        expected_power = expected_density * 2.0e6

        np.testing.assert_allclose(lb["noise_density_w_hz"], expected_density, rtol=1e-15)
        np.testing.assert_allclose(lb["noise_power_w"], expected_power, rtol=1e-15)
        np.testing.assert_allclose(lb["noise_power_dbw"], 10.0 * np.log10(expected_power), rtol=1e-15)
        np.testing.assert_allclose(lb["noise_power_dbm"], 10.0 * np.log10(expected_power) + 30.0, rtol=1e-15)

    def test_snr_and_cn0_are_derived_from_received_power(self):
        lb = self._budget(noise_bandwidth_hz=2.0e6, system_noise_temperature_k=290.0)

        np.testing.assert_allclose(
            lb["snr_rf_db"],
            lb["received_power_dbw"] - lb["noise_power_dbw"],
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            lb["cn0_db_hz"],
            lb["received_power_dbw"] - lb["noise_density_dbw_hz"],
            rtol=1e-15,
        )

    def test_doubling_bandwidth_reduces_snr_but_not_cn0(self):
        narrow = self._budget(noise_bandwidth_hz=1.0e6)
        wide = self._budget(noise_bandwidth_hz=2.0e6)

        np.testing.assert_allclose(narrow["cn0_db_hz"], wide["cn0_db_hz"], atol=1e-12)
        np.testing.assert_allclose(narrow["snr_rf_db"] - wide["snr_rf_db"], 10.0 * np.log10(2.0), atol=1e-12)

    @pytest.mark.parametrize(
        ("noise_bandwidth_hz", "system_noise_temperature_k", "match"),
        [
            (0.0, 290.0, "noise_bandwidth_hz"),
            (2.0e6, 0.0, "system_noise_temperature_k"),
        ],
    )
    def test_non_positive_noise_settings_raise(self, noise_bandwidth_hz, system_noise_temperature_k, match):
        with pytest.raises(ValueError, match=match):
            link_budget_settings(
                noise_bandwidth_hz=noise_bandwidth_hz,
                system_noise_temperature_k=system_noise_temperature_k,
            )
