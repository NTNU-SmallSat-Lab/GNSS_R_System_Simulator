"""Unit tests for functions/link_budget.py::compute_rx_look_direction_antenna_frame."""

import numpy as np

from functions.link_budget import compute_rx_look_direction_antenna_frame
from functions.rotations import eci_to_antenna_rotation_from_state


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
