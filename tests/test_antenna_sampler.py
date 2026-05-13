"""Unit tests for RxAntenna._bilinear_pattern_sample_linear — the bilinear pattern interpolator
introduced in the antenna-integration (b) work. Static-method, so we don't need to instantiate
the full RxAntenna class."""

import numpy as np
import pytest

from object_definitions.Rx_antenna_def import RxAntenna


# Convenience reference
sample = staticmethod(RxAntenna._bilinear_pattern_sample_linear)


def _pattern_with_value_at(n_theta, n_phi, theta_idx, phi_idx, value_db=0.0, floor_db=-60.0):
    """Build a pattern with `value_db` at one cell and `floor_db` everywhere else."""
    pattern = np.full((n_theta, n_phi), floor_db, dtype=float)
    pattern[theta_idx, phi_idx] = value_db
    return pattern


class TestBilinearPatternSampleLinear:
    def test_returns_cell_value_at_integer_grid_point(self):
        # A 4x8 pattern with a known peak at (theta_idx=2, phi_idx=3)
        pattern_db = _pattern_with_value_at(4, 8, theta_idx=2, phi_idx=3, value_db=0.0, floor_db=-60.0)
        # theta=180 * 2 / 3 = 120; phi=360 * 3 / 8 = 135
        result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=120.0, phi_deg=135.0)
        # peak in linear = 10**0 = 1.0; weights collapse to single cell when on grid
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_returns_floor_value_far_from_peak(self):
        pattern_db = _pattern_with_value_at(4, 8, theta_idx=2, phi_idx=3, value_db=0.0, floor_db=-60.0)
        # Sample a cell far from the peak; expect linear floor
        result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=0.0)
        assert result == pytest.approx(10.0 ** (-60.0 / 10.0), abs=1e-12)

    def test_linear_interpolation_midpoint_between_equal_cells(self):
        # Uniform pattern → midpoint sample returns the same linear value
        pattern_db = np.full((4, 8), -3.0, dtype=float)
        # Midpoint between two cells
        result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=60.0, phi_deg=22.5)
        expected_linear = 10.0 ** (-3.0 / 10.0)
        assert result == pytest.approx(expected_linear, rel=1e-12)

    def test_linear_interpolation_midpoint_between_different_cells(self):
        # Two cells: one at 0 dB (=1.0 lin), one at -6 dB (=0.251 lin).
        # Bilinear midpoint in LINEAR space = (1.0 + 0.251) / 2 ≈ 0.6256
        # If interpolation were in dB space, midpoint = -3 dB = 0.501 lin → would differ.
        pattern_db = np.full((2, 2), -6.0, dtype=float)
        pattern_db[0, 0] = 0.0
        # Midpoint between (0,0) and (0,1): theta=0, phi between samples 0 and 1
        # On a 2x2 grid with endpoint-False phi, samples are at phi=0 and phi=180.
        # Midpoint = phi=90
        result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=90.0)
        expected_linear = 0.5 * (10.0 ** 0.0) + 0.5 * (10.0 ** (-6.0 / 10.0))
        assert result == pytest.approx(expected_linear, rel=1e-9)

    def test_phi_wraparound_seam(self):
        # Build a pattern where cell 0 and cell N-1 differ; sample just inside the seam
        # to verify wraparound interpolates across the 360°/0° boundary.
        n_theta, n_phi = 2, 4
        pattern_db = np.full((n_theta, n_phi), -60.0, dtype=float)
        pattern_db[0, 0] = 0.0   # phi=0, lin=1
        pattern_db[0, -1] = -6.0  # phi=270, lin=0.251
        # phi=315 sits between sample 3 (phi=270, lin=0.251) and sample 0 (phi=360≡0, lin=1).
        # Halfway between them in linear = (0.251 + 1.0) / 2 ≈ 0.6256
        result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=315.0)
        expected = 0.5 * (10.0 ** (-6.0 / 10.0)) + 0.5 * (10.0 ** 0.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_phi_modulo_handles_values_beyond_360(self):
        pattern_db = np.full((2, 4), -3.0, dtype=float)
        a = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=45.0)
        b = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=45.0 + 360.0)
        c = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=0.0, phi_deg=45.0 - 720.0)
        assert a == pytest.approx(b, rel=1e-12)
        assert a == pytest.approx(c, rel=1e-12)

    def test_theta_clamping_at_poles(self):
        # theta values outside [0, 180] should be clamped to the pole sample
        pattern_db = np.full((4, 8), -10.0, dtype=float)
        pattern_db[0, :] = 0.0    # theta=0 row (north pole)
        pattern_db[-1, :] = -20.0  # theta=180 row (south pole)
        # Below 0
        np.testing.assert_allclose(
            RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=-30.0, phi_deg=0.0),
            10.0 ** 0.0,
            rtol=1e-12,
        )
        # Above 180
        np.testing.assert_allclose(
            RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta_deg=200.0, phi_deg=0.0),
            10.0 ** (-20.0 / 10.0),
            rtol=1e-12,
        )

    def test_uniform_pattern_returns_constant(self):
        # Sanity: a constant pattern should sample to the same value everywhere
        pattern_db = np.full((10, 20), -7.5, dtype=float)
        expected = 10.0 ** (-7.5 / 10.0)
        for theta in [0.0, 30.0, 90.0, 145.0, 180.0]:
            for phi in [0.0, 45.0, 180.0, 270.0, 359.0]:
                result = RxAntenna._bilinear_pattern_sample_linear(pattern_db, theta, phi)
                assert result == pytest.approx(expected, rel=1e-12)
