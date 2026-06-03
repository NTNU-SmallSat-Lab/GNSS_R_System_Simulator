"""
Unit tests for BistaticGeometry and build_bistatic_geometry in functions/link_budget.py.
"""

import numpy as np
import pytest

from functions.link_budget import BistaticGeometry, build_bistatic_geometry


def _stack_tx_rx(tx: np.ndarray, rx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape two (3,) vectors to (3, 1) for single-sample tests.
    """
    return tx.reshape(3, 1), rx.reshape(3, 1)


class TestBuildBistaticGeometry:
    def test_zenith_geometry_all_angles_zero(self):
        # Surface at (R, 0, 0) with outward normal +x; Tx and Rx both along the +x line.
        R = 6.371e6
        surf = np.array([[R, 0.0, 0.0]])
        normal = np.array([[1.0, 0.0, 0.0]])
        tx, rx = _stack_tx_rx(np.array([2 * R, 0.0, 0.0]), np.array([1.5 * R, 0.0, 0.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        np.testing.assert_allclose(geom.incidence_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.scattering_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.bistatic_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.range_tx_m, [R], rtol=1e-12)
        np.testing.assert_allclose(geom.range_rx_m, [0.5 * R], rtol=1e-12)

    def test_mirror_geometry_bistatic_is_twice_incidence(self):
        # Surface at origin, normal +z; Tx at (-3, 0, 4), Rx at (+3, 0, 4) (mirror symmetric).
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx, rx = _stack_tx_rx(np.array([-3.0, 0.0, 4.0]), np.array([3.0, 0.0, 4.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        expected_theta = np.degrees(np.arctan2(3.0, 4.0))  # ≈ 36.87°
        np.testing.assert_allclose(geom.incidence_angle_deg, [expected_theta], atol=1e-9)
        np.testing.assert_allclose(geom.scattering_angle_deg, [expected_theta], atol=1e-9)
        np.testing.assert_allclose(geom.bistatic_angle_deg, [2 * expected_theta], atol=1e-9)
        np.testing.assert_allclose(geom.range_tx_m, [5.0], atol=1e-12)
        np.testing.assert_allclose(geom.range_rx_m, [5.0], atol=1e-12)

    def test_backscatter_geometry_bistatic_zero(self):
        # Tx and Rx both straight above the surface — pure backscatter.
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx, rx = _stack_tx_rx(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 5.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        np.testing.assert_allclose(geom.incidence_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.scattering_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.bistatic_angle_deg, [0.0], atol=1e-9)

    def test_forward_scatter_bistatic_180(self):
        # Tx above, Rx directly below the surface point along the same line — pure forward scatter.
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx, rx = _stack_tx_rx(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -10.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        np.testing.assert_allclose(geom.incidence_angle_deg, [0.0], atol=1e-9)
        np.testing.assert_allclose(geom.scattering_angle_deg, [180.0], atol=1e-9)
        np.testing.assert_allclose(geom.bistatic_angle_deg, [180.0], atol=1e-9)

    def test_unit_vectors_have_unit_norm(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 5.0]])  # non-unit input, must be normalised internally
        tx, rx = _stack_tx_rx(np.array([-3.0, 0.0, 4.0]), np.array([3.0, 0.0, 4.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        np.testing.assert_allclose(np.linalg.norm(geom.n_tx2s, axis=1), [1.0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(geom.n_s2rx, axis=1), [1.0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(geom.n_surface, axis=1), [1.0], atol=1e-12)

    def test_directions_have_expected_signs(self):
        # Mirror geometry — verify n_tx2s points from Tx toward the surface (downward),
        # n_s2rx points from the surface toward Rx (upward).
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx, rx = _stack_tx_rx(np.array([-3.0, 0.0, 4.0]), np.array([3.0, 0.0, 4.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        # n_tx2s = (surf - tx)/|.| with Tx at (-3, 0, 4) -> direction (3, 0, -4)/5 = (0.6, 0, -0.8)
        np.testing.assert_allclose(geom.n_tx2s[0], [0.6, 0.0, -0.8], atol=1e-12)
        # n_s2rx = (rx - surf)/|.| with Rx at (3, 0, 4) -> direction (3, 0, 4)/5 = (0.6, 0, 0.8)
        np.testing.assert_allclose(geom.n_s2rx[0], [0.6, 0.0, 0.8], atol=1e-12)

    def test_nan_input_propagates_to_nan_outputs(self):
        # Sample 0 has a NaN Tx coordinate; sample 1 is valid.
        surf = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        tx = np.array([[np.nan, 0.0], [0.0, 0.0], [10.0, 10.0]])
        rx = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        assert np.isnan(geom.range_tx_m[0])
        assert np.isnan(geom.incidence_angle_deg[0])
        assert np.isnan(geom.bistatic_angle_deg[0])
        assert np.isnan(geom.n_tx2s[0]).all()
        assert np.isfinite(geom.range_tx_m[1])
        assert np.isfinite(geom.incidence_angle_deg[1])

    def test_zero_range_propagates_nan(self):
        # Tx co-located with the surface point.
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx, rx = _stack_tx_rx(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        assert np.isnan(geom.range_tx_m[0])
        assert np.isnan(geom.incidence_angle_deg[0])
        assert np.isnan(geom.bistatic_angle_deg[0])

    def test_zero_length_normal_propagates_nan(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 0.0]])  # degenerate
        tx, rx = _stack_tx_rx(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 5.0]))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        assert np.isnan(geom.n_surface[0]).all()
        assert np.isnan(geom.incidence_angle_deg[0])

    def test_multiple_samples_independent(self):
        # Sample 0: zenith (β=0, θ=0); sample 1: mirror geometry (β = 2θ).
        surf = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        tx = np.array([[0.0, -3.0], [0.0, 0.0], [10.0, 4.0]])
        rx = np.array([[0.0, 3.0], [0.0, 0.0], [5.0, 4.0]])

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        expected_theta = np.degrees(np.arctan2(3.0, 4.0))
        np.testing.assert_allclose(geom.incidence_angle_deg, [0.0, expected_theta], atol=1e-9)
        np.testing.assert_allclose(geom.bistatic_angle_deg, [0.0, 2 * expected_theta], atol=1e-9)

    def test_output_shapes(self):
        n = 5
        surf = np.zeros((n, 3))
        normal = np.tile([0.0, 0.0, 1.0], (n, 1))
        tx = np.tile(np.array([[0.0], [0.0], [10.0]]), (1, n))
        rx = np.tile(np.array([[0.0], [0.0], [5.0]]), (1, n))

        geom = build_bistatic_geometry(tx, rx, surf, normal)

        assert geom.n_surface.shape == (n, 3)
        assert geom.n_tx2s.shape == (n, 3)
        assert geom.n_s2rx.shape == (n, 3)
        assert geom.range_tx_m.shape == (n,)
        assert geom.range_rx_m.shape == (n,)
        assert geom.incidence_angle_deg.shape == (n,)
        assert geom.scattering_angle_deg.shape == (n,)
        assert geom.bistatic_angle_deg.shape == (n,)

    def test_invalid_tx_shape_raises(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0, 1.0]])
        tx = np.array([[0.0], [0.0]])  # (2, 1) — wrong
        rx = np.array([[0.0], [0.0], [5.0]])
        with pytest.raises(ValueError, match="tx_positions must have shape"):
            build_bistatic_geometry(tx, rx, surf, normal)

    def test_invalid_surface_shape_raises(self):
        surf = np.array([[0.0, 0.0]])  # (1, 2) — wrong
        normal = np.array([[0.0, 0.0, 1.0]])
        tx = np.array([[0.0], [0.0], [10.0]])
        rx = np.array([[0.0], [0.0], [5.0]])
        with pytest.raises(ValueError, match="surface_positions must have shape"):
            build_bistatic_geometry(tx, rx, surf, normal)

    def test_mismatched_n_raises(self):
        surf = np.array([[0.0, 0.0, 0.0]])  # n=1
        normal = np.array([[0.0, 0.0, 1.0]])
        tx = np.array([[0.0, 1.0], [0.0, 0.0], [10.0, 10.0]])  # n=2
        rx = np.array([[0.0], [0.0], [5.0]])
        with pytest.raises(ValueError, match="must have n=1"):
            build_bistatic_geometry(tx, rx, surf, normal)

    def test_surface_normal_shape_mismatch_raises(self):
        surf = np.array([[0.0, 0.0, 0.0]])
        normal = np.array([[0.0, 0.0]])
        tx = np.array([[0.0], [0.0], [10.0]])
        rx = np.array([[0.0], [0.0], [5.0]])
        with pytest.raises(ValueError, match="surface_normals must match"):
            build_bistatic_geometry(tx, rx, surf, normal)


class TestBistaticGeometryDataclass:
    def test_is_frozen(self):
        geom = BistaticGeometry(
            n_surface=np.zeros((1, 3)),
            n_tx2s=np.zeros((1, 3)),
            n_s2rx=np.zeros((1, 3)),
            range_tx_m=np.zeros(1),
            range_rx_m=np.zeros(1),
            incidence_angle_deg=np.zeros(1),
            scattering_angle_deg=np.zeros(1),
            bistatic_angle_deg=np.zeros(1),
        )
        with pytest.raises((AttributeError, Exception)):
            geom.range_tx_m = np.ones(1)
