"""Unit tests for the backscatter BRP geometry in BackScattPoint_def.py.

These tests target the pure-geometry helper `calculate_ellipsoid_intersection_t`
directly, so they don't need Satellite objects, propagated states, or pymap3d
conversions. The BRP is the point where the line through Tx and Rx, extended
past Rx, hits the Earth ellipsoid — i.e. the retroreflector geometry.
"""

import numpy as np
import pytest

from object_definitions.BackScattPoint_def import (
    a as WGS84_A,
    b as WGS84_B,
    calculate_ellipsoid_intersection_t,
)


class TestEllipsoidIntersection:
    def test_tx_directly_above_rx_brp_at_subpoint(self):
        # Tx and Rx aligned on the +x axis. The BRP should land on the +x ellipsoid
        # surface point (essentially R_E along x), i.e. directly below the LEO.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])      # LEO at 700 km
        tx_ecef = np.array([WGS84_A + 20_200_000.0, 0.0, 0.0])   # GPS at 20 200 km
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef)
        assert reason == "ok"
        assert brp is not None
        # BRP should be on the +x surface within numerical tolerance.
        assert brp[0] == pytest.approx(WGS84_A, rel=1e-6)
        assert brp[1] == pytest.approx(0.0, abs=1e-6)
        assert brp[2] == pytest.approx(0.0, abs=1e-6)

    def test_brp_lies_on_the_tx_rx_line(self):
        # For an off-axis Tx, the BRP must still be exactly on the line through Tx and Rx.
        # We verify by checking that (BRP - Tx) and (Rx - Tx) are parallel.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        tx_ecef = np.array([15_000_000.0, 12_000_000.0, 5_000_000.0])
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef)
        assert reason == "ok"
        v_to_rx = rx_ecef - tx_ecef
        v_to_brp = brp - tx_ecef
        # Parallel check: cross product magnitude / (|a||b|) should be ~0
        cross = np.cross(v_to_rx, v_to_brp)
        scale = np.linalg.norm(v_to_rx) * np.linalg.norm(v_to_brp)
        assert np.linalg.norm(cross) / scale == pytest.approx(0.0, abs=1e-9)
        # And BRP should be past Rx (further from Tx than Rx is).
        assert np.linalg.norm(v_to_brp) > np.linalg.norm(v_to_rx)

    def test_brp_on_ellipsoid_surface(self):
        # The returned BRP must satisfy the ellipsoid equation.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        tx_ecef = np.array([WGS84_A + 20_200_000.0, 1_000_000.0, 0.0])
        brp, _, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef)
        assert reason == "ok"
        # x²/a² + y²/a² + z²/b² should equal 1
        residual = brp[0] ** 2 / WGS84_A**2 + brp[1] ** 2 / WGS84_A**2 + brp[2] ** 2 / WGS84_B**2 - 1.0
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_tx_and_rx_too_far_apart_returns_opposite_side(self):
        # Central angle = 90° between Tx and Rx; default cutoff = 80° should reject.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        tx_ecef = np.array([0.0, WGS84_A + 20_200_000.0, 0.0])
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef)
        assert reason == "opposite_side"
        assert brp is None
        assert t is None

    def test_line_misses_earth_returns_outside_segment(self):
        # At ~70° central angle (LEO at 700 km, GPS at 20 200 km), the line through Tx and Rx
        # tangents away from Earth — the extended line never touches the ellipsoid. The
        # central-angle filter at default 80° lets this case through, and the quadratic's
        # discriminant then comes out negative.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        alpha_rad = np.deg2rad(70.0)
        tx_ecef = np.array([
            (WGS84_A + 20_200_000.0) * np.cos(alpha_rad),
            (WGS84_A + 20_200_000.0) * np.sin(alpha_rad),
            0.0,
        ])
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef)
        assert reason == "outside_segment"
        assert brp is None
        assert t is None

    def test_degenerate_tx_equals_rx(self):
        # Zero-length direction vector → degenerate.
        pos = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        brp, t, reason = calculate_ellipsoid_intersection_t(pos, pos)
        assert reason == "degenerate"
        assert brp is None
        assert t is None
