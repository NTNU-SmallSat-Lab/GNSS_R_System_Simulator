"""Unit tests for the backscatter BRP geometry in BackScattPoint_def.py.

The pure-geometry helper `calculate_ellipsoid_intersection_t` is exercised
directly (no Satellite objects, no pymap3d). The antenna-FOV pre-gate inside
`compute_brp_coordinates` is exercised through a tiny in-memory Satellite-like
stub plus the deterministic RxAntenna fixture.

The BRP is the point where the line through Tx and Rx, extended past Rx, hits
the Earth ellipsoid — i.e. the retroreflector geometry.
"""

import datetime as dt
import types

from antenna_fixture import make_test_antenna
import numpy as np
import pytest

from object_definitions.BackScattPoint_def import (
    BRP,
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
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, max_central_angle_deg=80.0)
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
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, max_central_angle_deg=80.0)
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
        brp, _, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, max_central_angle_deg=80.0)
        assert reason == "ok"
        # x²/a² + y²/a² + z²/b² should equal 1
        residual = brp[0] ** 2 / WGS84_A**2 + brp[1] ** 2 / WGS84_A**2 + brp[2] ** 2 / WGS84_B**2 - 1.0
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_tx_and_rx_too_far_apart_returns_opposite_side(self):
        # Central angle = 90° between Tx and Rx; cutoff = 80° should reject.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        tx_ecef = np.array([0.0, WGS84_A + 20_200_000.0, 0.0])
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, max_central_angle_deg=80.0)
        assert reason == "opposite_side"
        assert brp is None
        assert t is None

    def test_line_misses_earth_returns_outside_segment(self):
        # At ~70° central angle (LEO at 700 km, GPS at 20 200 km), the line through Tx and Rx
        # tangents away from Earth — the extended line never touches the ellipsoid. The
        # 80° central-angle filter lets this case through, and the quadratic's discriminant
        # then comes out negative.
        rx_ecef = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        alpha_rad = np.deg2rad(70.0)
        tx_ecef = np.array([
            (WGS84_A + 20_200_000.0) * np.cos(alpha_rad),
            (WGS84_A + 20_200_000.0) * np.sin(alpha_rad),
            0.0,
        ])
        brp, t, reason = calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, max_central_angle_deg=80.0)
        assert reason == "outside_segment"
        assert brp is None
        assert t is None

    def test_degenerate_tx_equals_rx(self):
        # Zero-length direction vector → degenerate.
        pos = np.array([WGS84_A + 700_000.0, 0.0, 0.0])
        brp, t, reason = calculate_ellipsoid_intersection_t(pos, pos, max_central_angle_deg=80.0)
        assert reason == "degenerate"
        assert brp is None
        assert t is None


def _make_satellite_stub(positions_eci: np.ndarray, velocities_eci: np.ndarray) -> types.SimpleNamespace:
    """Tiny stand-in for a Satellite — only the two attributes BRP.compute_brp_coordinates reads."""
    return types.SimpleNamespace(positions_eci=positions_eci, velocities_eci=velocities_eci)


class TestAntennaFovPreGate:
    """The pre-gate skips BRPs whose retroreflector look direction falls outside the LEO antenna FOV.

    Geometry: LEO at 700 km on the +x axis with velocity along +y (typical near-polar pass). With
    the fixture's R_antenna_to_body and the codebase's R_body_to_vnc_normal, the antenna boresight
    points along the LEO's local nadir, so the BRP look direction's off-boresight angle equals its
    off-nadir angle from the LEO.
    """

    @staticmethod
    def _build_state(tx_offset_along_y: float):
        """LEO at 700 km +x with +y velocity; Tx at GPS altitude, displaced along +y for a controlled off-nadir BRP."""
        rx_pos = np.array([[WGS84_A + 700_000.0], [0.0], [0.0]])         # shape (3, 1)
        rx_vel = np.array([[0.0], [7_500.0], [0.0]])                     # shape (3, 1)
        gnss_alt = WGS84_A + 20_200_000.0
        tx_pos = np.array([[gnss_alt], [tx_offset_along_y], [0.0]])
        tx_vel = np.array([[0.0], [3_900.0], [0.0]])
        times = np.array([dt.datetime(2025, 4, 2, 12, 0, 0, tzinfo=dt.UTC)])
        return tx_pos, tx_vel, rx_pos, rx_vel, times

    def test_nadir_brp_passes_fov_gate(self):
        # Tx directly above LEO (zero y offset) → BRP at sub-LEO point, look direction at boresight.
        antenna = make_test_antenna()
        tx_pos, tx_vel, rx_pos, rx_vel, times = self._build_state(tx_offset_along_y=0.0)
        brp = BRP(gps_index=0, leo_index=0, earth_model="elliptical", max_central_angle_deg=80.0, max_incidence_angle_deg=45.0)
        brp.compute_brp_coordinates(
            GNSS_state=_make_satellite_stub(tx_pos, tx_vel),
            Sat_state=_make_satellite_stub(rx_pos, rx_vel),
            times=times,
            antenna=antenna,
            frequency_hz=1575.42e6,
            tx_eirp_dbw=27.0,
        )
        assert np.isfinite(brp.refl_LL_vec).all()

    def test_far_off_nadir_brp_rejected_by_fov_gate(self):
        # Tx displaced along +y → BRP look direction tilts off nadir. With the antenna's fixed FOV
        # tightened to 5°, even a modest tilt is rejected by the pre-gate before the
        # ellipsoid math runs, so the BRP stays NaN.
        antenna = make_test_antenna()
        antenna.max_fov_off_boresight_deg = 5.0  # tighter than the fixture default 30°
        tx_pos, tx_vel, rx_pos, rx_vel, times = self._build_state(tx_offset_along_y=10_000_000.0)
        brp = BRP(gps_index=0, leo_index=0, earth_model="elliptical", max_central_angle_deg=80.0, max_incidence_angle_deg=45.0)
        brp.compute_brp_coordinates(
            GNSS_state=_make_satellite_stub(tx_pos, tx_vel),
            Sat_state=_make_satellite_stub(rx_pos, rx_vel),
            times=times,
            antenna=antenna,
            frequency_hz=1575.42e6,
            tx_eirp_dbw=27.0,
        )
        assert np.isnan(brp.refl_LL_vec).all()

    def test_no_antenna_skips_fov_gate(self):
        # Without an antenna, the pre-gate is bypassed and the BRP is computed by geometry alone.
        tx_pos, tx_vel, rx_pos, rx_vel, times = self._build_state(tx_offset_along_y=0.0)
        brp = BRP(gps_index=0, leo_index=0, earth_model="elliptical", max_central_angle_deg=80.0, max_incidence_angle_deg=45.0)
        brp.compute_brp_coordinates(
            GNSS_state=_make_satellite_stub(tx_pos, tx_vel),
            Sat_state=_make_satellite_stub(rx_pos, rx_vel),
            times=times,
            antenna=None,
            frequency_hz=1575.42e6,
            tx_eirp_dbw=27.0,
        )
        assert np.isfinite(brp.refl_LL_vec).all()
