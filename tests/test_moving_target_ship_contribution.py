"""Tests for the ship-contribution path in MovingTargetDetection.

These exercise the two new private methods directly on an instance that has bypassed __init__,
so we don't need a propagated constellation or a real ShipTrajectory shapefile.
"""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from object_definitions.MovingTargetDetection_def import (
    MovingTargetDetection,
    MovingTargetDetectionStruct,
)


@pytest.fixture
def mtd():
    """A MovingTargetDetection instance that has bypassed __init__; populate per-test."""
    return object.__new__(MovingTargetDetection)


BASE_TIME = dt.datetime(2026, 5, 27, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def detection_time():
    return BASE_TIME + dt.timedelta(minutes=25)             # mid-trajectory


def _trajectory(points, interval_minutes=10):
    timestamps = [BASE_TIME + dt.timedelta(minutes=interval_minutes * i) for i in range(len(points))]
    return SimpleNamespace(timestamps=timestamps, trajectory_points=points)


class TestComputeShipHeadingEci:
    def test_short_trajectory_returns_none(self, mtd, detection_time):
        # < 2 waypoints -> no heading definable.
        mtd.moving_target_object = _trajectory([(60.0, 5.0)])
        assert mtd._compute_ship_heading_eci(detection_time) is None

    def test_degenerate_segment_returns_none(self, mtd, detection_time):
        # Bracketing segment has identical endpoints -> stationary, no heading.
        mtd.moving_target_object = _trajectory([(60.0, 5.0)] * 5)
        assert mtd._compute_ship_heading_eci(detection_time) is None

    def test_mid_trajectory_uses_bracketing_segment(self, mtd, detection_time):
        # Detection at 12:25 -> falls inside the segment [12:20, 12:30] = points[2..3].
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]   # 5 waypoints, drifting north
        mtd.moving_target_object = _trajectory(points)
        heading = mtd._compute_ship_heading_eci(detection_time)
        assert heading is not None
        assert heading.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_detection_before_trajectory_uses_first_segment(self, mtd):
        # detection_time before the first waypoint -> use first segment (initial bearing).
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]
        mtd.moving_target_object = _trajectory(points)
        early_detection = BASE_TIME - dt.timedelta(minutes=5)
        heading = mtd._compute_ship_heading_eci(early_detection)
        assert heading is not None                          # initial bearing is defined
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_detection_after_trajectory_uses_last_segment(self, mtd):
        # detection_time past the last waypoint -> use final segment.
        points = [(60.0 + 0.1 * i, 5.0) for i in range(5)]
        mtd.moving_target_object = _trajectory(points)
        late_detection = BASE_TIME + dt.timedelta(hours=10)
        heading = mtd._compute_ship_heading_eci(late_detection)
        assert heading is not None
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)

    def test_interval_change_still_works(self, mtd):
        # Same logic should hold for a 5-min trajectory without code changes — the
        # helper reads waypoints directly, not the interval setting.
        points = [(60.0 + 0.1 * i, 5.0) for i in range(10)]
        mtd.moving_target_object = _trajectory(points, interval_minutes=5)
        detection = BASE_TIME + dt.timedelta(minutes=12)    # inside segment [10, 15]
        heading = mtd._compute_ship_heading_eci(detection)
        assert heading is not None
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-10)


def _build_refl_obj_mock():
    """One-sample refl_obj with the fields the helper reads from."""
    # Tx and Rx both directly above the surface point, near-zenith geometry.
    surf = np.array([[6.371e6, 0.0, 0.0]])
    tx_pos = np.array([[6.371e6 + 2.0e7], [0.0], [0.0]])   # ~ GPS altitude
    rx_pos = np.array([[6.371e6 + 7.0e5], [0.0], [0.0]])   # ~ LEO altitude
    return SimpleNamespace(
        gnss_pos_vec=tx_pos,
        sat_pos_vec=rx_pos,
        refl_ECI_vec=surf,
        link_budget={
            "tx_to_surface_range_m": np.array([2.0e7]),
            "surface_to_rx_range_m": np.array([7.0e5]),
            "rx_gain_dbi": np.array([10.0]),
            "received_power_w": np.array([1.0e-15]),
            "rcs_m2": np.array([1.0e3]),
        },
    )


class TestComputeShipPrAtEvent:
    def test_returns_nan_when_ship_dimensions_is_none(self, mtd, detection_time):
        mtd.ship_dimensions = None
        mtd.moving_target_object = _trajectory([(60.0 + 0.1 * i, 5.0) for i in range(5)])
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2 = mtd._compute_ship_pr_at_event(refl_obj, 0, "GPS", detection_time)
        assert np.isnan(pr_ship_w)
        assert np.isnan(rcs_ship_m2)

    def test_returns_nan_when_ship_is_stationary(self, mtd, detection_time):
        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.moving_target_object = _trajectory([(60.0, 5.0)] * 5)
        mtd.link_budget_frequency_hz = 1.57542e9
        mtd.tx_eirp_dbw_per_constellation = {"GPS": 27.0}
        mtd.link_budget_settings = None
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2 = mtd._compute_ship_pr_at_event(refl_obj, 0, "GPS", detection_time)
        assert np.isnan(pr_ship_w)
        assert np.isnan(rcs_ship_m2)

    def test_returns_finite_pr_for_well_formed_event(self, mtd, detection_time):
        mtd.ship_dimensions = {"length_m": 100.0, "beam_m": 15.0, "freeboard_m": 5.0}
        mtd.moving_target_object = _trajectory([(60.0 + 0.1 * i, 5.0) for i in range(5)])
        mtd.link_budget_frequency_hz = 1.57542e9
        mtd.tx_eirp_dbw_per_constellation = {"GPS": 27.0}
        mtd.link_budget_settings = None
        refl_obj = _build_refl_obj_mock()
        pr_ship_w, rcs_ship_m2 = mtd._compute_ship_pr_at_event(refl_obj, 0, "GPS", detection_time)
        # Geometry is near-zenith with a moving ship — deck plate dominates, RCS finite and positive.
        assert np.isfinite(rcs_ship_m2)
        assert rcs_ship_m2 > 0.0
        assert np.isfinite(pr_ship_w)
        assert pr_ship_w > 0.0


class TestDetectionStructDefaults:
    def test_legacy_path_leaves_ship_fields_none(self):
        # An untouched struct keeps all the per-event Pr fields as None — what the loop will see
        # when ship_dimensions is None (ocean-only / legacy mode).
        s = MovingTargetDetectionStruct()
        assert s.pr_ocean_w is None
        assert s.pr_ship_w is None
        assert s.pr_total_w is None
        assert s.scr_db is None
        assert s.rcs_ocean_m2 is None
        assert s.rcs_ship_m2 is None
