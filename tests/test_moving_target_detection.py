"""Tests for the pure helpers on MovingTargetDetection.

Full MovingTargetDetection construction needs a propagated constellation and a
ShipTrajectory (which itself needs the Natural Earth ocean shapefile). These
tests target only the bits that don't need that state:

  * _find_closest_time — instance method, but parameter-only; bypasses __init__
  * _polygon_within_map — static method; tested directly
"""

import datetime as dt

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from object_definitions.MovingTargetDetection_def import MovingTargetDetection


@pytest.fixture
def mtd():
    """A MovingTargetDetection instance that has bypassed __init__."""
    return object.__new__(MovingTargetDetection)


# --------------------------------------------------------------------------- #
# _find_closest_time                                                          #
# --------------------------------------------------------------------------- #

class TestFindClosestTime:
    def _times_sec(self, start_utc, step_sec, n):
        # Build a Unix-second array equivalent to start_utc + k*step_sec for k in range(n).
        base = start_utc.timestamp()
        return np.array([base + k * step_sec for k in range(n)])

    def test_exact_match_returns_zero_delta(self, mtd):
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        times_sec = self._times_sec(start, 30, 10)
        target = start + dt.timedelta(seconds=30 * 3)
        idx, delta = mtd._find_closest_time(target, times_sec)
        assert idx == 3
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_returns_nearest_when_between_samples(self, mtd):
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        times_sec = self._times_sec(start, 30, 10)
        # 9 s after sample 3 → still closer to sample 3 (delta 9s) than sample 4 (delta 21s).
        target = start + dt.timedelta(seconds=30 * 3 + 9)
        idx, delta = mtd._find_closest_time(target, times_sec)
        assert idx == 3
        assert delta == pytest.approx(9.0, abs=1e-9)

    def test_returns_later_sample_when_target_is_closer_to_it(self, mtd):
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        times_sec = self._times_sec(start, 30, 10)
        # 21 s after sample 3 (= 9 s before sample 4) → sample 4 is closer.
        target = start + dt.timedelta(seconds=30 * 3 + 21)
        idx, _ = mtd._find_closest_time(target, times_sec)
        assert idx == 4

    def test_target_before_first_sample_returns_first_index(self, mtd):
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        times_sec = self._times_sec(start, 30, 10)
        target = start - dt.timedelta(seconds=120)
        idx, delta = mtd._find_closest_time(target, times_sec)
        assert idx == 0
        assert delta == pytest.approx(120.0, abs=1e-9)

    def test_target_after_last_sample_returns_last_index(self, mtd):
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        times_sec = self._times_sec(start, 30, 10)
        target = start + dt.timedelta(seconds=30 * 9 + 500)
        idx, delta = mtd._find_closest_time(target, times_sec)
        assert idx == 9
        assert delta == pytest.approx(500.0, abs=1e-9)

    def test_naive_target_is_treated_as_utc(self, mtd):
        # ensure_utc_aware tags naive datetimes as UTC — the result must match the aware case.
        start = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        naive_target = dt.datetime(2025, 1, 1, 12, 1, 30)  # 90 s after start, naive
        times_sec = self._times_sec(start, 30, 10)
        idx, delta = mtd._find_closest_time(naive_target, times_sec)
        assert idx == 3  # 90 s / 30 s = sample 3
        assert delta == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# _polygon_within_map                                                         #
# --------------------------------------------------------------------------- #

class TestPolygonWithinMap:
    def test_polygon_fully_inside_bbox_returned_unchanged_in_area(self):
        # A polygon strictly inside the bbox should clip to itself (same area).
        polygon = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        clipped = MovingTargetDetection._polygon_within_map(polygon, -10, 10, -10, 10)
        assert clipped.area == pytest.approx(polygon.area, rel=1e-12)

    def test_polygon_fully_outside_bbox_returns_empty(self):
        polygon = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
        clipped = MovingTargetDetection._polygon_within_map(polygon, -10, 10, -10, 10)
        assert clipped.is_empty

    def test_polygon_straddling_bbox_clips_to_overlap(self):
        # 0..20 east-west, 0..10 north-south. Bbox 0..10 x 0..10 → clipped area is 10*10 = 100.
        polygon = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
        clipped = MovingTargetDetection._polygon_within_map(polygon, 0, 10, 0, 10)
        assert clipped.area == pytest.approx(100.0, rel=1e-9)

    def test_empty_input_polygon_passes_through(self):
        empty = Polygon()
        clipped = MovingTargetDetection._polygon_within_map(empty, -10, 10, -10, 10)
        assert clipped.is_empty

    def test_invalid_self_intersecting_polygon_is_buffered(self):
        # A bowtie polygon is invalid (self-intersecting); the function repairs it via buffer(0).
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        assert not bowtie.is_valid
        clipped = MovingTargetDetection._polygon_within_map(bowtie, -10, 10, -10, 10)
        # After buffer(0) repair → either a Polygon or MultiPolygon, but valid in either case.
        assert isinstance(clipped, Polygon | MultiPolygon)
        assert clipped.is_valid
