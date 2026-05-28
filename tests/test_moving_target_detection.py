"""Tests for the pure helpers on MovingTargetDetection.

Full MovingTargetDetection construction needs a propagated constellation and a
ShipTrajectory (which itself needs the Natural Earth ocean shapefile). These
tests target only the bits that don't need that state:

  * _find_closest_time — instance method, but parameter-only; bypasses __init__
  * _polygon_within_map — static method; tested directly
"""

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

import object_definitions.MovingTargetDetection_def as mtd_module
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
# visible propagation selection                                               #
# --------------------------------------------------------------------------- #

class TestVisiblePropagationSelection:
    def test_reflection_times_use_resampled_visible_times_not_satellite_zero(self, mtd):
        visible_time = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        mtd.timestamps_resampled = [visible_time]
        mtd.sat_constellation = SimpleNamespace(satellites=[SimpleNamespace()])

        assert mtd._get_reflection_times() == [visible_time]

    def test_satellite_without_propagated_state_is_not_visible_propagated(self):
        sat = SimpleNamespace()
        assert not MovingTargetDetection._satellite_has_visible_propagation(sat, expected_time_count=2)

    def test_satellite_with_only_nan_synced_state_is_not_visible_propagated(self):
        sat = SimpleNamespace(
            time_prop_vec=[object(), object()],
            positions_eci=np.full((3, 2), np.nan),
            velocities_eci=np.full((3, 2), np.nan),
        )
        assert not MovingTargetDetection._satellite_has_visible_propagation(sat, expected_time_count=2)

    def test_satellite_with_one_finite_synced_state_is_visible_propagated(self):
        sat = SimpleNamespace(
            time_prop_vec=[object(), object()],
            positions_eci=np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]),
            velocities_eci=np.array([[4.0, np.nan], [5.0, np.nan], [6.0, np.nan]]),
        )
        assert MovingTargetDetection._satellite_has_visible_propagation(sat, expected_time_count=2)


# --------------------------------------------------------------------------- #
# reflection plotting bounds                                                  #
# --------------------------------------------------------------------------- #

class _FakeAxis:
    def set_extent(self, *_args, **_kwargs):
        pass

    def set_title(self, *_args, **_kwargs):
        pass

    def set_xlabel(self, *_args, **_kwargs):
        pass

    def set_ylabel(self, *_args, **_kwargs):
        pass

    def add_feature(self, *_args, **_kwargs):
        pass

    def stock_img(self):
        pass

    def gridlines(self, *_args, **_kwargs):
        return SimpleNamespace(top_labels=True, right_labels=True)

    def plot(self, *_args, **_kwargs):
        pass

    def text(self, *_args, **_kwargs):
        pass

    def scatter(self, *_args, **_kwargs):
        pass

    def legend(self, *_args, **_kwargs):
        pass


class _FakeFigure:
    canvas = SimpleNamespace(
        draw=lambda: None,
        tostring_argb=lambda: bytes([0, 0, 0, 255]),
        get_width_height=lambda: (1, 1),
    )


class TestShowReflectionsAtTime:
    def test_uses_buffered_bounds_without_undefined_names(self, mtd, monkeypatch):
        time_i = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        refl_obj = SimpleNamespace(
            refl_ECI_vec=np.array([[1.0, 2.0, 3.0]]),
            refl_LL_vec=np.array([[60.0, 5.0]]),
            tx_satID=1,
            rx_satID=2,
        )
        reflection = SimpleNamespace(refl_obj=refl_obj, time_index=0)
        mtd.reflection_dict_timeSorted = {"GPS": {time_i: [reflection]}}
        mtd.gdf_time_series = pd.DataFrame({"datetime_s": ["2025-01-01 12:00:00"], "geometry": [Point(5.0, 60.0)]})
        mtd.bounding_box = (4.0, 6.0, 59.0, 61.0)
        mtd.bbox_buffer_deg = 1.0
        mtd.moving_target_object = SimpleNamespace(trajectory_points=[(60.0, 5.0)])
        mtd.sat_antenna = SimpleNamespace(frequencies=[])

        monkeypatch.setattr(mtd_module.plt, "subplots", lambda *args, **kwargs: (_FakeFigure(), _FakeAxis()))
        monkeypatch.setattr(mtd_module.plt, "gca", lambda: SimpleNamespace(get_legend_handles_labels=lambda: ([], [])))
        monkeypatch.setattr(mtd_module.plt, "tight_layout", lambda: None)
        monkeypatch.setattr(mtd_module.plt, "close", lambda *_args, **_kwargs: None)

        mtd.show_reflections_and_target_position_at_time_i(time_i, label_reflections=False)


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
