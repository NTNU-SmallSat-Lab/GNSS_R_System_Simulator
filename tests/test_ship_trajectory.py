"""Tests for pure-math helpers on ShipTrajectory.

The full ShipTrajectory constructor needs a navigation mesh built from the
Natural Earth ocean shapefile (>100 MB of data + a networkx graph build), so
these tests bypass __init__ via object.__new__ and exercise only the helpers
that depend on their arguments rather than instance state.

Tested:
  * _calculate_haversine_distance (scalar)
  * _calculate_haversine_vectorized (ndarray)
  * calculate_bearing
"""

import datetime as dt
import json

import numpy as np
import pytest

from object_definitions.Constants_def import Constants
from object_definitions.ShipTrajectory_def import ShipTrajectory


@pytest.fixture
def ship():
    """A ShipTrajectory instance that has bypassed __init__ — usable only with the helpers below."""
    return object.__new__(ShipTrajectory)


# --------------------------------------------------------------------------- #
# _calculate_haversine_distance                                               #
# --------------------------------------------------------------------------- #

class TestHaversineDistance:
    def test_zero_distance_for_same_point(self, ship):
        assert ship._calculate_haversine_distance((60.0, 10.0), (60.0, 10.0)) == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_latitude_is_about_111km(self, ship):
        # At the equator, one degree of latitude → πR/180 ≈ 111.319 km.
        distance = ship._calculate_haversine_distance((0.0, 0.0), (1.0, 0.0))
        expected = np.radians(1.0) * Constants.RADIUS_EARTH_EQUAT
        assert distance == pytest.approx(expected, rel=1e-9)

    def test_oslo_to_trondheim_known_distance(self, ship):
        # Spherical-Earth great-circle distance Oslo (59.9139, 10.7522) → Trondheim (63.4305, 10.3951).
        # ~391 km via WGS84 / 392 km via spherical Haversine at R = Constants.RADIUS_EARTH_EQUAT.
        distance = ship._calculate_haversine_distance((59.9139, 10.7522), (63.4305, 10.3951))
        assert distance == pytest.approx(392_000.0, rel=0.01)  # 1% tolerance

    def test_antipodal_points_half_circumference(self, ship):
        # Two antipodal points → π * R_earth.
        distance = ship._calculate_haversine_distance((0.0, 0.0), (0.0, 180.0))
        assert distance == pytest.approx(np.pi * Constants.RADIUS_EARTH_EQUAT, rel=1e-9)

    def test_distance_is_symmetric(self, ship):
        d_forward = ship._calculate_haversine_distance((40.0, -10.0), (50.0, 20.0))
        d_back = ship._calculate_haversine_distance((50.0, 20.0), (40.0, -10.0))
        assert d_forward == pytest.approx(d_back, rel=1e-12)


# --------------------------------------------------------------------------- #
# _calculate_haversine_vectorized                                             #
# --------------------------------------------------------------------------- #

class TestHaversineVectorized:
    def test_returns_array_with_one_element_per_pair(self, ship):
        pos1 = np.array([[0.0, 0.0], [60.0, 10.0]])
        pos2 = np.array([[1.0, 0.0], [60.0, 10.0]])
        distances = ship._calculate_haversine_vectorized(pos1, pos2)
        assert distances.shape == (2,)

    def test_matches_scalar_haversine_for_each_pair(self, ship):
        pos1 = np.array([[0.0, 0.0], [40.0, -74.0], [60.0, 10.0]])
        pos2 = np.array([[1.0, 0.0], [34.0, -118.0], [59.0, 11.0]])
        vec_distances = ship._calculate_haversine_vectorized(pos1, pos2)
        scalar_distances = np.array(
            [ship._calculate_haversine_distance(tuple(p1), tuple(p2)) for p1, p2 in zip(pos1, pos2, strict=True)]
        )
        np.testing.assert_allclose(vec_distances, scalar_distances, rtol=1e-12)

    def test_handles_clipping_for_numerically_tiny_negative_a(self, ship):
        # When pos1 == pos2 the haversine 'a' can drift slightly below zero in float; the
        # np.clip(a, 0, 1) guard ensures the result is exactly 0 rather than NaN from sqrt.
        pos = np.array([[59.9139, 10.7522]])
        distances = ship._calculate_haversine_vectorized(pos, pos)
        assert distances[0] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# calculate_bearing                                                           #
# --------------------------------------------------------------------------- #

class TestCalculateBearing:
    def test_due_north_bearing_is_zero(self, ship):
        bearing = ship.calculate_bearing((0.0, 0.0), (1.0, 0.0))
        assert bearing == pytest.approx(0.0, abs=1e-9)

    def test_due_east_bearing_is_90(self, ship):
        bearing = ship.calculate_bearing((0.0, 0.0), (0.0, 1.0))
        assert bearing == pytest.approx(90.0, abs=1e-9)

    def test_due_south_bearing_is_180(self, ship):
        bearing = ship.calculate_bearing((1.0, 0.0), (0.0, 0.0))
        assert bearing == pytest.approx(180.0, abs=1e-9)

    def test_due_west_bearing_is_270(self, ship):
        bearing = ship.calculate_bearing((0.0, 1.0), (0.0, 0.0))
        assert bearing == pytest.approx(270.0, abs=1e-9)

    def test_bearing_always_in_zero_to_360(self, ship):
        # Random offsets that would yield a negative atan2 result before the (b + 360) % 360 step.
        bearings = [
            ship.calculate_bearing((10.0, 20.0), (5.0, 19.0)),
            ship.calculate_bearing((50.0, 0.0), (40.0, -30.0)),
            ship.calculate_bearing((-30.0, 100.0), (-40.0, 120.0)),
        ]
        for b in bearings:
            assert 0.0 <= b < 360.0


# --------------------------------------------------------------------------- #
# trajectory cache loading                                                    #
# --------------------------------------------------------------------------- #

def _configured_cache_ship() -> ShipTrajectory:
    ship = object.__new__(ShipTrajectory)
    ship.ship_id = "CacheShip"
    ship.start_position = [54.0, 7.0]
    ship.end_position = [69.0, 33.0]
    ship.start_time = dt.datetime(2025, 5, 17, 12, 0, 0, tzinfo=dt.UTC)
    ship.ship_speed = 15.0 * 0.51444
    ship.interval_minutes = 10
    ship.fast_method = False
    ship.grid_resolution = 150_000.0
    ship.mesh_connection_radius = 300_000.0
    ship.water_buffer = 25_000.0
    ship.delta_m = 500_000.0
    ship.shapefile_path = {}
    return ship


def _matching_metadata(ship: ShipTrajectory) -> dict:
    return {
        "ship_id": ship.ship_id,
        "start_position": ship.start_position,
        "end_position": ship.end_position,
        "start_time": ship.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "ship_speed_m_s": ship.ship_speed,
        "time_interval_minutes": ship.interval_minutes,
        "fast_method": ship.fast_method,
        "mesh_connection_radius": ship.mesh_connection_radius,
        "water_buffer": ship.water_buffer,
        "grid_resolution": ship.grid_resolution,
        "delta_m": ship.delta_m,
    }


class TestTrajectoryCacheLoading:
    def test_matching_metadata_loads_cache(self, tmp_path, monkeypatch):
        ship = _configured_cache_ship()
        monkeypatch.setattr(Constants, "autogen_ship_shapefiles", str(tmp_path))
        (tmp_path / "CacheShip_trajectory.shp").write_text("")
        (tmp_path / "CacheShip_trajectory_metadata.json").write_text(json.dumps(_matching_metadata(ship)))

        called = {"load": False}
        monkeypatch.setattr(ship, "load_class_from_file", lambda: called.__setitem__("load", True))
        monkeypatch.setattr(ship, "plan_route", lambda _save_graph: pytest.fail("matching cache should not regenerate"))

        ship.initialize(None, save_graph=False)

        assert called["load"]

    def test_changed_metadata_still_loads_existing_cache(self, tmp_path, monkeypatch):
        ship = _configured_cache_ship()
        monkeypatch.setattr(Constants, "autogen_ship_shapefiles", str(tmp_path))
        (tmp_path / "CacheShip_trajectory.shp").write_text("")
        metadata = _matching_metadata(ship)
        metadata["ship_speed_m_s"] = ship.ship_speed + 1.0
        (tmp_path / "CacheShip_trajectory_metadata.json").write_text(json.dumps(metadata))

        called = {"load": False}
        monkeypatch.setattr(ship, "load_class_from_file", lambda: called.__setitem__("load", True))
        monkeypatch.setattr(ship, "plan_route", lambda _save_graph: pytest.fail("existing cache should load, not regenerate"))

        ship.initialize(None, save_graph=False)

        assert called["load"]
