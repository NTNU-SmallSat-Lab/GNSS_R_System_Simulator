"""
Tests for Coverage — gridding, validity-mask filtering, cluster splitting.

What's covered:
  * Static helpers: validity_mask, _get_reflection_clusters
  * Grid generation: _create_coverage_shapefile_deg, _create_coverage_shapefile_m
  * Bounding-box padding behavior

The full calculate_reflection_points_and_coverage pipeline (which propagates
satellites and runs the link-budget calculation per pair) is integration-scope
and not exercised here.
"""

import contextlib
import io
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from object_definitions.Coverage_def import Coverage

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def input_shapefile_path(tmp_path):
    """
    Write a single 10°x10° rectangular polygon to a shapefile and return its absolute path.

    The polygon spans (lon: 0..10, lat: 0..10) — a clean grid-cell-aligned region that lets the
    cell-count assertions remain integer-exact regardless of latitude scaling.
    """
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
    shp_path = tmp_path / "region.shp"
    gdf.to_file(str(shp_path), driver="ESRI Shapefile")
    return str(shp_path)


@pytest.fixture
def coverage_deg(input_shapefile_path):
    """
    Coverage object with a 2.5°x2.5° equiangular grid → exactly 16 cells over the 10x10 region.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return Coverage(
            sat_const=None,                  # stored, not used in __init__
            gnss_const={"GPS": SimpleNamespace()},  # iterated for keys only
            shapefile=input_shapefile_path,
            grid_size_deg=2.5,
            overwrite_shapefile=True,
            max_central_angle_deg=80.0,
        )


# --------------------------------------------------------------------------- #
# validity_mask                                                               #
# --------------------------------------------------------------------------- #

class TestValidityMask:
    BBOX = (-10.0, 10.0, 30.0, 70.0)  # (min_lon, max_lon, min_lat, max_lat)

    def _make_refl(self, lats_lons):
        # Coverage.validity_mask only reads .refl_LL_vec from the reflection-like input.
        return SimpleNamespace(refl_LL_vec=np.array(lats_lons))

    def test_all_inside_bbox_mask_all_true(self):
        refl = self._make_refl([[40.0, 0.0], [50.0, 5.0], [60.0, -5.0]])
        mask, in_box = Coverage.validity_mask(refl, self.BBOX)
        assert mask.tolist() == [True, True, True]
        np.testing.assert_array_equal(in_box, refl.refl_LL_vec)

    def test_all_outside_bbox_mask_all_false_and_nan_filled(self):
        refl = self._make_refl([[80.0, 0.0], [20.0, 0.0], [50.0, 50.0]])
        mask, in_box = Coverage.validity_mask(refl, self.BBOX)
        assert mask.tolist() == [False, False, False]
        assert np.isnan(in_box).all()

    def test_all_nan_input_short_circuits_to_all_false(self):
        refl = self._make_refl([[np.nan, np.nan]] * 4)
        mask, in_box = Coverage.validity_mask(refl, self.BBOX)
        # The branch returns a zeros mask and an all-NaN copy without touching the bbox math.
        assert not mask.any()
        assert np.isnan(in_box).all()

    def test_mixed_in_out_yields_partial_mask(self):
        refl = self._make_refl([[40.0, 0.0], [80.0, 0.0], [50.0, 5.0], [50.0, 50.0]])
        mask, in_box = Coverage.validity_mask(refl, self.BBOX)
        assert mask.tolist() == [True, False, True, False]
        # Inside points pass through; outside points become NaN.
        assert (in_box[0] == refl.refl_LL_vec[0]).all()
        assert np.isnan(in_box[1]).all()
        assert (in_box[2] == refl.refl_LL_vec[2]).all()
        assert np.isnan(in_box[3]).all()

    def test_bbox_edges_are_inclusive(self):
        # Lat boundary at exactly 30 and 70, lon boundary at exactly -10 and 10 — all must be True.
        refl = self._make_refl([[30.0, -10.0], [70.0, 10.0]])
        mask, _ = Coverage.validity_mask(refl, self.BBOX)
        assert mask.tolist() == [True, True]

    def test_none_bbox_accepts_all_finite_points(self):
        refl = self._make_refl([[40.0, 0.0], [np.nan, np.nan], [-80.0, 170.0]])
        mask, filtered = Coverage.validity_mask(refl, None)
        assert mask.tolist() == [True, False, True]
        assert np.isnan(filtered[1]).all()


# --------------------------------------------------------------------------- #
# _get_reflection_clusters                                                    #
# --------------------------------------------------------------------------- #

class TestGetReflectionClusters:
    def _refl_with_n_points(self, n):
        # Only len(reflection_obj.refl_LL_vec) is read; values don't matter for the split.
        return SimpleNamespace(refl_LL_vec=np.zeros((n, 2)))

    def test_all_valid_yields_a_single_run(self):
        refl = self._refl_with_n_points(5)
        mask = np.ones(5, dtype=bool)
        clusters = Coverage._get_reflection_clusters(refl, mask)
        # np.split inserts an empty list when there's no split point; one cluster has the indices.
        non_empty = [c for c in clusters if len(c) > 0]
        assert len(non_empty) == 1
        np.testing.assert_array_equal(non_empty[0], np.arange(5))

    def test_invalid_gap_in_middle_creates_two_runs(self):
        # Indices 0,1 valid; 2 invalid; 3,4 valid. Should give two non-trivial clusters.
        mask = np.array([True, True, False, True, True])
        refl = self._refl_with_n_points(5)
        clusters = Coverage._get_reflection_clusters(refl, mask)
        assert [list(c) for c in clusters] == [[0, 1], [3, 4]]

    def test_all_invalid_returns_no_full_cluster(self):
        mask = np.zeros(5, dtype=bool)
        refl = self._refl_with_n_points(5)
        clusters = Coverage._get_reflection_clusters(refl, mask)
        assert clusters == []

    def test_empty_reflection_returns_empty_clusters(self):
        refl = self._refl_with_n_points(0)
        clusters = Coverage._get_reflection_clusters(refl, np.array([], dtype=bool))
        assert clusters == []


# --------------------------------------------------------------------------- #
# Grid generation (equiangular)                                               #
# --------------------------------------------------------------------------- #

class TestCoverageEquiangularGrid:
    def test_grid_cell_count_matches_expected(self, coverage_deg):
        # 10°x10° region, 2.5° grid → 4 cells per side → 16 cells total.
        # (Boundary cells will exactly cover the region because both dimensions are multiples of 2.5.)
        assert len(coverage_deg.grid_gdf) == 16

    def test_grid_type_is_equiangular(self, coverage_deg):
        assert coverage_deg.grid_type == "equiangular"

    def test_gridded_shapefile_uses_deg_suffix(self, coverage_deg):
        assert coverage_deg.gridded_shapefile_path.endswith("_gridded_deg.shp")

    def test_coverage_dict_initialized_per_constellation(self, coverage_deg):
        # gnss_const = {"GPS": ...} so coverage_dict has one entry per cell under "GPS".
        assert "GPS" in coverage_deg.coverage_dict
        assert len(coverage_deg.coverage_dict["GPS"]) == 16

    def test_each_coverage_cell_has_position_set(self, coverage_deg):
        for cell in coverage_deg.coverage_dict["GPS"]:
            assert cell.position is not None
            # Centroid x/y were tuples set from grid_gdf — sanity check.
            x, y = cell.position
            assert 0.0 <= x <= 10.0
            assert 0.0 <= y <= 10.0


# --------------------------------------------------------------------------- #
# Grid generation (equidistant, meters)                                       #
# --------------------------------------------------------------------------- #

class TestCoverageEquidistantGrid:
    def test_meter_grid_creates_cells(self, input_shapefile_path):
        with contextlib.redirect_stdout(io.StringIO()):
            coverage = Coverage(
                sat_const=None,
                gnss_const={"GPS": SimpleNamespace()},
                shapefile=input_shapefile_path,
                grid_size_m=200_000,  # 200 km cells
                overwrite_shapefile=True,
                max_central_angle_deg=80.0,
            )
        # A 10°x10° region near the equator (~1100 km on a side) with 200 km cells → at least a
        # handful of cells; the exact count depends on lat-scaling and intersection filtering,
        # so we assert a meaningful range rather than an exact integer.
        assert len(coverage.grid_gdf) > 10
        assert len(coverage.grid_gdf) < 200

    def test_grid_type_is_equidistant(self, input_shapefile_path):
        with contextlib.redirect_stdout(io.StringIO()):
            coverage = Coverage(
                sat_const=None,
                gnss_const={"GPS": SimpleNamespace()},
                shapefile=input_shapefile_path,
                grid_size_m=200_000,
                overwrite_shapefile=True,
                max_central_angle_deg=80.0,
            )
        assert coverage.grid_type == "equidistant"
        assert coverage.gridded_shapefile_path.endswith("_gridded_m.shp")


# --------------------------------------------------------------------------- #
# _get_bounding_box                                                           #
# --------------------------------------------------------------------------- #

class TestGetBoundingBox:
    def test_equiangular_returns_none(self, coverage_deg):
        # Equiangular grids don't carry a buffered bounding box.
        assert coverage_deg.bounding_box is None

    def test_equidistant_bounding_box_pads_by_5_degrees(self, input_shapefile_path):
        with contextlib.redirect_stdout(io.StringIO()):
            coverage = Coverage(
                sat_const=None,
                gnss_const={"GPS": SimpleNamespace()},
                shapefile=input_shapefile_path,
                grid_size_m=200_000,
                overwrite_shapefile=True,
                max_central_angle_deg=80.0,
            )
        bbox = coverage.bounding_box
        assert bbox is not None
        min_lon, max_lon, min_lat, max_lat = bbox
        # The input polygon spans (0..10, 0..10). After 5° padding (and floor/ceil) the bbox
        # should be at most -5..15 in both axes — meters-based grids may shrink slightly due
        # to the intersection filter, but the padding always overshoots the original bounds.
        assert min_lon <= -5
        assert max_lon >= 15
        assert min_lat <= -5
        assert max_lat >= 15


# --------------------------------------------------------------------------- #
# earth_model validation                                                      #
# --------------------------------------------------------------------------- #

class TestEarthModelValidation:
    def test_invalid_earth_model_raises_value_error(self, input_shapefile_path):
        with pytest.raises(ValueError, match="earth_model"):
            with contextlib.redirect_stdout(io.StringIO()):
                Coverage(
                    sat_const=None,
                    gnss_const={"GPS": SimpleNamespace()},
                    shapefile=input_shapefile_path,
                    grid_size_deg=2.5,
                    overwrite_shapefile=True,
                    earth_model="NOT_A_VALID_MODEL",
                    max_central_angle_deg=80.0,
                )
