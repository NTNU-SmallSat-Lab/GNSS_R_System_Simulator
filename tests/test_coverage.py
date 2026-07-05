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
import datetime as dt
import io
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from object_definitions.Coverage_def import Coverage, backscatter_swath_half_width_m

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


# --------------------------------------------------------------------------- #
# Backscatter area coverage                                                   #
# --------------------------------------------------------------------------- #

def _build_coverage(input_shapefile_path, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return Coverage(
            sat_const=None,
            gnss_const={"GPS": SimpleNamespace()},
            shapefile=input_shapefile_path,
            grid_size_deg=2.5,
            overwrite_shapefile=True,
            max_central_angle_deg=80.0,
            **kwargs,
        )


class TestBackscatterSwathHalfWidth:
    def test_representative_leo_gnss_geometry(self):
        # beta_max = 5 deg, R_r = 1000 km, R_t = 20 000 km:
        # r = radians(5) / (1/R_r - 1/R_t) = 0.0872665 / 9.5e-7 ~= 91.9 km.
        r = backscatter_swath_half_width_m(1.0e6, 2.0e7, 5.0)
        np.testing.assert_allclose(float(r), np.radians(5.0) / (1.0 / 1.0e6 - 1.0 / 2.0e7), rtol=1e-12)
        assert 9.0e4 < float(r) < 9.5e4

    def test_distant_transmitter_limit_is_beta_times_range(self):
        # R_t -> inf: the acceptance half-width collapses to beta_max * R_r.
        r = backscatter_swath_half_width_m(1.0e6, 1.0e15, 5.0)
        np.testing.assert_allclose(float(r), np.radians(5.0) * 1.0e6, rtol=1e-6)

    def test_wider_acceptance_gives_wider_swath(self):
        narrow = float(backscatter_swath_half_width_m(1.0e6, 2.0e7, 5.0))
        wide = float(backscatter_swath_half_width_m(1.0e6, 2.0e7, 10.0))
        assert wide > narrow

    def test_equal_ranges_yield_nan(self):
        # Tx at the receiver range: the curvature difference vanishes, no finite swath.
        assert np.isnan(float(backscatter_swath_half_width_m(1.0e6, 1.0e6, 5.0)))

    def test_out_of_range_angle_raises(self):
        with pytest.raises(ValueError, match="max_bistatic_bksc_angle_deg"):
            backscatter_swath_half_width_m(1.0e6, 2.0e7, 0.0)
        with pytest.raises(ValueError, match="max_bistatic_bksc_angle_deg"):
            backscatter_swath_half_width_m(1.0e6, 2.0e7, 90.0)


class TestBackscatterCoverageValidation:
    def test_backscatter_requires_incidence_gate(self, input_shapefile_path):
        with pytest.raises(ValueError, match="max_incidence_angle_deg"):
            _build_coverage(input_shapefile_path, reflection_type="backscatter", max_bistatic_bksc_angle_deg=5.0)

    def test_backscatter_requires_bistatic_angle(self, input_shapefile_path):
        with pytest.raises(ValueError, match="max_bistatic_bksc_angle_deg"):
            _build_coverage(input_shapefile_path, reflection_type="backscatter", max_incidence_angle_deg=45.0)

    def test_bistatic_angle_range_validated(self, input_shapefile_path):
        with pytest.raises(ValueError, match="max_bistatic_bksc_angle_deg"):
            _build_coverage(
                input_shapefile_path, reflection_type="backscatter", max_incidence_angle_deg=45.0, max_bistatic_bksc_angle_deg=95.0
            )

    def test_invalid_reflection_type_raises(self, input_shapefile_path):
        with pytest.raises(ValueError, match="reflection_type"):
            _build_coverage(input_shapefile_path, reflection_type="sideways")

    def test_specular_warns_on_unused_bistatic_angle(self, input_shapefile_path):
        with pytest.warns(UserWarning, match="max_bistatic_bksc_angle_deg"):
            _build_coverage(input_shapefile_path, max_bistatic_bksc_angle_deg=5.0)


class TestBackscatterSwathCoverage:
    """A synthetic BRP track through the grid: the buffered swath must cover cells the thin line misses."""

    RX_ALT_M = 1.0e6
    TX_ALT_M = 2.0e7

    def _fake_brp_track(self, lats, lons):
        # Surface points on a sphere with Rx and Tx stacked radially above each point (exact retro
        # geometry), so the surface-to-satellite slant ranges equal the stacked altitudes.
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        R = 6.371e6
        clat, slat = np.cos(np.radians(lats)), np.sin(np.radians(lats))
        clon, slon = np.cos(np.radians(lons)), np.sin(np.radians(lons))
        up = np.column_stack((clat * clon, clat * slon, slat))          # (n, 3) radial unit vectors
        n = len(lats)
        return SimpleNamespace(
            refl_LL_vec=np.column_stack((lats, lons)),
            refl_ECI_vec=R * up,
            sat_pos_vec=((R + self.RX_ALT_M) * up).T,                   # (3, n)
            gnss_pos_vec=((R + self.TX_ALT_M) * up).T,                  # (3, n)
            refl_ang_vec=np.zeros(n),
            power=np.ones(n),
            tx_satID="G01",
            rx_satID="LEO-1",
        )

    def _visited_cells(self, coverage):
        return {idx for idx in range(16) if len(coverage.coverage_dict["GPS"][idx].times) > 0}

    def _run(self, coverage):
        # North-south track at lon 5.1 deg: the thin line stays inside one 2.5-deg cell column, while
        # the ~92 km (~0.83 deg) swath reaches into the neighbouring column on the west side.
        lats = np.linspace(0.5, 9.5, 19)
        lons = np.full_like(lats, 5.1)
        track = self._fake_brp_track(lats, lons)
        times = [dt.datetime(2026, 5, 27, 12, 0, tzinfo=dt.UTC) + dt.timedelta(seconds=10 * k) for k in range(len(lats))]
        coverage.reflection_dict["GPS"] = [track]
        coverage._calculate_coverage(times, "GPS")
        return self._visited_cells(coverage)

    def test_thin_specular_track_covers_one_cell_column(self, input_shapefile_path):
        coverage = _build_coverage(input_shapefile_path)
        assert len(self._run(coverage)) == 4                            # 4 rows x 1 column

    def test_backscatter_swath_covers_neighbouring_column(self, input_shapefile_path):
        coverage = _build_coverage(
            input_shapefile_path, reflection_type="backscatter", max_incidence_angle_deg=45.0, max_bistatic_bksc_angle_deg=5.0
        )
        visited = self._run(coverage)
        assert len(visited) == 8                                        # 4 rows x 2 columns: swath adds the west column

    def test_swath_width_matches_the_analytic_half_width(self, input_shapefile_path):
        coverage = _build_coverage(
            input_shapefile_path, reflection_type="backscatter", max_incidence_angle_deg=45.0, max_bistatic_bksc_angle_deg=5.0
        )
        lats = np.linspace(0.5, 9.5, 19)
        track = self._fake_brp_track(lats, np.full_like(lats, 5.1))
        cluster = np.arange(len(lats))
        swath_deg = coverage._cluster_swath_half_width_deg(track, cluster, track.refl_LL_vec)
        expected_m = np.radians(5.0) / (1.0 / self.RX_ALT_M - 1.0 / self.TX_ALT_M)
        assert 0.7 < swath_deg < 1.0                                    # ~91.9 km at mid latitudes
        np.testing.assert_allclose(swath_deg * 111.0e3, expected_m, rtol=0.02)  # ~111 km per degree latitude
