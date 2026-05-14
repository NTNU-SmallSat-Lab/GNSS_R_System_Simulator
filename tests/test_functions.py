"""Unit tests for pure helpers in SRC/functions/functions.py.

Covers the side-effect-free helpers — datetime coercion, vector math,
GNSS-arg normalization, geodetic/ECEF conversion, and the
vector-to-vector rotation matrix. The IO-heavy helpers
(load_gnss_constellation_dict, propagate_all_orbits,
compute_visible_times_and_resample_ship, propagate_only_visible_satellites)
need a propagated constellation and live in higher-tier tests.
"""

import datetime as dt

import numpy as np
import pytest

import functions.functions as fn
from object_definitions.Constants_def import Constants


# --------------------------------------------------------------------------- #
# ensure_utc_aware                                                            #
# --------------------------------------------------------------------------- #

class TestEnsureUtcAware:
    def test_none_returns_none(self):
        assert fn.ensure_utc_aware(None) is None

    def test_naive_datetime_gets_utc_tzinfo(self):
        naive = dt.datetime(2025, 1, 1, 12, 0, 0)
        result = fn.ensure_utc_aware(naive)
        assert result.tzinfo == dt.UTC
        assert result.replace(tzinfo=None) == naive

    def test_utc_aware_passthrough(self):
        aware = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        result = fn.ensure_utc_aware(aware)
        assert result is aware

    def test_other_timezone_is_not_converted(self):
        # ensure_utc_aware preserves the tzinfo if any is set; it does NOT convert non-UTC tz to UTC.
        # This documents the existing contract — to_utc_datetime_array is the conversion helper.
        tz_plus2 = dt.timezone(dt.timedelta(hours=2))
        aware = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=tz_plus2)
        result = fn.ensure_utc_aware(aware)
        assert result.tzinfo == tz_plus2


# --------------------------------------------------------------------------- #
# to_utc_datetime_array                                                       #
# --------------------------------------------------------------------------- #

class TestToUtcDatetimeArray:
    def test_single_datetime_returns_one_element_object_array(self):
        single = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        result = fn.to_utc_datetime_array(single)
        assert isinstance(result, np.ndarray)
        assert result.dtype == object
        assert result.shape == (1,)
        assert result[0] == single

    def test_list_of_naive_datetimes_tagged_utc(self):
        naive_list = [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
        result = fn.to_utc_datetime_array(naive_list)
        assert all(t.tzinfo == dt.UTC for t in result)

    def test_mixed_tz_inputs_converted_to_utc(self):
        tz_plus2 = dt.timezone(dt.timedelta(hours=2))
        mixed = [
            dt.datetime(2025, 1, 1, 14, tzinfo=tz_plus2),  # 12:00 UTC
            dt.datetime(2025, 1, 1, 12, tzinfo=dt.UTC),
        ]
        result = fn.to_utc_datetime_array(mixed)
        assert result[0] == dt.datetime(2025, 1, 1, 12, tzinfo=dt.UTC)
        assert result[1] == dt.datetime(2025, 1, 1, 12, tzinfo=dt.UTC)
        assert all(t.tzinfo == dt.UTC for t in result)

    def test_numpy_datetime64_array_converted(self):
        np_times = np.array(["2025-01-01T12:00:00", "2025-01-02T12:00:00"], dtype="datetime64[us]")
        result = fn.to_utc_datetime_array(np_times)
        assert result.dtype == object
        assert all(isinstance(t, dt.datetime) for t in result)
        assert all(t.tzinfo == dt.UTC for t in result)


# --------------------------------------------------------------------------- #
# angle_between_vectors                                                       #
# --------------------------------------------------------------------------- #

class TestAngleBetweenVectors:
    def test_parallel_vectors_zero_degrees(self):
        v = np.array([1.0, 2.0, 3.0])
        assert fn.angle_between_vectors(v, 2 * v) == pytest.approx(0.0, abs=1e-9)

    def test_antiparallel_vectors_180_degrees(self):
        v = np.array([1.0, 2.0, 3.0])
        assert fn.angle_between_vectors(v, -v) == pytest.approx(180.0, abs=1e-9)

    def test_perpendicular_vectors_90_degrees(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        assert fn.angle_between_vectors(x, y) == pytest.approx(90.0, abs=1e-9)

    def test_radians_unit(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        assert fn.angle_between_vectors(x, y, unit="radians") == pytest.approx(np.pi / 2, abs=1e-9)

    def test_invalid_unit_raises_value_error(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="Unit must be either 'radians' or 'degrees'"):
            fn.angle_between_vectors(x, y, unit="gradians")


# --------------------------------------------------------------------------- #
# skew_matrix                                                                 #
# --------------------------------------------------------------------------- #

class TestSkewMatrix:
    def test_returns_3x3_array(self):
        K = fn.skew_matrix(np.array([1.0, 2.0, 3.0]))
        assert K.shape == (3, 3)

    def test_diagonal_is_zero(self):
        K = fn.skew_matrix(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(np.diag(K), np.zeros(3))

    def test_is_skew_symmetric(self):
        v = np.array([1.5, -2.0, 7.3])
        K = fn.skew_matrix(v)
        np.testing.assert_allclose(K.T, -K, atol=1e-12)

    def test_kv_equals_zero(self):
        # Skew matrix applied to its generating vector gives the zero vector (v x v = 0).
        v = np.array([1.0, 2.0, 3.0])
        K = fn.skew_matrix(v)
        np.testing.assert_allclose(K @ v, np.zeros(3), atol=1e-12)

    def test_kw_equals_cross_product(self):
        # Skew matrix of v applied to w must equal v x w.
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        K = fn.skew_matrix(v)
        np.testing.assert_allclose(K @ w, np.cross(v, w), atol=1e-12)


# --------------------------------------------------------------------------- #
# normalize_gnss_constellation_arg                                            #
# --------------------------------------------------------------------------- #

class TestNormalizeGnssConstellationArg:
    AVAILABLE = ["GPS", "GLONASS", "GALILEO", "BEIDOU"]

    def test_none_returns_all_available(self):
        result = fn.normalize_gnss_constellation_arg(None, self.AVAILABLE)
        assert result == self.AVAILABLE
        assert result is not self.AVAILABLE  # defensive copy

    def test_plain_comma_separated_string(self):
        result = fn.normalize_gnss_constellation_arg("GPS,GALILEO", self.AVAILABLE)
        assert result == ["GPS", "GALILEO"]

    def test_bracketed_string_strips_brackets(self):
        result = fn.normalize_gnss_constellation_arg("[GPS, GALILEO]", self.AVAILABLE)
        assert result == ["GPS", "GALILEO"]

    def test_whitespace_around_items_is_stripped(self):
        result = fn.normalize_gnss_constellation_arg("  GPS  ,  GALILEO  ", self.AVAILABLE)
        assert result == ["GPS", "GALILEO"]

    def test_list_input_returns_list_of_strings(self):
        result = fn.normalize_gnss_constellation_arg(["GPS", "GALILEO"], self.AVAILABLE)
        assert result == ["GPS", "GALILEO"]

    def test_tuple_input_returns_list(self):
        result = fn.normalize_gnss_constellation_arg(("GPS", "GALILEO"), self.AVAILABLE)
        assert result == ["GPS", "GALILEO"]

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="gnss_constellation must be str, list, tuple, or None"):
            fn.normalize_gnss_constellation_arg(42, self.AVAILABLE)


# --------------------------------------------------------------------------- #
# delta_lon_from_distance / delta_lat_from_distance                           #
# --------------------------------------------------------------------------- #

class TestDeltaLonFromDistance:
    def test_zero_distance_gives_zero_delta(self):
        assert fn.delta_lon_from_distance(45.0, 0.0) == 0.0

    def test_at_equator_uses_equatorial_radius(self):
        # At lat=0, cos(lat)=1 and N=a (equatorial radius). delta_lon = x_dist / a (in radians).
        x_dist = 1000.0  # 1 km eastward
        expected_deg = np.degrees(x_dist / Constants.RADIUS_EARTH_EQUAT)
        assert fn.delta_lon_from_distance(0.0, x_dist) == pytest.approx(expected_deg, rel=1e-12)

    def test_sign_propagates(self):
        assert fn.delta_lon_from_distance(45.0, -1000.0) == pytest.approx(
            -fn.delta_lon_from_distance(45.0, 1000.0), rel=1e-12
        )

    def test_higher_latitude_gives_larger_delta_lon(self):
        # Same east-west distance covers more degrees of longitude near the poles.
        x_dist = 1000.0
        assert fn.delta_lon_from_distance(60.0, x_dist) > fn.delta_lon_from_distance(0.0, x_dist)


class TestDeltaLatFromDistance:
    def test_zero_distance_gives_zero_delta(self):
        assert fn.delta_lat_from_distance(45.0, 0.0) == 0.0

    def test_sign_propagates(self):
        assert fn.delta_lat_from_distance(45.0, -1000.0) == pytest.approx(
            -fn.delta_lat_from_distance(45.0, 1000.0), rel=1e-12
        )

    def test_one_km_north_is_about_one_degree_per_111km(self):
        # 1 degree of latitude ~111 km. So 1 km is ~ 1/111 deg, give or take with flattening.
        delta = fn.delta_lat_from_distance(45.0, 1000.0)
        assert 0.008 < delta < 0.010


# --------------------------------------------------------------------------- #
# geodetic_to_ecef                                                            #
# --------------------------------------------------------------------------- #

class TestGeodeticToEcef:
    def test_origin_at_equator_prime_meridian(self):
        X, Y, Z = fn.geodetic_to_ecef(0.0, 0.0, 0.0)
        assert X == pytest.approx(Constants.RADIUS_EARTH_EQUAT, rel=1e-12)
        assert Y == pytest.approx(0.0, abs=1e-6)
        assert Z == pytest.approx(0.0, abs=1e-6)

    def test_equator_90_east(self):
        X, Y, Z = fn.geodetic_to_ecef(0.0, 90.0, 0.0)
        assert X == pytest.approx(0.0, abs=1e-6)
        assert Y == pytest.approx(Constants.RADIUS_EARTH_EQUAT, rel=1e-12)
        assert Z == pytest.approx(0.0, abs=1e-6)

    def test_north_pole_z_equals_polar_radius(self):
        # At lat=90, X=Y=0 and Z = (1-e2)*N = polar radius b = a*(1-f).
        X, Y, _ = fn.geodetic_to_ecef(90.0, 0.0, 0.0)
        assert X == pytest.approx(0.0, abs=1e-6)
        assert Y == pytest.approx(0.0, abs=1e-6)
        # polar radius is a*(1-f); allow small numeric tolerance
        polar_radius = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)
        _, _, Z_pole = fn.geodetic_to_ecef(90.0, 0.0, 0.0)
        assert Z_pole == pytest.approx(polar_radius, rel=1e-6)

    def test_height_adds_radially_at_equator(self):
        h = 100_000.0
        X, _, _ = fn.geodetic_to_ecef(0.0, 0.0, h)
        assert X == pytest.approx(Constants.RADIUS_EARTH_EQUAT + h, rel=1e-12)


# --------------------------------------------------------------------------- #
# rotation_matrix_from_vectors                                                #
# --------------------------------------------------------------------------- #

class TestRotationMatrixFromVectors:
    def test_same_vector_returns_identity(self):
        v = np.array([1.0, 2.0, 3.0])
        R = fn.rotation_matrix_from_vectors(v, v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_aligns_v1_to_v2_direction(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        R = fn.rotation_matrix_from_vectors(v1, v2)
        np.testing.assert_allclose(R @ v1, v2, atol=1e-12)

    def test_rotation_aligns_non_axis_aligned_vectors(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-2.0, 1.0, 4.0])
        R = fn.rotation_matrix_from_vectors(v1, v2)
        # R should rotate the *unit* of v1 onto the unit of v2.
        v1_hat = v1 / np.linalg.norm(v1)
        v2_hat = v2 / np.linalg.norm(v2)
        np.testing.assert_allclose(R @ v1_hat, v2_hat, atol=1e-12)

    def test_returns_orthogonal_matrix_with_det_one(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, -1.0, 2.0])
        R = fn.rotation_matrix_from_vectors(v1, v2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_antiparallel_vectors_give_180_degree_rotation(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        R = fn.rotation_matrix_from_vectors(v1, v2)
        # The rotation should map v1 to v2.
        np.testing.assert_allclose(R @ v1, v2, atol=1e-12)
        # And it should be an involution (R @ R = I) for a 180° rotation.
        np.testing.assert_allclose(R @ R, np.eye(3), atol=1e-12)


# --------------------------------------------------------------------------- #
# color_gnss_const                                                            #
# --------------------------------------------------------------------------- #

class TestColorGnssConst:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("GPS", "red"),
            ("GLONASS", "orange"),
            ("GALILEO", "blue"),
            ("BEIDOU", "green"),
            ("QZSS", "purple"),
            ("IRNSS", "pink"),
        ],
    )
    def test_known_constellation_returns_expected_color(self, name, expected):
        assert fn.color_gnss_const(name) == expected

    def test_unknown_constellation_returns_none(self):
        assert fn.color_gnss_const("NOT_A_CONSTELLATION") is None

    def test_case_sensitive(self):
        # Current contract: only the exact upper-case name is mapped.
        assert fn.color_gnss_const("gps") is None
