"""Geometry-level tests for Reflection — no constellation propagation required.

What's covered:
  * Validators: _is_valid_earth_model, _check_los_satellites, _prevent_longitude_wraparound
  * Geometric primitives: get_rotation_3D_to_plane_coords, get_reflection_point2d
  * Spherical reflection solver: get_reflection_point_ff
  * Elliptical reflection solver: _find_specular_point_ellipse_ecef
  * Spherical vs elliptical agreement at low flattening difference

compute_reflection_coordinates requires propagated Satellite objects and a link-budget
configuration; it's better covered by a higher-tier integration test.

The construction-only tests live in test_reflection.py; this module focuses on physics.
"""

import contextlib
import io

import numpy as np
import pytest

from object_definitions.Constants_def import Constants
from object_definitions.Reflection_def import Reflection


# A LEO altitude (~500 km) and GNSS altitude (~MEO, ~20200 km) chosen so both
# satellites are well above the Earth and produce a clear reflection geometry.
R_LEO = Constants.RADIUS_EARTH_EQUAT + 500e3
R_GNSS = Constants.RADIUS_EARTH_EQUAT + 20_200e3


@pytest.fixture
def refl():
    """A Reflection object with sane defaults; construction noise is suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return Reflection(tx_satID=1, rx_satID=2, earth_model="spherical")


# --------------------------------------------------------------------------- #
# _is_valid_earth_model                                                       #
# --------------------------------------------------------------------------- #

class TestIsValidEarthModel:
    @pytest.mark.parametrize("model", ["spherical", "elliptical"])
    def test_valid_models_accepted(self, model):
        assert Reflection._is_valid_earth_model(model) is True

    @pytest.mark.parametrize("model", ["sperical", "", "SPHERICAL", "Spherical", None])
    def test_invalid_models_rejected(self, model):
        assert Reflection._is_valid_earth_model(model) is False


# --------------------------------------------------------------------------- #
# _check_los_satellites                                                       #
# --------------------------------------------------------------------------- #

class TestCheckLosSatellites:
    def test_same_side_of_earth_has_los(self):
        # Two LEO points roughly above the equator at the same longitude → clear sight.
        r1 = np.array([R_LEO, 0.0, 0.0])
        r2 = np.array([R_LEO, 0.0, 100e3])  # 100 km north
        assert Reflection._check_los_satellites(r1, r2) is True

    def test_opposite_sides_through_earth_blocked(self):
        # Antipodal LEOs → the straight line crosses through the Earth's center.
        r1 = np.array([R_LEO, 0.0, 0.0])
        r2 = np.array([-R_LEO, 0.0, 0.0])
        assert Reflection._check_los_satellites(r1, r2) is False

    def test_leo_to_gnss_unblocked_when_geometry_clear(self):
        # LEO at (R_LEO, 0, 0); GNSS at 45 deg above the same plane.
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        assert Reflection._check_los_satellites(leo, gnss) is True

    def test_leo_to_far_side_gnss_blocked(self):
        # LEO at +x; GNSS on opposite side at −x → blocked by Earth.
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([-R_GNSS, 0.0, 0.0])
        assert Reflection._check_los_satellites(leo, gnss) is False


# --------------------------------------------------------------------------- #
# _prevent_longitude_wraparound                                               #
# --------------------------------------------------------------------------- #

class TestPreventLongitudeWraparound:
    def test_no_wraparound_passes_through_unchanged(self):
        lons = np.array([10.0, 11.0, 12.0, 13.0])
        lats = np.array([60.0, 60.5, 61.0, 61.5])
        out_lons, out_lats = Reflection._prevent_longitude_wraparound(lons.copy(), lats.copy())
        np.testing.assert_array_equal(out_lons, lons)
        np.testing.assert_array_equal(out_lats, lats)

    def test_single_wraparound_replaces_post_jump_point_with_nan(self):
        # Jump from +179 to −179 (eastbound crossing of the antimeridian).
        lons = np.array([178.0, 179.0, -179.0, -178.0])
        lats = np.array([60.0, 60.0, 60.0, 60.0])
        out_lons, out_lats = Reflection._prevent_longitude_wraparound(lons.copy(), lats.copy())
        # The jump is between indices 1 and 2 → index 2 should be NaN.
        assert np.isnan(out_lons[2])
        assert np.isnan(out_lats[2])
        # Surrounding points untouched.
        assert out_lons[0] == 178.0
        assert out_lons[1] == 179.0
        assert out_lons[3] == -178.0

    def test_multiple_wraparounds_all_replaced(self):
        lons = np.array([170.0, -170.0, 170.0, -170.0])  # ping-pong across antimeridian
        lats = np.array([60.0, 60.0, 60.0, 60.0])
        out_lons, _ = Reflection._prevent_longitude_wraparound(lons.copy(), lats.copy())
        # All three jumps exceed 180° → indices 1, 2, 3 NaN'd out.
        assert np.isnan(out_lons[1])
        assert np.isnan(out_lons[2])
        assert np.isnan(out_lons[3])


# --------------------------------------------------------------------------- #
# get_rotation_3D_to_plane_coords                                             #
# --------------------------------------------------------------------------- #

class TestGetRotation3DToPlaneCoords:
    def test_returns_orthogonal_matrix_with_det_one(self, refl):
        sat = np.array([R_LEO, 0.0, 0.0]).reshape(-1, 1)
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0]).reshape(-1, 1)
        R = refl.get_rotation_3D_to_plane_coords(sat, gnss)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_maps_orbital_plane_onto_z_zero(self, refl):
        # When sat and gnss lie in z=0 already, the rotation plane normal is +/-z;
        # the in-plane rotation should keep their z-component zero (or very near).
        # Use a non-trivial in-plane geometry so the rotation is non-identity.
        sat = np.array([R_LEO, 0.0, 0.0]).reshape(-1, 1)
        gnss = np.array([R_GNSS * np.cos(np.pi / 3), R_GNSS * np.sin(np.pi / 3), 0.0]).reshape(-1, 1)
        R = refl.get_rotation_3D_to_plane_coords(sat, gnss)
        rotated_sat = R @ sat
        rotated_gnss = R @ gnss
        # After rotation the z-component of in-plane vectors stays zero (the orbit
        # plane is the rotation plane in this setup).
        assert abs(rotated_sat[2, 0]) < 1e-6
        assert abs(rotated_gnss[2, 0]) < 1e-6

    def test_inclined_orbit_plane_rotated_to_xy(self, refl):
        # Inclined plane: sat in xz-plane, GNSS in xy-plane. Rotation should flatten
        # the cross-product (the orbit normal) onto +/-z.
        sat = np.array([R_LEO, 0.0, 200e3]).reshape(-1, 1)
        gnss = np.array([R_GNSS * np.cos(np.pi / 6), R_GNSS * np.sin(np.pi / 6), 0.0]).reshape(-1, 1)
        R = refl.get_rotation_3D_to_plane_coords(sat, gnss)
        rotated_sat = R @ sat
        rotated_gnss = R @ gnss
        # Both rotated vectors must now lie in z=0.
        assert abs(rotated_sat[2, 0]) < 1e-6
        assert abs(rotated_gnss[2, 0]) < 1e-6


# --------------------------------------------------------------------------- #
# get_reflection_point2d                                                      #
# --------------------------------------------------------------------------- #

class TestGetReflectionPoint2D:
    def test_returns_point_on_unit_circle_after_normalization(self):
        # The function normalizes coords by R_EARTH_EQUAT internally, then returns
        # the reflection point scaled back up by the same factor. The result must
        # therefore lie on the Earth's surface (||N|| == R_EARTH_EQUAT).
        sat = np.array([R_LEO, 0.0])
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4)])
        N, _, _ = Reflection.get_reflection_point2d(sat, gnss)
        assert N is not None
        assert np.linalg.norm(N) == pytest.approx(Constants.RADIUS_EARTH_EQUAT, rel=1e-6)

    def test_symmetric_geometry_reflection_on_axis_of_symmetry(self):
        # Sat and gnss mirrored across the x-axis at the same radius — the
        # specular point must sit on the symmetry axis (y = 0). Equal radii
        # are deliberate here: the symmetry argument only holds when the two
        # transmitters are interchangeable.
        r = R_LEO
        sat = np.array([r * np.cos(np.pi / 6), r * np.sin(np.pi / 6)])
        gnss = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6)])
        N, _, _ = Reflection.get_reflection_point2d(sat, gnss)
        assert N is not None
        assert N[1] == pytest.approx(0.0, abs=1.0)  # 1 m tolerance


# --------------------------------------------------------------------------- #
# get_reflection_point_ff (spherical solver, 3D)                              #
# --------------------------------------------------------------------------- #

class TestGetReflectionPointFf:
    def test_returns_nan_when_los_blocked(self, refl):
        # GNSS behind the Earth from LEO's perspective.
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([-R_GNSS, 0.0, 0.0])
        N, t2, t3 = refl.get_reflection_point_ff(gnss, leo)
        assert np.all(np.isnan(N))
        assert np.isnan(t2)
        assert np.isnan(t3)

    def test_reflection_lies_on_earth_surface(self, refl):
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        N, _, _ = refl.get_reflection_point_ff(gnss, leo)
        assert np.linalg.norm(N) == pytest.approx(Constants.RADIUS_EARTH_EQUAT, rel=1e-6)

    def test_reflection_in_orbital_plane(self, refl):
        # Both satellites in z=0 → reflection must be in z=0 too.
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        N, _, _ = refl.get_reflection_point_ff(gnss, leo)
        assert abs(N[2]) < 1.0  # tolerance: 1 m

    def test_law_of_reflection_angles_equal(self, refl):
        # The defining specular property: the angle Sat→N→zenith equals the
        # angle GNSS→N→zenith (both measured from the local normal).
        leo = np.array([R_LEO, 0.0, 0.0])
        gnss = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        _, t2, t3 = refl.get_reflection_point_ff(gnss, leo)
        assert t2 == pytest.approx(t3, abs=1e-3)

    def test_symmetric_geometry_reflection_on_symmetry_axis(self, refl):
        # Sat and gnss mirrored across the y=0 plane at equal radii — the
        # reflection lands on the symmetry plane (y = 0). Equal radii are
        # required for the mirror symmetry to make the two roles interchangeable.
        # Angular separation is kept small (20°) so the line-of-sight clears the
        # Earth — at LEO altitude, separations beyond ~44° pass through the planet.
        # The test exercises geometry, not realistic LEO/GNSS altitudes.
        r = R_LEO
        theta = np.deg2rad(10.0)
        leo = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        gnss = np.array([r * np.cos(theta), -r * np.sin(theta), 0.0])
        N, _, _ = refl.get_reflection_point_ff(gnss, leo)
        assert not np.any(np.isnan(N))
        assert N[1] == pytest.approx(0.0, abs=1.0)  # 1 m tolerance


# --------------------------------------------------------------------------- #
# _find_specular_point_ellipse_ecef (elliptical solver)                       #
# --------------------------------------------------------------------------- #

class TestFindSpecularPointEllipseEcef:
    def test_reflection_point_lies_on_ellipsoid(self, refl):
        a = Constants.RADIUS_EARTH_EQUAT
        b = a * (1 - Constants.EARTH_FLATTENING)
        tx_ecef = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        rx_ecef = np.array([R_LEO, 0.0, 0.0])
        S, _, _ = refl._find_specular_point_ellipse_ecef(tx_ecef, rx_ecef)
        # The ellipsoid constraint must be satisfied to high precision.
        ellipsoid_residual = S[0] ** 2 / a**2 + S[1] ** 2 / a**2 + S[2] ** 2 / b**2 - 1
        assert abs(ellipsoid_residual) < 1e-8

    def test_law_of_reflection_holds_in_ellipsoid_normal(self, refl):
        tx_ecef = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        rx_ecef = np.array([R_LEO, 0.0, 0.0])
        _, t2, t3 = refl._find_specular_point_ellipse_ecef(tx_ecef, rx_ecef)
        assert not np.isnan(t2)
        assert not np.isnan(t3)
        # The two angles (measured from the local ellipsoid normal) should match.
        assert t2 == pytest.approx(t3, abs=1e-3)


# --------------------------------------------------------------------------- #
# Spherical vs elliptical agreement                                           #
# --------------------------------------------------------------------------- #

class TestSphericalVsEllipticalAgreement:
    """The elliptical solver should agree with the spherical FF solution to within
    a few kilometres at most — Earth's flattening (~1/298) is small enough that
    the two models can never differ much for sub-orbital reflection geometry."""

    def test_reflection_points_agree_within_a_few_km(self, refl):
        # Use ECEF for both solvers (the FF method works the same in either ECI or
        # ECEF since it's geometry-only).
        tx = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        rx = np.array([R_LEO, 0.0, 0.0])
        N_sphere, _, _ = refl.get_reflection_point_ff(tx, rx)
        S_ellipse, _, _ = refl._find_specular_point_ellipse_ecef(tx, rx)
        assert not np.any(np.isnan(N_sphere))
        assert not np.any(np.isnan(S_ellipse))
        # Difference should be small — empirically <20 km for equatorial geometry,
        # and well under that for off-equator points where flattening matters less.
        delta_m = np.linalg.norm(N_sphere - S_ellipse)
        assert delta_m < 20_000.0  # 20 km tolerance

    def test_at_equator_reflection_z_is_zero_for_both(self, refl):
        # In-plane (z=0) geometry: both solvers must keep the reflection in z=0
        # (the orbital plane), within numerical tolerance.
        tx = np.array([R_GNSS * np.cos(np.pi / 4), R_GNSS * np.sin(np.pi / 4), 0.0])
        rx = np.array([R_LEO, 0.0, 0.0])
        N_sphere, _, _ = refl.get_reflection_point_ff(tx, rx)
        S_ellipse, _, _ = refl._find_specular_point_ellipse_ecef(tx, rx)
        assert abs(N_sphere[2]) < 1.0   # spherical: exact
        assert abs(S_ellipse[2]) < 1.0  # elliptical: numerically near-zero
