"""Unit tests for functions/rotations.py (quat_to_rotation_matrix, build_vnc_frame,
eci_to_antenna_rotation). Pure-math tests — no filesystem or simulator state required."""

import numpy as np
import pytest

from functions.rotations import (
    build_vnc_frame,
    eci_to_antenna_rotation,
    eci_to_antenna_rotation_from_state,
    quat_to_rotation_matrix,
)


class _FakeAntenna:
    """Duck-typed antenna providing only the attribute the rotation helper reads."""

    def __init__(self, R_antenna_to_body):
        self.R_antenna_to_body = R_antenna_to_body


# --------------------------------------------------------------------------- #
# quat_to_rotation_matrix                                                     #
# --------------------------------------------------------------------------- #

class TestQuatToRotationMatrix:
    def test_identity_quaternion_returns_identity(self):
        R = quat_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_180deg_rotation_about_x(self):
        R = quat_to_rotation_matrix([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-12)

    def test_180deg_rotation_about_y(self):
        R = quat_to_rotation_matrix([0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(R, np.diag([-1.0, 1.0, -1.0]), atol=1e-12)

    def test_180deg_rotation_about_z(self):
        R = quat_to_rotation_matrix([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(R, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)

    def test_90deg_rotation_about_z_rotates_xhat_to_yhat(self):
        # 90° about z: q = [cos(45°), 0, 0, sin(45°)]
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        R = quat_to_rotation_matrix([c, 0.0, 0.0, s])
        x_hat = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(R @ x_hat, [0.0, 1.0, 0.0], atol=1e-12)

    def test_non_unit_quaternion_is_normalised_internally(self):
        # Scaling the quaternion should not change the rotation
        R_unit = quat_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
        R_scaled = quat_to_rotation_matrix([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R_unit, R_scaled, atol=1e-12)

    def test_result_is_orthogonal_with_det_plus_one(self):
        R = quat_to_rotation_matrix([0.7, 0.1, 0.5, 0.2])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# build_vnc_frame                                                             #
# --------------------------------------------------------------------------- #

class TestBuildVncFrame:
    def test_equatorial_circular_orbit_canonical_axes(self):
        # r along +x, v along +y → V_hat = +y, N_hat = +z, C_hat = +x
        r = np.array([7e6, 0.0, 0.0])
        v = np.array([0.0, 7e3, 0.0])
        R = build_vnc_frame(r, v)
        assert R is not None
        np.testing.assert_allclose(R[0], [0.0, 1.0, 0.0], atol=1e-12)  # V
        np.testing.assert_allclose(R[1], [0.0, 0.0, 1.0], atol=1e-12)  # N
        np.testing.assert_allclose(R[2], [1.0, 0.0, 0.0], atol=1e-12)  # C

    def test_transforms_velocity_to_pure_v_axis(self):
        # The velocity in VNC should be (|v|, 0, 0) by construction
        r = np.array([7e6, 1e5, 3e5])
        v = np.array([1e2, 7e3, -1e2])
        R = build_vnc_frame(r, v)
        v_vnc = R @ v
        np.testing.assert_allclose(v_vnc, [np.linalg.norm(v), 0.0, 0.0], atol=1e-9)

    def test_zero_velocity_returns_none(self):
        assert build_vnc_frame(np.array([7e6, 0.0, 0.0]), np.zeros(3)) is None

    def test_zero_position_returns_none(self):
        assert build_vnc_frame(np.zeros(3), np.array([0.0, 7e3, 0.0])) is None

    def test_radial_velocity_returns_none(self):
        # v parallel to r → cross(r, v) = 0 → orbit normal undefined
        r = np.array([7e6, 0.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])
        assert build_vnc_frame(r, v) is None

    def test_output_is_orthonormal_for_generic_state(self):
        r = np.array([7e6, 1e6, 5e5])
        v = np.array([1e3, 7e3, -2e3])
        R = build_vnc_frame(r, v)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# eci_to_antenna_rotation                                                     #
# --------------------------------------------------------------------------- #

class TestEciToAntennaRotation:
    def test_all_identity_returns_identity(self):
        R = eci_to_antenna_rotation(np.eye(3), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3), np.eye(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_q_none_defaults_to_identity(self):
        R_none = eci_to_antenna_rotation(np.eye(3), None, np.eye(3), np.eye(3))
        R_identity = eci_to_antenna_rotation(np.eye(3), np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3), np.eye(3))
        np.testing.assert_allclose(R_none, R_identity, atol=1e-12)

    def test_chain_composition_matches_manual(self):
        # Three distinct 90° rotations about z so we can verify the chain order by hand.
        # R_eci_to_vnc, R_body_to_vnc, R_antenna_to_body all rotate +x→+y.
        cz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        R = eci_to_antenna_rotation(cz, q_identity, cz, cz)
        # Expected: R_antenna_to_body.T @ R_body_to_vnc.T @ R_eci_to_vnc
        expected = cz.T @ cz.T @ cz
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_attitude_appears_correctly_in_chain(self):
        # Verify that the attitude quaternion enters as R_attitude in R_vnc_to_body = R_attitude @ R_body_to_vnc.T
        # 90° rotation about z as a quaternion
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        q = np.array([c, 0.0, 0.0, s])
        R_attitude = quat_to_rotation_matrix(q)
        R = eci_to_antenna_rotation(np.eye(3), q, np.eye(3), np.eye(3))
        # With identity for R_eci_to_vnc, R_body_to_vnc, R_antenna_to_body:
        # R_vnc_to_body = R_attitude @ I.T = R_attitude
        # R = I.T @ R_attitude @ I = R_attitude
        np.testing.assert_allclose(R, R_attitude, atol=1e-12)

    def test_inverse_via_transpose_round_trips(self):
        # Round-trip: applying R then R.T should recover the input vector (rotation matrices are orthogonal)
        cz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        R = eci_to_antenna_rotation(cz, None, cz, cz)
        v = np.array([0.3, -0.7, 0.65])
        np.testing.assert_allclose(R.T @ (R @ v), v, atol=1e-12)


# --------------------------------------------------------------------------- #
# eci_to_antenna_rotation_from_state                                          #
# --------------------------------------------------------------------------- #

class TestEciToAntennaRotationFromState:
    def test_matches_manual_two_step_composition(self):
        # The helper must produce exactly the same matrix as build_vnc_frame + eci_to_antenna_rotation.
        from object_definitions.Constants_def import Constants

        r = np.array([7e6, 1e5, 3e5])
        v = np.array([1e2, 7e3, -1e2])
        R_antenna_to_body = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        antenna = _FakeAntenna(R_antenna_to_body)
        q = np.array([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])

        R_helper = eci_to_antenna_rotation_from_state(r, v, antenna, q)
        R_eci_to_vnc = build_vnc_frame(r, v)
        R_manual = eci_to_antenna_rotation(R_eci_to_vnc, q, Constants.R_body_to_vnc_normal, R_antenna_to_body)
        np.testing.assert_allclose(R_helper, R_manual, atol=1e-12)

    def test_degenerate_state_returns_none(self):
        # Zero velocity → build_vnc_frame returns None → helper must propagate None.
        antenna = _FakeAntenna(np.eye(3))
        assert eci_to_antenna_rotation_from_state(np.array([7e6, 0.0, 0.0]), np.zeros(3), antenna) is None
