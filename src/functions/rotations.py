import numpy as np

from object_definitions.Constants_def import Constants


def quat_to_rotation_matrix(q: list) -> np.ndarray:
    """Convert a quaternion ``[w, x, y, z]`` to a 3x3 rotation matrix.

    Parameters
    ----------
    q : list
        Quaternion as ``[w, x, y, z]``.

    Returns
    -------
    numpy.ndarray
        Rotation matrix, shape ``(3, 3)``.
    """
    w, x, y, z = q
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def build_vnc_frame(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray | None:
    """Build the VNC (Velocity-Normal-Cross-track) frame from a satellite state in ECI.

    Returns the 3x3 rotation matrix ``R_eci_to_vnc`` such that ``v_vnc = R_eci_to_vnc @ v_eci``. The rows
    are the V, N, C unit vectors expressed in ECI.

    Parameters
    ----------
    r_eci : numpy.ndarray
        Satellite position in ECI [m], shape ``(3,)``.
    v_eci : numpy.ndarray
        Satellite velocity in ECI [m/s], shape ``(3,)``.

    Returns
    -------
    numpy.ndarray or None
        The rotation matrix, or ``None`` for a degenerate state (zero velocity / position, or position
        parallel to velocity so the orbit normal is undefined).
    """
    v_norm = np.linalg.norm(v_eci)
    if v_norm <= 0.0:
        return None
    r_norm = np.linalg.norm(r_eci)
    if r_norm <= 0.0:
        return None

    v_hat = v_eci / v_norm
    h_vec = np.cross(r_eci, v_hat)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= 0.0:
        return None
    n_hat = h_vec / h_norm

    c_hat = np.cross(v_hat, n_hat)
    c_norm = np.linalg.norm(c_hat)
    if c_norm <= 0.0:
        return None
    c_hat = c_hat / c_norm

    return np.vstack([v_hat, n_hat, c_hat])


def eci_to_antenna_rotation(
    R_eci_to_vnc: np.ndarray,
    q_sat: np.ndarray | None,
    R_body_to_vnc: np.ndarray,
    R_antenna_to_body: np.ndarray,
) -> np.ndarray:
    """Compose the full ECI-to-antenna rotation from the VNC frame and the attitude / mounting matrices.

    Chain: ECI -> VNC -> body -> antenna. Vector-coordinate-transform convention, so
    ``v_antenna = R_eci_to_antenna @ v_eci`` and the reverse rotation is the transpose.

    Parameters
    ----------
    R_eci_to_vnc : numpy.ndarray
        From :func:`build_vnc_frame`, shape ``(3, 3)``.
    q_sat : numpy.ndarray or None
        Attitude quaternion ``[w, x, y, z]`` (default identity).
    R_body_to_vnc : numpy.ndarray
        Body-to-VNC rotation (``Constants.R_body_to_vnc_normal``).
    R_antenna_to_body : numpy.ndarray
        Antenna-to-body mounting rotation (``RxAntenna.R_antenna_to_body``).

    Returns
    -------
    numpy.ndarray
        The full ECI-to-antenna rotation, shape ``(3, 3)``.
    """
    if q_sat is None:
        q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    R_attitude = quat_to_rotation_matrix(q_sat)
    R_vnc_to_body = R_attitude @ R_body_to_vnc.T
    return R_antenna_to_body.T @ R_vnc_to_body @ R_eci_to_vnc


def eci_to_antenna_rotation_from_state(
    sat_pos_eci: np.ndarray,
    sat_vel_eci: np.ndarray,
    antenna,
    q_sat: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build VNC from the satellite state and compose the full ECI→antenna rotation in one call. Returns None if the VNC frame is degenerate."""
    R_eci_to_vnc = build_vnc_frame(sat_pos_eci, sat_vel_eci)
    if R_eci_to_vnc is None:
        return None
    return eci_to_antenna_rotation(R_eci_to_vnc, q_sat, Constants.R_body_to_vnc_normal, antenna.R_antenna_to_body)
