import numpy as np
import pymap3d

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.Satellite_def import Satellite

# WGS84 constants
a = Constants.RADIUS_EARTH_EQUAT
b = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)

class BRP:
    @staticmethod
    def _is_valid_earth_model(model):
        return model in ["spherical", "elliptical"]

    def __init__(self, gps_index, leo_index, cutoff_angle=45, earth_model="spherical", gnss_const=None,
                 max_steering_angle_deg=40, min_tx_elevation=5, min_bistatic_angle_deg=0,
                 max_bistatic_angle_deg=0.0):

        # Earth model check
        if not BRP._is_valid_earth_model(earth_model):
            raise ValueError(f"Invalid earth_model {earth_model!r}. Use either 'spherical' or 'elliptical'.")

        self.tx_satID = gps_index
        self.rx_satID = leo_index
        self.cutoff_angle = cutoff_angle
        self.earth_model = earth_model
        self.gnss_const = gnss_const
        self.max_steering_angle_deg = max_steering_angle_deg
        self.min_tx_elevation = min_tx_elevation
        self.min_bistatic_angle_deg = min_bistatic_angle_deg
        self.max_bistatic_angle_deg = max_bistatic_angle_deg

        # Initialize vectors exactly like Reflection
        self.time_vec = []
        self.refl_ECI_vec = []
        self.refl_LL_vec = None
        self.refl_ang_vec = []
        self.gnss_pos_vec = []
        self.gnss_vel_vec = []
        self.sat_pos_vec = []
        self.sat_vel_vec = []
        self.link_budget = {}

    def compute_brp_coordinates(
        self,
        GNSS_state: Satellite,
        Sat_state: Satellite,
        times,
        antenna: RxAntenna | None = None,
        frequency_hz: float | None = None,
        link_budget_settings: dict | lb.BistaticLinkBudgetSettings | None = None,
        q_sat: np.ndarray | None = None,
    ):
        if frequency_hz is None:
            frequency_hz = Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
        if q_sat is None:
            q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # --- Convert ECI → ECEF first
        # pymap3d expects a Python list/sequence of datetimes for batch conversions.
        times_list = list(times)
        T_ecef_all = np.column_stack(pymap3d.eci2ecef(
            GNSS_state.positions_eci[0], GNSS_state.positions_eci[1], GNSS_state.positions_eci[2], times_list,
        ))
        R_ecef_all = np.column_stack(pymap3d.eci2ecef(
            Sat_state.positions_eci[0], Sat_state.positions_eci[1], Sat_state.positions_eci[2], times_list,
        ))

        num_positions = len(times)

        brp_ECI_vec = np.full((num_positions, 3), np.nan)
        brp_LL_vec  = np.full((num_positions, 2), np.nan)
        incidence_angles = np.full(num_positions, np.nan)

        # --- Valid indices
        valid_mask = (np.isfinite(T_ecef_all).all(axis=1) &
                    np.isfinite(R_ecef_all).all(axis=1))
        valid_idx = np.where(valid_mask)[0]

        # --- Forward direction: finite differences of ECEF positions (along-track unit vector)
        R_fwd_all = np.empty_like(R_ecef_all)
        R_fwd_all[1:-1] = R_ecef_all[2:] - R_ecef_all[:-2]
        R_fwd_all[0]    = R_ecef_all[1]  - R_ecef_all[0]
        R_fwd_all[-1]   = R_ecef_all[-1] - R_ecef_all[-2]
        fwd_norms = np.linalg.norm(R_fwd_all, axis=1, keepdims=True)
        fwd_norms = np.where(fwd_norms < 1e-10, 1.0, fwd_norms)
        R_fwd_all = R_fwd_all / fwd_norms

        # --- Compute forward-looking BRP for each valid time step.
        # The Rx steers forward (along-track) to the farthest point where
        # Tx elevation >= min_tx_elevation AND bistatic angle <= max_bistatic_angle_deg.
        # This matches the corner-reflector geometry where the ship is far ahead
        # of both satellites (small bistatic angle).
        for k in valid_idx:
            T_ecef = T_ecef_all[k]
            R_ecef = R_ecef_all[k]
            R_fwd  = R_fwd_all[k]

            brp, tx_elev = find_brp_forward(
                R_ecef, R_fwd, T_ecef, a, b,
                self.min_tx_elevation, self.max_bistatic_angle_deg,
            )
            if brp is None:
                continue

            # Minimum bistatic angle filter (keeps backscatter distinct from specular)
            if self.min_bistatic_angle_deg > 0:
                to_tx = (T_ecef - brp) / np.linalg.norm(T_ecef - brp)
                to_rx = (R_ecef - brp) / np.linalg.norm(R_ecef - brp)
                bistatic = np.degrees(np.arccos(np.clip(np.dot(to_tx, to_rx), -1.0, 1.0)))
                if bistatic < self.min_bistatic_angle_deg:
                    continue

            incidence_angles[k] = 90.0 - tx_elev
            brp_ECI_vec[k] = np.array(pymap3d.ecef2eci(brp[0], brp[1], brp[2], times[k]), dtype=float)

        # --- Compute lat/lon only for valid BRPs
        valid_brp_idx = np.where(~np.isnan(brp_ECI_vec[:,0]))[0]

        if len(valid_brp_idx) > 0:
            lat, lon, _ = pymap3d.eci2geodetic(
                brp_ECI_vec[valid_brp_idx, 0],
                brp_ECI_vec[valid_brp_idx, 1],
                brp_ECI_vec[valid_brp_idx, 2],
                list(times[valid_brp_idx]),
            )
            brp_LL_vec[valid_brp_idx] = np.column_stack((lat, lon))

        link_budget = lb.compute_reflection_link_budget(
            tx_positions_eci=GNSS_state.positions_eci,
            rx_positions_eci=Sat_state.positions_eci,
            rx_velocities_eci=Sat_state.velocities_eci,
            reflection_positions_eci=brp_ECI_vec,
            incidence_deg=incidence_angles,
            antenna=antenna,
            frequency_hz=frequency_hz,
            settings=link_budget_settings,
            q_sat=q_sat,
        )
        power = link_budget["received_power_w"]

        self.time_vec = times

        # --- Store other fields
        self.refl_ECI_vec = brp_ECI_vec
        self.refl_LL_vec  = brp_LL_vec
        self.refl_ang_vec = incidence_angles
        self.gnss_pos_vec = GNSS_state.positions_eci
        self.gnss_vel_vec = GNSS_state.velocities_eci
        self.sat_pos_vec  = Sat_state.positions_eci
        self.sat_vel_vec  = Sat_state.velocities_eci
        self.link_budget = link_budget
        self.power = power

    def _compute_footprint(self, index: int, antenna: RxAntenna, frequency: float, q_sat=None, earth_model="spherical") -> tuple:
        """
        =========================================================================================================
        Thin wrapper around :func:`functions.functions.compute_footprint`.

        INPUTS:
            index:       [int]                 Sample index along the propagated time vector
            antenna:     [RxAntenna]           Receiver antenna
            frequency:   [float]               Carrier frequency [MHz]
            q_sat:       [np.ndarray | None]   Attitude quaternion [w, x, y, z] (default: identity)
            earth_model: [str]                 "spherical" or "elliptical"

        OUTPUT:
            result: [tuple]   (polygon_footprint, center_point) — Shapely Polygon and [lat, lon] center
        =========================================================================================================
        """
        return fn.compute_footprint(
            self.sat_pos_vec[:, index],
            self.sat_vel_vec[:, index],
            self.refl_ECI_vec[index, :],
            self.time_vec[index],
            antenna,
            frequency,
            q_sat=q_sat,
            earth_model=earth_model,
        )


# ===============================
#   GEOMETRY FUNCTIONS
# ===============================
def calculate_central_angle(R, T):
    R_norm = np.linalg.norm(R)
    T_norm = np.linalg.norm(T)
    if R_norm == 0 or T_norm == 0:
        return np.nan
    cos_gamma = np.clip(np.dot(R, T) / (R_norm * T_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_gamma))


def geocentric_nadir_on_ellipsoid(pos_ecef, a, b):
    """Scale an ECEF position vector onto the ellipsoid surface along the geocentric radial."""
    x, y, z = pos_ecef
    lam = 1.0 / np.sqrt((x / a) ** 2 + (y / a) ** 2 + (z / b) ** 2)
    return pos_ecef * lam


def off_nadir_angle(rx_ecef, point_ecef):
    """Angle [deg] between Rx nadir direction and the look vector from Rx to point_ecef."""
    nadir_dir = -rx_ecef / np.linalg.norm(rx_ecef)
    look_vec = point_ecef - rx_ecef
    look_dir = look_vec / np.linalg.norm(look_vec)
    cos_a = np.clip(np.dot(nadir_dir, look_dir), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def find_brp_in_tx_direction(rx_ecef, tx_ecef, a, b, max_steering_deg):
    """
    Find the Backward Reflection Point for quasi-monostatic geometry.

    Rx steers toward the Tx nadir point on the ellipsoid.  If that direction is
    within max_steering_deg of Rx nadir, the Tx nadir IS the BRP.  Otherwise the
    look direction is clamped to max_steering_deg off-nadir (in the plane containing
    nadir and the Tx direction) and intersected with the ellipsoid.

    Returns (brp_ecef, rx_look_angle_deg) or (None, None) if no valid intersection.
    """
    tx_nadir = geocentric_nadir_on_ellipsoid(tx_ecef, a, b)
    look_angle = off_nadir_angle(rx_ecef, tx_nadir)
    if look_angle <= max_steering_deg:
        return tx_nadir, look_angle

    max_rad = np.radians(max_steering_deg)
    nadir_dir = -rx_ecef / np.linalg.norm(rx_ecef)

    # Horizontal component of the Tx-nadir direction (in the nadir–Tx plane)
    to_tx_dir = (tx_nadir - rx_ecef)
    to_tx_dir = to_tx_dir / np.linalg.norm(to_tx_dir)
    horiz = to_tx_dir - np.dot(to_tx_dir, nadir_dir) * nadir_dir
    horiz_norm = np.linalg.norm(horiz)
    if horiz_norm < 1e-10:
        return None, None
    horiz = horiz / horiz_norm

    look_dir = np.cos(max_rad) * nadir_dir + np.sin(max_rad) * horiz

    # Intersect the ray rx_ecef + t*look_dir with the WGS-84 ellipsoid
    x0, y0, z0 = rx_ecef
    dx, dy, dz = look_dir
    A_c = (dx / a) ** 2 + (dy / a) ** 2 + (dz / b) ** 2
    B_c = 2.0 * (x0 * dx / a ** 2 + y0 * dy / a ** 2 + z0 * dz / b ** 2)
    C_c = (x0 / a) ** 2 + (y0 / a) ** 2 + (z0 / b) ** 2 - 1.0
    disc = B_c ** 2 - 4.0 * A_c * C_c
    if disc < 0:
        return None, None
    sqrt_d = np.sqrt(disc)
    t_vals = [t for t in [(-B_c + sqrt_d) / (2.0 * A_c), (-B_c - sqrt_d) / (2.0 * A_c)] if t > 0]
    if not t_vals:
        return None, None
    return rx_ecef + min(t_vals) * look_dir, max_steering_deg


def find_brp_forward(rx_ecef, rx_fwd_ecef, tx_ecef, a, b, min_tx_elevation_deg, max_bistatic_deg):
    """
    Forward-looking BRP for corner-reflector backscatter geometry.

    Steers in the along-track direction and scans from the geometric horizon inward,
    returning the farthest-forward point where:
      - Tx elevation at BRP >= min_tx_elevation_deg
      - bistatic angle at BRP <= max_bistatic_deg  (skipped when max_bistatic_deg <= 0)

    Scanning horizon→nadir guarantees the first valid hit has the smallest
    bistatic angle (β decreases as the BRP moves farther ahead of both satellites).

    Returns (brp_ecef, tx_elev_deg) or (None, None) if no valid point exists.
    """
    nadir_dir = -rx_ecef / np.linalg.norm(rx_ecef)

    # Horizontal along-track component
    horiz = rx_fwd_ecef - np.dot(rx_fwd_ecef, nadir_dir) * nadir_dir
    h_norm = np.linalg.norm(horiz)
    if h_norm < 1e-10:
        return None, None
    horiz /= h_norm

    # Off-nadir angle to the geometric horizon: arcsin(R_earth / |rx_ecef|)
    horizon_deg = np.degrees(np.arcsin(np.clip(a / np.linalg.norm(rx_ecef), 0.0, 1.0)))

    for theta_deg in np.arange(horizon_deg - 0.5, 0.0, -1.0):
        theta_rad = np.radians(theta_deg)
        look_dir = np.cos(theta_rad) * nadir_dir + np.sin(theta_rad) * horiz

        # Ray–ellipsoid intersection
        x0, y0, z0 = rx_ecef
        dx, dy, dz = look_dir
        Ac = (dx / a) ** 2 + (dy / a) ** 2 + (dz / b) ** 2
        Bc = 2.0 * (x0 * dx / a**2 + y0 * dy / a**2 + z0 * dz / b**2)
        Cc = (x0 / a) ** 2 + (y0 / a) ** 2 + (z0 / b) ** 2 - 1.0
        disc = Bc**2 - 4.0 * Ac * Cc
        if disc < 0:
            continue
        sqrt_d = np.sqrt(disc)
        t_vals = [t for t in [(-Bc + sqrt_d) / (2.0 * Ac), (-Bc - sqrt_d) / (2.0 * Ac)] if t > 0]
        if not t_vals:
            continue
        brp = rx_ecef + min(t_vals) * look_dir

        # Tx elevation at BRP
        ray_tx = tx_ecef - brp
        n_surf = np.array([brp[0] / a**2, brp[1] / a**2, brp[2] / b**2])
        n_surf /= np.linalg.norm(n_surf)
        tx_elev = np.degrees(np.arcsin(np.clip(
            np.dot(ray_tx / np.linalg.norm(ray_tx), n_surf), -1.0, 1.0
        )))
        if tx_elev < min_tx_elevation_deg:
            continue

        # Bistatic angle at BRP
        if max_bistatic_deg > 0:
            to_tx = ray_tx / np.linalg.norm(ray_tx)
            to_rx = (rx_ecef - brp) / np.linalg.norm(rx_ecef - brp)
            bistatic = np.degrees(np.arccos(np.clip(np.dot(to_tx, to_rx), -1.0, 1.0)))
            if bistatic > max_bistatic_deg:
                continue

        return brp, tx_elev

    return None, None


def calculate_ellipsoid_intersection_t(R, T, a=a, b=b, cutoff_angle=45):
    R = np.asarray(R, float)
    T = np.asarray(T, float)

    central_angle = calculate_central_angle(R, T)
    if central_angle >= cutoff_angle:
        return None, None, "opposite_side"

    d = T - R
    nd = np.linalg.norm(d)
    if nd == 0:
        return None, None, "degenerate"

    u = d / nd
    x0, y0, z0 = R
    ux, uy, uz = u

    A_p = b**2 * (ux**2 + uy**2) + a**2 * uz**2
    B_t = b**2 * (x0*ux + y0*uy) + a**2 * z0*uz
    C_p = b**2 * (x0**2 + y0**2 - a**2) + a**2 * z0**2

    disc = B_t**2 - A_p*C_p
    if disc < 0:
        return None, None, "outside_segment"

    sd = np.sqrt(disc)
    t1 = (-B_t + sd) / A_p
    t2 = (-B_t - sd) / A_p

    if t2 > 0:
        t = t2
    elif t1 > 0:
        t = t1
    else:
        return None, None, "outside_segment"

    P = R + t*u
    return P, t, "ok"
