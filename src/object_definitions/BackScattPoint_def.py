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

    def __init__(self, gps_index, leo_index, cutoff_angle=45, earth_model="spherical", gnss_const=None):

        # Earth model check
        if not BRP._is_valid_earth_model(earth_model):
            raise ValueError(f"Invalid earth_model {earth_model!r}. Use either 'spherical' or 'elliptical'.")

        self.tx_satID = gps_index
        self.rx_satID = leo_index
        self.cutoff_angle = cutoff_angle
        self.earth_model = earth_model
        self.gnss_const = gnss_const

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

        brp_ECEF_vec = np.full((num_positions, 3), np.nan)
        brp_ECI_vec = np.full((num_positions, 3), np.nan)
        brp_LL_vec  = np.full((num_positions, 2), np.nan)
        incidence_angles = np.full(num_positions, np.nan)

        # --- Valid indices
        valid_mask = (np.isfinite(T_ecef_all).all(axis=1) &
                    np.isfinite(R_ecef_all).all(axis=1))
        valid_idx = np.where(valid_mask)[0]

        # --- Compute BRP for each GLOBAL index (geometry only; defer ECI conversion to batch below)
        for k in valid_idx:
            T_ecef = T_ecef_all[k]
            R_ecef = R_ecef_all[k]

            brp, tval, reason = calculate_ellipsoid_intersection_t(T_ecef, R_ecef, a=a, b=b, cutoff_angle=self.cutoff_angle)
            if reason == "ok":
                ray_vec = R_ecef - brp
                # Incidence angle is measured from the local ellipsoid normal at the BRP
                # (the geocentric position vector differs from the geodetic normal by up to ~0.19°).
                n_surface = np.array([brp[0] / a**2, brp[1] / a**2, brp[2] / b**2])
                n_surface = n_surface / np.linalg.norm(n_surface)
                cos_theta = np.dot(ray_vec, n_surface) / np.linalg.norm(ray_vec)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_deg = np.degrees(np.arccos(cos_theta))

                if theta_deg < self.cutoff_angle:
                    brp_ECEF_vec[k] = brp
                    incidence_angles[k] = theta_deg

        # --- Batch ECEF → ECI and ECI → geodetic conversions for valid BRPs
        valid_brp_idx = np.where(~np.isnan(brp_ECEF_vec[:, 0]))[0]

        if len(valid_brp_idx) > 0:
            valid_times_list = list(times[valid_brp_idx])
            brp_eci_x, brp_eci_y, brp_eci_z = pymap3d.ecef2eci(
                brp_ECEF_vec[valid_brp_idx, 0],
                brp_ECEF_vec[valid_brp_idx, 1],
                brp_ECEF_vec[valid_brp_idx, 2],
                valid_times_list,
            )
            brp_ECI_vec[valid_brp_idx, 0] = brp_eci_x
            brp_ECI_vec[valid_brp_idx, 1] = brp_eci_y
            brp_ECI_vec[valid_brp_idx, 2] = brp_eci_z

            lat, lon, _ = pymap3d.eci2geodetic(
                brp_ECI_vec[valid_brp_idx, 0],
                brp_ECI_vec[valid_brp_idx, 1],
                brp_ECI_vec[valid_brp_idx, 2],
                valid_times_list,
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
