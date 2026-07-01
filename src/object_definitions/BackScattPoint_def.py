import numpy as np
import pymap3d

import functions.functions as fn
import functions.link_budget as lb
from functions.rotations import eci_to_antenna_rotation_from_state
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.Satellite_def import Satellite

# WGS84 constants
a = Constants.RADIUS_EARTH_EQUAT
b = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)

class BRP:
    """Bistatic reflection point (BRP) for the backscatter mode, one (Tx GNSS, Rx LEO) pair.

    In backscatter mode the transmitter, the surface point, and the receiver are nearly in line (a
    retroreflection), so the reflection point is where the straight line through Tx and Rx meets the
    Earth's surface on the receiver side. This class solves that point at every time sample, applies the
    geometry gates, and (with a link-budget configuration) evaluates the received power. It is the
    backscatter-mode counterpart of :class:`~object_definitions.Reflection_def.Reflection`.

    Parameters
    ----------
    gps_index : Any
        Transmitting (GNSS) satellite identifier.
    leo_index : Any
        Receiving (LEO) satellite identifier.
    max_central_angle_deg : float, keyword-only
        Maximum Tx-Rx geocentric central angle [deg]; a coarse geometric gate.
    max_incidence_angle_deg : float, keyword-only
        Maximum incidence angle at the BRP [deg]; rejects too-grazing geometry.
    earth_model : {"spherical", "elliptical"}, optional
        Surface model (default "spherical").
    gnss_const : str, optional
        GNSS constellation label.

    Attributes
    ----------
    refl_ECI_vec : numpy.ndarray
        BRP positions in ECI [m], shape (n, 3); NaN where no valid point.
    refl_ECEF_vec : numpy.ndarray
        BRP positions in ECEF [m].
    refl_LL_vec : numpy.ndarray
        BRP positions as (latitude, longitude) [deg].
    refl_ang_vec : numpy.ndarray
        Incidence angle at the BRP [deg].
    link_budget : dict
        Link-budget output (empty in deferred mode).
    power : numpy.ndarray or None
        Received power [W] (None in deferred mode).
    """

    @staticmethod
    def _is_valid_earth_model(model):
        return model in ["spherical", "elliptical"]

    def __init__(self, gps_index, leo_index, *, max_central_angle_deg, max_incidence_angle_deg, earth_model="spherical", gnss_const=None):

        # Earth model check
        if not BRP._is_valid_earth_model(earth_model):
            raise ValueError(f"Invalid earth_model {earth_model!r}. Use either 'spherical' or 'elliptical'.")
        try:
            max_central_angle_deg = float(max_central_angle_deg)
            max_incidence_angle_deg = float(max_incidence_angle_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("Geometry gate angles must be numeric.") from exc
        if not np.isfinite(max_central_angle_deg) or not (0.0 < max_central_angle_deg <= 180.0):
            raise ValueError("max_central_angle_deg must be finite and in (0, 180] deg.")
        if not np.isfinite(max_incidence_angle_deg) or not (0.0 < max_incidence_angle_deg <= 90.0):
            raise ValueError("max_incidence_angle_deg must be finite and in (0, 90] deg.")

        self.tx_satID = gps_index
        self.rx_satID = leo_index
        self.max_central_angle_deg = float(max_central_angle_deg)
        self.max_incidence_angle_deg = float(max_incidence_angle_deg)
        self.earth_model = earth_model
        self.gnss_const = gnss_const

        # Initialize vectors exactly like Reflection
        self.time_vec = []
        self.refl_ECI_vec = []
        self.refl_LL_vec = None
        self.refl_ECEF_vec = None     # ECEF reflection points; set in compute_brp_coordinates
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
        tx_eirp_dbw: float,
        antenna: RxAntenna | None = None,
        frequency_hz: float | None = None,
        link_budget_settings: dict | lb.BistaticLinkBudgetSettings | None = None,
        q_sat: np.ndarray | None = None,
        compute_link_budget: bool = True,
    ):
        """Solve the BRP at every sample and (optionally) evaluate the link budget.

        Fills the reflection attributes in place. Samples failing the central-angle gate, the antenna
        field-of-view pre-gate, the ellipsoid intersection, or the incidence-angle gate are left as NaN.

        Parameters
        ----------
        GNSS_state : Satellite
            Propagated transmitting GNSS satellite (ECI positions / velocities).
        Sat_state : Satellite
            Propagated receiving LEO satellite (ECI positions / velocities).
        times : numpy.ndarray
            UTC timestamps aligned with the propagated states.
        tx_eirp_dbw : float
            Transmit EIRP for this (constellation, band) [dBW].
        antenna : RxAntenna or None, optional
            Receiver antenna (drives the field-of-view pre-gate; None => isotropic 0 dBi).
        frequency_hz : float or None, optional
            Carrier frequency [Hz]; defaults to GPS L1.
        link_budget_settings : dict or BistaticLinkBudgetSettings or None, optional
            Rx-system link-budget settings.
        q_sat : numpy.ndarray or None, optional
            Attitude quaternion [w, x, y, z] (default identity).
        compute_link_budget : bool, optional
            If False, skip the link budget (deferred per-event mode); default True.
        """
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
        if self.earth_model == "elliptical":
            surface_a = a
            surface_b = b
        else:
            surface_a = surface_b = Constants.R_EARTH_MEAN

        # --- Valid indices
        valid_mask = (np.isfinite(T_ecef_all).all(axis=1) &
                    np.isfinite(R_ecef_all).all(axis=1))
        valid_idx = np.where(valid_mask)[0]

        # Whether to gate on antenna FOV before computing the BRP. The retroreflector look
        # direction from the LEO is parallel to (LEO - Tx), so we can check the off-boresight
        # angle without solving for the BRP first — saves the quadratic for samples the antenna
        # could never receive from anyway.
        antenna_fov_gating = (
            antenna is not None
            and getattr(antenna, "max_across_track_steering_deg", None) is not None
            and getattr(antenna, "max_along_track_steering_deg", None) is not None
        )

        # --- Compute BRP for each GLOBAL index (geometry only; defer ECI conversion to batch below)
        for k in valid_idx:
            T_ecef = T_ecef_all[k]
            R_ecef = R_ecef_all[k]

            if antenna_fov_gating:
                R_eci_to_antenna = eci_to_antenna_rotation_from_state(
                    Sat_state.positions_eci[:, k],
                    Sat_state.velocities_eci[:, k],
                    antenna,
                    q_sat,
                )
                if R_eci_to_antenna is None:
                    continue
                look_eci = Sat_state.positions_eci[:, k] - GNSS_state.positions_eci[:, k]
                look_eci_norm = np.linalg.norm(look_eci)
                if look_eci_norm <= 0.0:
                    continue
                look_a = R_eci_to_antenna @ (look_eci / look_eci_norm)
                # Two-plane steering: along-track tilts along antenna x; across-track tilts along antenna y.
                # reject if the beam can't be steered to this direction (behind, or beyond per-axis limit).
                if look_a[2] <= 0.0:
                    continue
                across_track_scan_deg = np.degrees(np.arcsin(np.clip(look_a[1], -1.0, 1.0)))
                along_track_scan_deg = np.degrees(np.arcsin(np.clip(look_a[0], -1.0, 1.0)))
                if abs(across_track_scan_deg) > antenna.max_across_track_steering_deg + 1e-9 or abs(along_track_scan_deg) > antenna.max_along_track_steering_deg + 1e-9:
                    continue

            brp, tval, reason = calculate_ellipsoid_intersection_t(
                T_ecef,
                R_ecef,
                a=surface_a,
                b=surface_b,
                max_central_angle_deg=self.max_central_angle_deg,
            )
            if reason == "ok":
                ray_vec = R_ecef - brp
                # Incidence angle is measured from the selected surface model's local normal.
                n_surface = np.array([brp[0] / surface_a**2, brp[1] / surface_a**2, brp[2] / surface_b**2])
                n_surface = n_surface / np.linalg.norm(n_surface)
                cos_theta = np.dot(ray_vec, n_surface) / np.linalg.norm(ray_vec)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_deg = np.degrees(np.arccos(cos_theta))

                if theta_deg < self.max_incidence_angle_deg:
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

        if compute_link_budget:
            link_budget = lb.compute_reflection_link_budget(
                tx_positions_eci=GNSS_state.positions_eci,
                rx_positions_eci=Sat_state.positions_eci,
                rx_velocities_eci=Sat_state.velocities_eci,
                reflection_positions_eci=brp_ECI_vec,
                incidence_deg=incidence_angles,
                antenna=antenna,
                frequency_hz=frequency_hz,
                tx_eirp_dbw=tx_eirp_dbw,
                settings=link_budget_settings,
                q_sat=q_sat,
                surface_normals_eci=(
                    lb.ellipsoid_surface_normals_eci(brp_ECEF_vec, times)
                    if self.earth_model == "elliptical"
                    else None
                ),
                tx_velocities_eci=GNSS_state.velocities_eci,
            )
            power = link_budget["received_power_w"]
        else:
            # Deferred mode: the link budget is computed per detection event by the caller.
            link_budget = {}
            power = None

        self.time_vec = times

        # --- Store other fields
        self.refl_ECI_vec = brp_ECI_vec
        self.refl_LL_vec  = brp_LL_vec
        self.refl_ang_vec = incidence_angles
        self.gnss_pos_vec = GNSS_state.positions_eci
        self.gnss_vel_vec = GNSS_state.velocities_eci
        self.sat_pos_vec  = Sat_state.positions_eci
        self.sat_vel_vec  = Sat_state.velocities_eci
        self.refl_ECEF_vec = brp_ECEF_vec  # ECEF reflection points; for deferred per-event budget
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
def calculate_ellipsoid_intersection_t(tx_ecef, rx_ecef, *, max_central_angle_deg, a=a, b=b):
    """Intersect the line through Tx and Rx with the WGS-84 ellipsoid; return the near-side surface point past Rx.

    The line is parameterised as tx_ecef + t * u, with u pointing from Tx toward Rx. The accepted
    root must satisfy ``t > |Rx - Tx|``, placing the surface point beyond the receiver toward Earth.
    """
    tx_ecef = np.asarray(tx_ecef, float)
    rx_ecef = np.asarray(rx_ecef, float)

    central_angle = fn.angle_between_vectors(tx_ecef, rx_ecef)
    if central_angle >= max_central_angle_deg:
        return None, None, "opposite_side"

    d = rx_ecef - tx_ecef
    nd = np.linalg.norm(d)
    if nd == 0:
        return None, None, "degenerate"

    u = d / nd
    x0, y0, z0 = tx_ecef
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

    # The BRP is beyond the receiver along the Tx -> Rx ray, not merely any positive line root.
    if t2 > nd:
        t = t2
    elif t1 > nd:
        t = t1
    else:
        return None, None, "outside_segment"

    P = tx_ecef + t*u
    return P, t, "ok"
