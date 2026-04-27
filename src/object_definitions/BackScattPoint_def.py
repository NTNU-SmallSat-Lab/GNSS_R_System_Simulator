import numpy as np
from astropy.time import Time
import pymap3d
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord
import datetime

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.Constants_def import Constants
from object_definitions.Satellite_def import Satellite
from object_definitions.Rx_antenna_def import RxAntenna

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
            print("Wrong earth model specified. Use either 'spherical' or 'elliptical'.")
            return None

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
        self.power_proxy = []
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
        settings = lb.coerce_link_budget_settings(link_budget_settings)
        if frequency_hz is None:
            frequency_hz = Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
        if q_sat is None:
            q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # --- Convert ECI → ECEF first
        T_ecef_all = eci_gcrs_to_ecef(GNSS_state.positions_eci.T, times)
        R_ecef_all = eci_gcrs_to_ecef(Sat_state.positions_eci.T, times)

        num_positions = len(times)

        brp_ECI_vec = np.full((num_positions, 3), np.nan)
        brp_LL_vec  = np.full((num_positions, 2), np.nan)
        incidence_angles = np.full(num_positions, np.nan)

        # --- Valid indices
        valid_mask = (np.isfinite(T_ecef_all).all(axis=1) &
                    np.isfinite(R_ecef_all).all(axis=1))
        valid_idx = np.where(valid_mask)[0]

        # --- Compute BRP for each GLOBAL index
        for k in valid_idx:
            T_ecef = T_ecef_all[k]
            R_ecef = R_ecef_all[k]

            brp, tval, reason = calculate_ellipsoid_intersection_t(T_ecef, R_ecef, a=a, b=b, cutoff_angle=self.cutoff_angle)
            if reason == "ok":
                ray_vec = R_ecef - brp
                cos_theta = np.dot(ray_vec, brp) / (np.linalg.norm(ray_vec) * np.linalg.norm(brp))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_deg = np.degrees(np.arccos(cos_theta))

                if theta_deg < self.cutoff_angle:
                    brp_ECI_vec[k] = ecef_to_eci(brp, times[k])
                    incidence_angles[k] = theta_deg

        # --- Compute lat/lon only for valid BRPs
        valid_brp_idx = np.where(~np.isnan(brp_ECI_vec[:,0]))[0]

        if len(valid_brp_idx) > 0:
            lat_list, lon_list = [], []

            for k in valid_brp_idx:
                x, y, z = brp_ECI_vec[k]

                # Convert a *single* Skyfield scalar time → Astropy Time
                #t_astropy = Time(times[k].tt, format='jd', scale='tt')
                t_astropy = times[k]

                lat, lon, _ = pymap3d.eci2geodetic(x, y, z, t_astropy)
                lat_list.append(lat)
                lon_list.append(lon)

            brp_LL_vec[valid_brp_idx] = np.column_stack((lat_list, lon_list))

        distance_R_km = np.linalg.norm((brp_ECI_vec - Sat_state.positions_eci.T) * 1e-3, axis=1)
        distance_T_km = np.linalg.norm((brp_ECI_vec - GNSS_state.positions_eci.T) * 1e-3, axis=1)
        power_proxy = 1 / (distance_R_km**2 * distance_T_km**2) * 1e15

        distance_R_m = distance_R_km * 1e3
        distance_T_m = distance_T_km * 1e3

        # No antenna pattern → assume isotropic (0 dBi). Otherwise NaN gains
        # (e.g. reflection outside FOV) propagate to NaN power.
        if antenna is None:
            rx_gain_dbi = np.zeros(num_positions, dtype=float)
        else:
            rx_gain_dbi = np.full(num_positions, np.nan, dtype=float)
            freq_mhz = float(frequency_hz) / 1e6
            for i in range(num_positions):
                if not np.isfinite(distance_R_m[i]) or not np.isfinite(distance_T_m[i]):
                    continue
                if np.isnan(brp_ECI_vec[i]).any():
                    continue
                look_dir_a = lb.compute_rx_look_direction_antenna_frame(
                    Sat_state.positions_eci[:, i],
                    Sat_state.velocities_eci[:, i],
                    brp_ECI_vec[i, :],
                    antenna,
                    q_sat=q_sat,
                )
                rx_gain_dbi[i] = antenna.get_gain_dbi_for_direction(look_dir_a, freq_mhz)

        link_budget = lb.compute_bistatic_link_budget(
            range_tx_m=distance_T_m,
            range_rx_m=distance_R_m,
            incidence_deg=incidence_angles,
            rx_gain_dbi=rx_gain_dbi,
            frequency_hz=frequency_hz,
            settings=settings,
        )
        power = link_budget["received_power_w"]

        self.time_vec = times#np.array([t.utc_datetime() for t in times], dtype=object)

        # --- Store other fields
        self.refl_ECI_vec = brp_ECI_vec
        self.refl_LL_vec  = brp_LL_vec
        self.refl_ang_vec = incidence_angles
        self.gnss_pos_vec = GNSS_state.positions_eci
        self.gnss_vel_vec = GNSS_state.velocities_eci
        self.sat_pos_vec  = Sat_state.positions_eci
        self.sat_vel_vec  = Sat_state.velocities_eci
        self.power_proxy = power_proxy
        self.link_budget = link_budget
        self.power = power

    def _compute_footprint(self, index: int, antenna: RxAntenna, frequency: float, q_sat=np.array([1.0, 0.0, 0.0, 0.0]), earth_model="spherical") -> tuple:
        """Thin wrapper around :func:`functions.functions.compute_footprint`."""
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
#   ECI <---> ECEF CONVERSION
# ===============================
def eci_gcrs_to_ecef(r_eci, epoch):
    import numpy as np
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord

    r = np.atleast_2d(r_eci).astype(float)
    single = r.shape[0] == 1
    t = epoch
    # Convert Skyfield Time to Astropy Time
    #try:
    #    t = Time([e.utc_datetime() for e in epoch], scale='utc')  # multiple times
    #except TypeError:
    #    t = Time(epoch.utc_datetime(), scale='utc')  # single time

    # Broadcast single position to multiple times if needed
    if r.shape[0] == 1 and len(np.atleast_1d(t)) > 1:
        r = np.tile(r, (len(t), 1))

    cart = CartesianRepresentation(r[:,0]*u.m, r[:,1]*u.m, r[:,2]*u.m)
    sc = SkyCoord(cart, frame=GCRS(obstime=t))
    sc_itrs = sc.transform_to(ITRS(obstime=t))

    out = np.vstack([
        sc_itrs.cartesian.x.to(u.m).value,
        sc_itrs.cartesian.y.to(u.m).value,
        sc_itrs.cartesian.z.to(u.m).value
    ]).T

    return out[0] if single else out

def ecef_to_eci(r_ecef, epoch):
    import numpy as np
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord

    # Ensure Nx3 array
    r = np.atleast_2d(r_ecef).astype(float)
    single = r.shape[0] == 1

    t = epoch
    # Convert Skyfield Time to Astropy Time
    #try:
    #    t = Time([e.utc_datetime() for e in epoch], scale='utc')  # multiple times
    #except TypeError:
    #    t = Time(epoch.utc_datetime(), scale='utc')  # single time

    # Broadcast single position to multiple times if needed
    if r.shape[0] == 1 and len(np.atleast_1d(t)) > 1:
        r = np.tile(r, (len(t), 1))

    # Convert to CartesianRepresentation and wrap in ITRS frame
    cart = CartesianRepresentation(r[:,0]*u.m, r[:,1]*u.m, r[:,2]*u.m)
    sc = SkyCoord(cart, frame=ITRS(obstime=t))
    sc_gcrs = sc.transform_to(GCRS(obstime=t))

    # Extract x, y, z
    out = np.vstack([
        sc_gcrs.cartesian.x.to(u.m).value,
        sc_gcrs.cartesian.y.to(u.m).value,
        sc_gcrs.cartesian.z.to(u.m).value
    ]).T

    return out[0] if single else out

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

    t = None
    if t2 > 0: t = t2
    elif t1 > 0: t = t1
    else: 
        return None, None, "outside_segment"

    P = R + t*u
    return P, t, "ok"
