import numpy as np
from astropy.time import Time
import pymap3d
from object_definitions.Constants_def import Constants
from object_definitions.Satellite_def import Satellite
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS, SkyCoord
import datetime
from shapely.geometry import Polygon

import functions.functions as fn
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

    def compute_brp_coordinates(self, GNSS_state: Satellite, Sat_state: Satellite, times):

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

        distance_R = np.linalg.norm((brp_ECI_vec - Sat_state.positions_eci.T) * 1e-3, axis=1)
        distance_T = np.linalg.norm((brp_ECI_vec - GNSS_state.positions_eci.T) * 1e-3, axis=1)
        power = 1 / (distance_R**2 * distance_T**2) * 1e15# --- Store times as a Python-datetime array (NO helper function!)

        self.time_vec = times#np.array([t.utc_datetime() for t in times], dtype=object)

        # --- Store other fields
        self.refl_ECI_vec = brp_ECI_vec
        self.refl_LL_vec  = brp_LL_vec
        self.refl_ang_vec = incidence_angles
        self.gnss_pos_vec = GNSS_state.positions_eci
        self.gnss_vel_vec = GNSS_state.velocities_eci
        self.sat_pos_vec  = Sat_state.positions_eci
        self.sat_vel_vec  = Sat_state.velocities_eci
        self.power = power

    def _compute_footprint(self, index: int, antenna: RxAntenna, frequency: float, q_sat=np.array([1.0, 0.0, 0.0, 0.0]), earth_model="spherical") -> tuple:
        """
        Computes the footprint of the antenna model and returns a Shapely polygon in lat/lon coordinates.
        INPUTS:
        - index:              Time index for position vectors
        - earth_model:        Earth model (spherical or elliptical)

        OUTPUTS:
        - footprint_polygon: Shapely polygon representing the elliptical footprint
        """
        # Get Satellite and Reflection positions
        sat_pos = self.sat_pos_vec[:, index]
        refl_pos = self.refl_ECI_vec[index, :]
        time_i = self.time_vec[index]

        # Check if the satellite position is valid
        if np.any(np.isnan(sat_pos)) or np.any(np.isnan(refl_pos)):
            print(f"Invalid satellite or reflection position at index {index}. Skipping footprint calculation.")
            return Polygon(), np.array([])

        l_sr = np.linalg.norm(refl_pos - sat_pos)  # Beam length (slant range)
        n_ar_eci = (refl_pos - sat_pos) / l_sr  # Satellite-to-reflection unit vector in ECI coordinates

        # ========== VNC COORDINATE SYSTEM {vnc} (Velocity Normal Cross-track) ==========
        # "Orbit-NED"  coordinate system decomposed in ECI coordinates
        v_s_i = self.sat_vel_vec[:, index]  # Satellite velocity vector in ECI coordinates
        r_s_i  = sat_pos # Satellite position vector in ECI coordinates
        # Check for valid velocity
        if np.linalg.norm(v_s_i) < 1e-6:
            print(f"Warning: Invalid satellite velocity at index {index}")
            return Polygon(), np.array([])
        if np.linalg.norm(r_s_i) < 1e-6:
            print(f"Warning: Invalid satellite position at index {index}")
            return Polygon(), np.array([])

        # 1. V-axis (satellite travel direction)
        v_hat = v_s_i / np.linalg.norm(v_s_i)  # Unit vector in the direction of satellite motion

        # 2. N-axis (Orbit Normal direction, perpendicular to the orbit plane)
        h_vec = np.cross(r_s_i, v_hat) # Angular momentum vector (cross product of position and velocity)
        if np.linalg.norm(h_vec) < 1e-12:
            print(f"Warning: Satellite position and velocity are collinear at index {index}")
            return np.array([])
        n_hat = h_vec / np.linalg.norm(h_vec) # Unit vector in the direction of the orbit normal

        # 3. C-axis (Cross-track direction, perpendicular to both V and N)
        c_hat = np.cross(v_hat, n_hat)
        c_hat = c_hat / np.linalg.norm(c_hat)  # Normalize the cross-track direction

        R_eci_to_vnc = np.vstack([
            v_hat,  # V: Velocity
            n_hat,  # N: Normal
            c_hat   # C: Cross-track
        ])

        # ========== BEAM / RAY COORDINATE SYSTEM {r} ==========
        # Create perpendicular axes for the beam cone (orthogonal to the beam direction)
        v_s_proj_i = v_s_i - np.dot(v_s_i, n_ar_eci) * n_ar_eci
        # Handle edge case where velocity (v_s_vnc) is parallel to the antenna beam (n_ar_eci) (This should never happen)
        if np.linalg.norm(v_s_proj_i) < 1e-6:
            # Create arbitrary perpendicular vector
            if abs(n_ar_eci[2]) < 0.9:
                temp_vec = np.array([0, 0, 1])  # Use z-axis
            else:
                temp_vec = np.array([1, 0, 0])  # Use x-axis
            v_s_proj_i = temp_vec - np.dot(temp_vec, n_ar_eci) * n_ar_eci

        # "Antenna beam coordinate axes" in the beam-perpendicular plane decomposed in ECI coordinates
        x_r_eci = v_s_proj_i / np.linalg.norm(v_s_proj_i)  # Unit vector in the direction of the beam projection
        y_r_eci = np.cross(n_ar_eci, x_r_eci)  # Cross-track component (perpendicular to the beam and x_antenna)
        y_r_eci = y_r_eci / np.linalg.norm(y_r_eci)   # Normalize the cross-track component
        z_r_eci = n_ar_eci
        # Rotation matrix from ECI to reflection-beam coordinate system (columns are reflection-beam axes in ECI)
        R_eci_to_reflection = np.vstack([
            x_r_eci,
            y_r_eci,
            z_r_eci
        ])

        # ========== CALCULATE BEAM/RAY ELEVATION (in flight direction = theta) AND AZIMUTH (accross flight dir) = phi)==========
        R_attitude = fn.quat_to_rotation_matrix(q_sat)  # Rotation matrix from body-fixed to orbital coordinate system (using quaternion)
        R_vnc_to_body = R_attitude @ Constants.R_body_to_vnc_normal.T  # Rotation matrix from VNC to body-fixed coordinate system
        # Beam direction in the antenna coordinate system
        n_ar_a = antenna.R_antenna_to_body.T @ R_vnc_to_body @ R_eci_to_vnc @ n_ar_eci  # Satellite-to-reflection unit vector in antenna coordinate system
        # Sanity check for n_ar_a (should point away from the antenna)
        if n_ar_a[2] < 0:
            print(f"Warning: Reflection behind antenna! {index}.")
            return Polygon(), np.array([])
        epsilon = np.arctan2(n_ar_a[0], n_ar_a[2])  # Elevation in flight direction [rad]
        psi = np.arctan2(n_ar_a[1], n_ar_a[2])    # Azimuth across flight direction [rad]

        # ========== GET THE ROTATION MATRIX FROM NADIR POINTING TO THE REFLECTION BEAM POINTING ==========
        beam_dir_a = antenna._get_antenna_footprint(n_ar_a, frequency, psi, epsilon)  # Get the antenna footprint in the antenna coordinate system

        # ========== VECTORIZED BEAM DIRECTIONS ==========
        beam_dirs_i = (R_eci_to_vnc.T @ R_vnc_to_body.T @ antenna.R_antenna_to_body @ beam_dir_a.T)  # Rotate beam directions to the orbital coordinate system
        no_footprint_points = beam_dirs_i.shape[1]  # Number of footprint points

        # ========== VECTORIZED INTERSECTION ==========
        # Use mean Earth radius for speed (or reflection point radius)
        if earth_model == "spherical":
            earth_radius = Constants.R_EARTH_MEAN
        else:
            # Use radius at reflection point for all (faster than per-point calculation)
            latitude_rad = np.arcsin(np.clip(refl_pos[2] / np.linalg.norm(refl_pos), -1, 1))
            a = Constants.RADIUS_EARTH_EQUAT
            f = Constants.EARTH_FLATTENING
            b = a * (1 - f)
            # Calculate geocentric radius using the formula:
            # R(φ) = √[(a²cos²φ + b²sin²φ)]
            earth_radius = np.sqrt((a**2 * np.cos(latitude_rad) ** 2) + (b**2 * np.sin(latitude_rad) ** 2))

        a = 1.0
        b = 2.0 * np.dot(sat_pos, beam_dirs_i)
        c = np.dot(sat_pos, sat_pos) - earth_radius**2

        discriminant = b**2 - 4 * a * c
        valid_discriminant = discriminant >= 0

        # Calculate intersection points
        t_vals = np.full(no_footprint_points, np.nan)
        t_vals[valid_discriminant] = (-b[valid_discriminant] - np.sqrt(discriminant[valid_discriminant])) / 2.0

        # Filter for positive intersection points
        positive_t_mask = t_vals > 0
        valid_mask = valid_discriminant & positive_t_mask

        # Calculate all intersection points at once
        intersections_eci = sat_pos[np.newaxis, :] + t_vals[:, np.newaxis] * beam_dirs_i.T

        # ========== VECTORIZED LAT/LON CONVERSION ==========
        times = np.array([time_i] * no_footprint_points)  # Use the same time for all points
        lat_lon_points = np.full((no_footprint_points, 2), np.nan)
        if valid_mask.size > 0:
            try:
                lat, lon, alt = pymap3d.eci2geodetic(
                    intersections_eci[valid_mask, 0], intersections_eci[valid_mask, 1], intersections_eci[valid_mask, 2], times[valid_mask]
                )
                lat_lon_points[valid_mask] = np.column_stack((lat, lon))
            except Exception as e:
                print(f"Warning: Failed to convert some points to geodetic coordinates: {e}")
                lat_lon_points[valid_mask] = np.nan

        # ========== OUTPUTS ==========
        # Return the footprint polygon in lat/lon coordinates
        ellipse_points = np.column_stack((lat_lon_points[:, 0], lat_lon_points[:, 1]))
        # Calculate the center point of the ellipse
        lat_center, lon_center, _ = pymap3d.eci2geodetic(refl_pos[0], refl_pos[1], refl_pos[2], time_i)
        center_point = np.array([lat_center, lon_center])

        footprint_points = ellipse_points[:, [1, 0]]
        # Create a shapely polygon from the points
        if np.isnan(footprint_points).any():
            # Footprint is partly tangenting earth
            # Remove the nan values
            # TODO this could be handled better!
            footprint_points = footprint_points[~np.isnan(footprint_points).any(axis=1)]

        polygon_footprint = Polygon(footprint_points)

        return polygon_footprint, center_point


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
