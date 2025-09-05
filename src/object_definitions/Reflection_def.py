import numpy as np
import pymap3d
from scipy.optimize import minimize
from shapely.geometry import Polygon

import functions.functions as fn
from object_definitions.Constants_def import Constants
from object_definitions.Satellite_def import Satellite
from object_definitions.Rx_antenna_def import RxAntenna

class Reflection:
    def __init__(
        self,
        tx_satID=None,
        rx_satID=None,
        earth_model=None,
        angle=None,
        gnss_const=None,
    ):
        """
        =========================================================================================================
        Reflection class:
            - Defines a reflection object: Initialized with the following parameters:
                INPUTS:
                - tx_satID:     Transmitting satellite ID (GNSS satellite IS)
                - rx_satID:     Receiving satellite ID    (LEO satellite ID)
                - earth model:                            (spherical or elliptical)
                - angle:                                  (Cutoff angle) (default: 45 degrees)
                - gnss_const:   GNSS constellation        (GPS, GLONASS, GALILEO, BEIDOU) (optional)

                ATTRIBUTES:
                - earth_model:  Earth model               (spherical or elliptical)
                - time_vec:     Time vector               (vector with UTC datetime objects, when reflection occurs)
                - refl_ECI_vec: Reflection points         (vector containing ECI coordinates of point reflections)
                - refl_LL_vec:  Reflection points         (vector containing Lat-Lon coordinates of point reflections)

            NOTE: One reflection object contains all the reflections for one given GNSS satellite and one given LEO
                  satellite in a certain time period.
        =========================================================================================================
        """
        # ENTRY point to the Object (Initialization)
        if earth_model is None:
            # If not specified, use the 'spherical' earth model
            print("Earth model not specified - Using default ('spherical')")
            earth_model = "spherical"
        else:
            if not Reflection._is_valid_earth_model(earth_model):
                # Raise a warning and return an empty object if a wrong earth model is chosen
                print("Wrong earth model specified. Use either 'spherical' or 'elliptical'.")
                return None
            else:
                # - Robustness - : If the earth model is correct, initialize the object
                pass

        if angle is None:
            # If not specified, use the default cutoff angle of 45 degrees
            print("Cutoff angle not specified - Using default (45 degrees)")
            angle = 45

        if tx_satID is None or rx_satID is None:
            Warning("Transmitting or Receiving satellite IDs are not specified    \n WARNING: Reflection object not initialized")
            return None
        self.initialization(earth_model, angle, tx_satID, rx_satID, gnss_const)

    def initialization(self, earth_model, angle, tx_satID, rx_satID, gnss_const):
        self.earth_model = earth_model  # 'spherical' or 'circular'
        self.gnss_const = gnss_const  # Could be [GPS, GLONASS, GALILEO, BEIDOU]
        self.cutoff_angle = angle
        self.tx_satID = tx_satID
        self.rx_satID = rx_satID
        self.time_vec = []
        self.refl_ECI_vec = []
        self.refl_LL_vec = None
        self.gnss_pos_vec = []
        self.gnss_vel_vec = []
        self.sat_pos_vec = []
        self.sat_vel_vec = []
        self.refl_ang_vec = []

    def compute_reflection_coordinates(self, GNSS_state: Satellite, Sat_state: Satellite, times: np.ndarray):
        if type(times) is not np.ndarray:
            times = np.array(times, dtype="datetime64[s]")  # Ensure times is a numpy array of datetime64 objects
        # Preallocate outputs for efficiency
        num_positions = Sat_state.positions_eci.shape[1]
        theta_vec = np.full((num_positions, 3), np.nan)
        specular_point_vec = np.full((num_positions, 3), np.nan)

        # Calculate the angles (theta) between GPS and satellite positions
        dot_product = np.einsum("ij,ij->j", GNSS_state.positions_eci, Sat_state.positions_eci)
        magnitude_a = np.linalg.norm(GNSS_state.positions_eci, axis=0)
        magnitude_b = np.linalg.norm(Sat_state.positions_eci, axis=0)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        theta = np.degrees(np.arccos(cos_theta))

        # Precompute line-of-sight (LOS) and valid theta indices for efficiency
        los_mask = np.array([self._check_los_satellites(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i]) for i in range(num_positions)])
        valid_indices = (theta < self.cutoff_angle) & los_mask

        # Loop through satellite positions with valid LOS to compute reflection points
        for i in np.where(valid_indices)[0]:
            if self.earth_model == "elliptical":
                specular_point_ecef, theta_degrees2, theta_degrees3 = self.find_specular_point_ellipse(
                    GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i]
                )
                theta_vec[i, :] = [theta_degrees2, theta_degrees3, theta[i]]
                specular_point_vec[i, :] = specular_point_ecef
            elif self.earth_model == "spherical":
                reflection_point, theta_2_circ, theta_3_circ = self.get_reflection_point_ff(GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i])
                theta_vec[i, :] = [theta_2_circ, theta_3_circ, theta[i]]
                specular_point_vec[i, :] = reflection_point

        # Compute geodetic coordinates for valid specular points
        valid_specular_points = ~np.isnan(specular_point_vec[:, 0])
        latlon_ref = np.full((num_positions, 2), np.nan)
        if valid_specular_points.any():
            lat, lon, _ = pymap3d.eci2geodetic(
                specular_point_vec[valid_specular_points, 0],
                specular_point_vec[valid_specular_points, 1],
                specular_point_vec[valid_specular_points, 2],
                times[valid_specular_points],
            )
            latlon_ref[valid_specular_points, :] = np.column_stack((lat, lon))
            lons, lats = self._prevent_longitude_wraparound(latlon_ref[:, 1], latlon_ref[:, 0])
        else:
            lons = np.nan
            lats = np.nan

        distance_R = np.linalg.norm((specular_point_vec - np.transpose(Sat_state.positions_eci)) * 1e-3, axis=1)
        distance_T = np.linalg.norm((specular_point_vec - np.transpose(GNSS_state.positions_eci)) * 1e-3, axis=1)

        power = 1 / (distance_R**2 * distance_T**2) * 1e15

        # Store results in the object
        self.time_vec = times  # Time vector numpy array
        self.refl_ECI_vec = (
            specular_point_vec  # Reflection points         ECI coordinates      # TODO: Change from vector n x 3  to 3 x n (same as sat_pos_vec)
        )
        self.refl_LL_vec = np.column_stack(
            (lats, lons)
        )  # Reflection points         Lat-Lon coordinates  # TODO: Change from vector n x 2 to 2 x n (same as sat_pos_vec)
        self.gnss_pos_vec = GNSS_state.positions_eci  # GNSS satellite positions  ECI coordinates
        self.gnss_vel_vec = GNSS_state.velocities_eci  # GNSS satellite velocities ECI coordinates
        self.sat_pos_vec = Sat_state.positions_eci  # LEO satellite positions   ECI coordinates
        self.sat_vel_vec = Sat_state.velocities_eci  # LEO satellite velocities  ECI coordinates
        self.refl_ang_vec = (
            theta_vec  # Reflection angles         ECI coordinates [deg] # TODO: Change from vector n x 3 to 3 x n (actually 2 x n) (same as sat_pos_vec)
        )
        self.power = power  # Power of the reflection   []

    def get_reflection_point_ff(self, gnssPos, satPos):
        """
        This function calculates the reflection point between a satellite and a GNSS satellite position
        (Same method as in FreeFlyer) for an spherical earth.
        """
        # Convert positions to column vectors
        satCoord = np.array(satPos).reshape(-1, 1)
        GPSCoord = np.array(gnssPos).reshape(-1, 1)

        # Check for line of sight (LOS)
        d = satCoord - GPSCoord
        t_values = np.linspace(0, 1, 200)  # Sample points along the line

        # Verify LOS by checking if any point along the line passes through Earth
        for t in t_values:
            r_t = GPSCoord + t * d
            distance_to_earth_center = np.linalg.norm(r_t)
            if distance_to_earth_center < Constants.RADIUS_EARTH_EQUAT:
                # No line of sight, obstructed by Earth
                return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # Get rotation matrix
        Rot = self.get_rotation_ecef_to_plane_coords(satCoord, GPSCoord)

        # Rotate satellite and GPS coordinates
        satCoord1 = Rot @ satCoord
        GPSCoord1 = Rot @ GPSCoord

        # Find 2D reflection point
        N, A, B = self.get_reflection_point2d(satCoord1.flatten(), GPSCoord1.flatten())

        # Transform reflection point to 3D (z = 0)
        N = np.array([N[0], N[1], 0])

        # Convert reflection point back to ECEF coordinates
        ReflectionPoint = Rot.T @ N

        # Calculate angles between vectors
        vec_1 = ReflectionPoint - np.array([0, 0, 0])
        vec_2 = satPos - ReflectionPoint
        vec_3 = gnssPos - ReflectionPoint

        theta_degrees2 = self._angle_between_vectors(vec_1, vec_2)
        theta_degrees3 = self._angle_between_vectors(vec_1, vec_3)

        return ReflectionPoint, theta_degrees2, theta_degrees3

    def find_specular_point_ellipse(self, Tx_ecef, Rx_ecef):
        # Check if transmitting and receiving satellites have a clear line of sight (if so there is no LOS => no reflection)
        if not self._check_los_satellites(Rx_ecef, Tx_ecef):
            # No line of sight, return NaN values
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # Radius at the poles for elliptical Earth model
        a = Constants.RADIUS_EARTH_EQUAT
        b = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)

        # Ellipsoid constraint
        def ellipsoid_constraint(S):
            return (S[0] ** 2 / Constants.RADIUS_EARTH_EQUAT**2 + S[1] ** 2 / Constants.RADIUS_EARTH_EQUAT**2 + S[2] ** 2 / (b) ** 2) - 1

        # Reflection constraint
        def reflection_error(S):
            """
            For specular reflection: angle of incidence = angle of reflection
            Both angles measured from the surface normal.
            """
            S = np.array(S)
            tx_to_S = S - Tx_ecef  # Vector from the transmitting satellite to the specular point
            S_to_rx = Rx_ecef - S  # Vector from the specular point to the receiving satellite

            # Normalize the vectors
            tx_to_S = tx_to_S / np.linalg.norm(tx_to_S)
            S_to_rx = S_to_rx / np.linalg.norm(S_to_rx)

            # compute ellipsoid normal at S (outward)
            Nx = 2 * S[0] / (a**2)
            Ny = 2 * S[1] / (a**2)
            Nz = 2 * S[2] / (b**2)
            n_surface = np.array([Nx, Ny, Nz])
            n_surface = n_surface / np.linalg.norm(n_surface)  # Normalize the normal vector

            # Specular reflection: angle of incidence = angle of reflection
            # => cos(θ_incident) = cos(θ_reflected)
            cos_incident = -np.dot(tx_to_S, n_surface)  # Angle between incoming ray and normal
            cos_reflected = np.dot(S_to_rx, n_surface)  # Angle between outgoing ray and normal

            # The reflection condition: both rays should make equal angles with normal
            return cos_incident - cos_reflected

        def objective_function(S):
            try:
                return reflection_error(S) ** 2
            except (ZeroDivisionError, np.linalg.LinAlgError):
                # Avoid this point if the reflection error calculation fails
                return 1e10

        # Initial guess for the specular point from "get_reflection_point_FF"
        try:
            spec_init_guess_ecef, _, _ = self.get_reflection_point_ff(Tx_ecef, Rx_ecef)

            if np.any(np.isnan(spec_init_guess_ecef)):
                # If the initial guess is NaN, return NaN values
                spec_init_guess_ecef = (Tx_ecef + Rx_ecef) / 2  # Use midpoint as fallback
        except Exception:
            # Use fallback method if the initial guess fails
            spec_init_guess_ecef = (Tx_ecef + Rx_ecef) / 2  # Use midpoint as fallback

        # Ensure the initial guess is on the ellipsoid
        scale = 1.0 / np.sqrt((spec_init_guess_ecef[0] / a) ** 2 + (spec_init_guess_ecef[1] / a) ** 2 + (spec_init_guess_ecef[2] / b) ** 2)
        spec_init_guess_ecef = scale * spec_init_guess_ecef

        # Run optimization to find the specular point
        result = minimize(
            objective_function,
            spec_init_guess_ecef,
            constraints={"type": "eq", "fun": ellipsoid_constraint},
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 200, "disp": False},
        )

        if not (result.success and result.fun < 1e-3):
            # if it fails to meet tight tolerance, warn or bail
            print(f"Ellipse optimization did not converge tightly: success={result.success}, residual={result.fun:.2e}")
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # Extract final specular point S
        S = result.x

        # coerce exactly onto ellipsoid (avoid tiny drift)
        scale = 1.0 / np.sqrt((S[0] / a) ** 2 + (S[1] / a) ** 2 + (S[2] / b) ** 2)
        S = scale * S

        # Recompute the incident/reflected angles with respect to the true normal
        v_in = S - Tx_ecef
        v_in /= np.linalg.norm(v_in)
        v_out = Rx_ecef - S
        v_out /= np.linalg.norm(v_out)

        Nx = 2 * S[0] / (a**2)
        Ny = 2 * S[1] / (a**2)
        Nz = 2 * S[2] / (b**2)
        n_surface = np.array([Nx, Ny, Nz])
        n_surface /= np.linalg.norm(n_surface)

        cos_inc = np.dot(-v_in, n_surface)
        cos_refl = np.dot(v_out, n_surface)
        theta_degrees2 = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
        theta_degrees3 = np.degrees(np.arccos(np.clip(cos_refl, -1, 1)))

        return S, theta_degrees2, theta_degrees3

    def get_rotation_ecef_to_plane_coords(self, satCoord, GPSCoord):
        # Calculate the rotation matrix to transform from ECEF to plane coordinates
        plane = np.cross(satCoord.flatten(), GPSCoord.flatten())
        plane = plane / np.linalg.norm(plane)
        z = np.array([0, 0, 1])

        vec = np.cross(plane, z)
        s = np.linalg.norm(vec)
        c = np.dot(plane, z)

        # Handle the case where the vectors are parallel or anti-parallel (plane i== z)
        # Minimal fix: prevent division by zero but keep same approach
        if s == 0:
            vec2 = np.array([1, 0, 0])  # In case of the plane being parallel to z, use a default vector
        else:
            vec2 = vec / s  # Normalize the vector

        theta = np.arccos(np.dot(plane, z))
        c2 = np.cos(theta)
        s2 = np.sin(theta)

        Sv = self._skew_matrix(vec2)
        Rot = np.eye(3) + s2 * Sv + Sv @ Sv * (1 - c2)
        return Rot

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

    @staticmethod
    def get_reflection_point2d(satCoord1, GPSCoord1):
        """
        This function calculates the reflection point in 2D coordinates.
        INPUTS:
            satCoord1:  Satellite coordinates in 2D (x, y) [m]
            GPSCoord1:  GPS coordinates in 2D (x, y) [m]
        OUTPUTS:
            N:          Reflection point in 2D (x, y) [m]
        """
        # Calculates the reflection point in 2D
        A = satCoord1[:2] / Constants.RADIUS_EARTH_EQUAT
        B = GPSCoord1[:2] / Constants.RADIUS_EARTH_EQUAT

        u = np.dot(A, A)
        v = np.dot(A, B)
        w = np.dot(B, B)

        r0 = w - 1
        r1 = 2 * (w - v)
        r2 = u + 2 * v + w - 4 * u * w
        r3 = 4 * (v**2 - u * w)
        r4 = 4 * u * (u * w - v**2)

        coefvct = [r4, r3, r2, r1, r0]  # Coefficients for polynomial
        roots = np.roots(coefvct)  # Find roots of the polynomial
        roots = roots.real  # Assuming real roots for simplicity

        for x in roots:
            if x > 0:
                y = (1 + x - 2 * u * x**2) / (1 + 2 * v * x)
                if y > 0:
                    N = x * A + y * B
                    return Constants.RADIUS_EARTH_EQUAT * N, A, B
        return None, None, None

    @staticmethod
    def _angle_between_vectors(v1, v2):
        # Calculate the angle between two vectors in degrees
        dot_product = np.dot(v1, v2)
        magnitude_a = np.linalg.norm(v1)
        magnitude_b = np.linalg.norm(v2)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        theta_radians = np.arccos(cos_theta)
        return np.degrees(theta_radians)

    @staticmethod
    def _check_los_satellites(r1, r2) -> bool:
        """
        Checks if there is a clear Line Of Sight (LOS) between two points r1 and r2,
        given the Earth's radius in kilometers (spherical earth).
        """
        d = r2 - r1

        t_values = np.linspace(0, 1, 100)
        points_along_line = r1[np.newaxis, :] + t_values[:, np.newaxis] * d[np.newaxis, :]
        distances_to_center = np.linalg.norm(points_along_line, axis=1)

        return not np.any(distances_to_center < Constants.RADIUS_EARTH_EQUAT)

    @staticmethod
    def _is_valid_earth_model(earth_model) -> bool:
        # Check if earth_model is either 'spherical' or 'elliptical'
        valid_models = ["spherical", "elliptical"]
        return earth_model in valid_models

    @staticmethod
    def _skew_matrix(v):
        # Returns the skew symmetric matrix of the vector v
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def _prevent_longitude_wraparound(lons, lats):
        # Find where the difference between consecutive longitudes exceeds 180 degrees -> avoid plotting lines from -179 to 179 deg longitude
        wraparound = np.abs(np.diff(lons)) > 180
        # Replace the longitude and latitude at the wraparound point with NaN
        lons[1:][wraparound] = np.nan
        lats[1:][wraparound] = np.nan
        return lons, lats
