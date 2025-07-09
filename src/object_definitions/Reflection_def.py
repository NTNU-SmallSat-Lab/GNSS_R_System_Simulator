from datetime import datetime, timezone, UTC

import numpy as np
import pymap3d
from scipy.optimize import minimize

import functions.functions as fn
from object_definitions.Constants_def import Constants


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

    def compute_reflection_coordinates(self, GNSS_state, Sat_state, times):
        times = np.array(times)

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
        los_mask = np.array(
            [self._check_los_satellites(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i]) for i in range(num_positions)]
        )
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
                reflection_point, theta_2_circ, theta_3_circ = self.get_reflection_point_ff(
                    GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i]
                )
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
            lons, lats = self.__prevent_longitude_wraparound(latlon_ref[:, 1], latlon_ref[:, 0])
        else:
            lons = np.nan
            lats = np.nan

        distance_R = np.linalg.norm((specular_point_vec - np.transpose(Sat_state.positions_eci)) * 1e-3, axis=1)
        distance_T = np.linalg.norm((specular_point_vec - np.transpose(GNSS_state.positions_eci)) * 1e-3, axis=1)

        power = 1 / (distance_R**2 * distance_T**2) * 1e15

        # Store results in the object
        self.time_vec = times  # Time vector numpy array
        self.refl_ECI_vec = specular_point_vec  # Reflection points         ECI coordinates      # TODO: Change from vector n x 3  to 3 x n (same as sat_pos_vec)
        self.refl_LL_vec = np.column_stack(
            (lats, lons)
        )  # Reflection points         Lat-Lon coordinates  # TODO: Change from vector n x 2 to 2 x n (same as sat_pos_vec)
        self.gnss_pos_vec = GNSS_state.positions_eci  # GNSS satellite positions  ECI coordinates
        self.gnss_vel_vec = GNSS_state.velocities_eci  # GNSS satellite velocities ECI coordinates
        self.sat_pos_vec = Sat_state.positions_eci  # LEO satellite positions   ECI coordinates
        self.sat_vel_vec = Sat_state.velocities_eci  # LEO satellite velocities  ECI coordinates
        self.refl_ang_vec = theta_vec  # Reflection angles         ECI coordinates [deg] # TODO: Change from vector n x 3 to 3 x n (actually 2 x n) (same as sat_pos_vec)
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

    def _compute_footprint_ellipse(self, index, no_footprint_points, beamwidth_deg, q_sat=np.array([1.0, 0.0, 0.0, 0.0]), method='spherical', wavelength=None) -> np.ndarray:
        """
        Computes the elliptical footprint and returns a Shapely polygon in lat/lon coordinates.
        INPUTS:
          - index:              Time index for position vectors
          - no_footprint_points: Number of points to generate for the ellipse
          - beamwidth_deg:      Antenna beamwidth [flight_dir, swath_dir] in degrees
          - q_sat:             Satellite attitude quaternion
          - wavelength:        Wavelength of the signal (in meters) [optional]

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
            return np.array([])

        n_sr_eci = (refl_pos - sat_pos) / np.linalg.norm(refl_pos - sat_pos)  # Beam direction
        l_sr = np.linalg.norm(refl_pos - sat_pos)  # Beam length (slant range)

        # ========== ORBITAL COORDINATE SYSTEM ==========
        # NED coordinate system decomposed in ECI coordinates)
        v_s_e = self.sat_vel_vec[:, index]  # Satellite velocity vector in ECI coordinates
        # Check for valid velocity
        if np.linalg.norm(v_s_e) < 1e-6:
            print(f"Warning: Invalid satellite velocity at index {index}")
            return np.array([])

        # Angular momentum vector (orbit normal)
        h_vec = np.cross(v_s_e, sat_pos)  # h = v × r
        h_magnitude = np.linalg.norm(h_vec)

        # 1. Along-track (X): Direction of motion
        x_orbit_in_eci = v_s_e / np.linalg.norm(v_s_e)
        # 2. Cross-track (Y): Orbit normal direction
        y_orbit_in_eci = h_vec / h_magnitude  # Perpendicular to orbital plane
        # 3. Radial (Z): Downward direction (circular orbit = towards center of Earth)
        z_orbit_in_eci = np.cross(x_orbit_in_eci, y_orbit_in_eci)

        # Ensure orthonormality(Gram-Schmidt)
        y_orbit_in_eci = y_orbit_in_eci / np.linalg.norm(y_orbit_in_eci)
        z_orbit_in_eci = z_orbit_in_eci / np.linalg.norm(z_orbit_in_eci)
        x_orbit_in_eci = np.cross(y_orbit_in_eci, z_orbit_in_eci)
        x_orbit_in_eci = x_orbit_in_eci / np.linalg.norm(x_orbit_in_eci)

        # ========== SATELLITE ATTITUDE ==========
        R_body_to_orbit = fn.quat_to_rotation_matrix(q_sat)
        R_orbit_to_eci = np.column_stack([x_orbit_in_eci, y_orbit_in_eci, z_orbit_in_eci])
        R_body_to_eci = R_orbit_to_eci @ R_body_to_orbit # Might be needed on a later stage

        # ========== ANTENNA-BEAM COORDINATE SYSTEM ==========
        # Create perpendicular axes for the beam cone (orthogonal to the beam direction)
        sat_vel_projected = v_s_e - np.dot(v_s_e, n_sr_eci) * n_sr_eci

        # Handle edge case where velocity (v_s_e) is parallel to the antenna beam (n_sr_eci) (This should never happen)
        if np.linalg.norm(sat_vel_projected) < 1e-6:
            # Create arbitrary perpendicular vector
            if abs(n_sr_eci[2]) < 0.9:
                temp_vec = np.array([0, 0, 1])  # Use z-axis
            else:
                temp_vec = np.array([1, 0, 0])  # Use x-axis
            sat_vel_projected = temp_vec - np.dot(temp_vec, n_sr_eci) * n_sr_eci

        # "Ellipse coordinate axes" in the beam-perpendicular plane decomposed in ECI coordinates
        u_axis = sat_vel_projected / np.linalg.norm(sat_vel_projected)  # Perpendicular to beam, (satellite motion component)
        v_axis = np.cross(n_sr_eci, u_axis)  # Perpendicular to beam and u_axis (cross-track component)
        v_axis = v_axis / np.linalg.norm(v_axis)
        w_axis = n_sr_eci  # Along the beam direction

            # Choose exact method
        if method == 'spherical':
            beam_directions = self._generate_beam_directions_spherical(n_sr_eci, u_axis, v_axis, beamwidth_deg, no_footprint_points)
        elif method == 'rodrigues':
            beam_directions = self._generate_beam_directions_rodrigues(n_sr_eci, u_axis, v_axis, beamwidth_deg, no_footprint_points)
        elif method == 'matrix':
            beam_directions = self._generate_beam_directions_matrix(n_sr_eci, u_axis, v_axis, beamwidth_deg, no_footprint_points)
        else:
            raise ValueError("Method must be 'spherical', 'rodrigues', or 'matrix'")

        # Generate beam directions within the cone
        ellipse_points = []

        for i, beam_dir_i in enumerate(beam_directions):
        # Find intersection with Earth
            a = 1.0
            b = 2.0 * np.dot(sat_pos, beam_dir_i)
            c = np.dot(sat_pos, sat_pos) - Constants.R_EARTH_MEAN**2

            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / 2.0 # First intersection (entry point)
                t2 = (-b + np.sqrt(discriminant)) / 2.0 # Second intersection (exit point)

                # Chose the first positive intersection point (entry point)
                valid_t_values = [t for t in [t1, t2] if t > 0.0]

                if valid_t_values:
                    t_chosen = min(valid_t_values)  # Choose the first intersection
                    intersection_eci = sat_pos + t_chosen * beam_dir_i

                    # Verify intersection is on Earth's surface (within reasonable tolerance)
                    distance_from_center = np.linalg.norm(intersection_eci)
                    if abs(distance_from_center - Constants.R_EARTH_MEAN) < 5000:  # Within 5km tolerance
                        # Convert to lat/lon
                        try:
                            lat, lon, alt = pymap3d.eci2geodetic(
                                intersection_eci[0], intersection_eci[1], intersection_eci[2], time_i
                            )
                            #lat, lon, alt = self.eci_to_geodetic(intersection_eci, time_i)
                            # Additional check: altitude should be near zero
                            if abs(alt) < 10000:  # Within 10km of surface
                                ellipse_points.append((lat, lon))
                            else:
                                print(f"Warning: Point {i} has altitude {alt:.1f}m, skipping")
                        except Exception as e:
                            print(f"Warning: Failed to convert point {i} to geodetic: {e}")
                            continue
                    else:
                        print(f"Warning: Point {i} not on Earth surface (distance: {distance_from_center/1000:.1f}km)")
                else:
                    print(f"Warning: No valid positive intersection for point {i}")
            else:
                print(f"Warning: No intersection with Earth for point {i}")
            lat_s, lon_s, _ = pymap3d.eci2geodetic(refl_pos[0], refl_pos[1], refl_pos[2], time_i)
            center_point = (lat_s, lon_s)
        return ellipse_points, center_point

    def _generate_beam_directions_spherical(self, n_sr_eci, u_axis, v_axis, beamwidth_deg, no_points):
        """
        Generate exact beam directions using spherical coordinates.
        [Best accuracy]
        """
        beam_directions = []
        beamwidth_rad = [bw * np.pi / 180 for bw in beamwidth_deg]

        for i in range(no_points):
            # Parametric ellipse parameter
            t = 2 * np.pi * i / no_points

            # Angular positions on the ellipse (in the beam's local coordinate system)
            alpha_u = (beamwidth_rad[0]/2) * np.cos(t)  # Angle in u direction  
            alpha_v = (beamwidth_rad[1]/2) * np.sin(t)  # Angle in v direction

            # Convert to spherical coordinates
            theta_total = np.sqrt(alpha_u**2 + alpha_v**2)  # Total angle from beam center
            if theta_total > 0:
                phi = np.arctan2(alpha_v, alpha_u)  # Azimuthal angle around beam center

                # Exact spherical coordinate transformation
                beam_dir_i = (np.cos(theta_total) * n_sr_eci + 
                             np.sin(theta_total) * (np.cos(phi) * u_axis + np.sin(phi) * v_axis))
            else:
                # Center of beam
                beam_dir_i = n_sr_eci.copy()

            beam_directions.append(beam_dir_i)

        return beam_directions

    def _generate_beam_directions_rodrigues(self, n_sr_eci, u_axis, v_axis, beamwidth_deg, no_points):
        """
        Generate exact beam directions using Rodrigues' rotation formula.
        This method allows for precise control over rotations.
        """
        def rodrigues_rotation(v, k, theta):
            """
            Rotate vector v around axis k by angle theta using Rodrigues' formula.
            """
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            return (v * cos_theta + 
                    np.cross(k, v) * sin_theta + 
                    k * np.dot(k, v) * (1 - cos_theta))

        beam_directions = []
        beamwidth_rad = [bw * np.pi / 180 for bw in beamwidth_deg]

        for i in range(no_points):
            # Parametric ellipse parameter
            t = 2 * np.pi * i / no_points

            # Angular offsets
            theta_u = (beamwidth_rad[0]/2) * np.cos(t)
            theta_v = (beamwidth_rad[1]/2) * np.sin(t)

            # Start with central beam direction
            beam_dir_i = n_sr_eci.copy()

            # Apply rotations sequentially
            if abs(theta_u) > 1e-10:  # Avoid numerical issues for zero rotation
                beam_dir_i = rodrigues_rotation(beam_dir_i, v_axis, theta_u)
            if abs(theta_v) > 1e-10:
                beam_dir_i = rodrigues_rotation(beam_dir_i, u_axis, theta_v)

            beam_directions.append(beam_dir_i)

        return beam_directions

    def _generate_beam_directions_matrix(self, n_sr_eci, u_axis, v_axis, beamwidth_deg, no_points):
        """
        Generate exact beam directions using rotation matrices.
        This is often the most computationally efficient for many points.
        """
        def rotation_matrix_axis_angle(axis, angle):
            """Create rotation matrix for rotation around axis by angle (Rodrigues' formula)"""
            axis = axis / np.linalg.norm(axis)  # Ensure unit vector
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Skew-symmetric matrix
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]], 
                          [-axis[1], axis[0], 0]])

            # Rodrigues' rotation matrix formula
            R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
            return R

        beam_directions = []
        beamwidth_rad = [bw * np.pi / 180 for bw in beamwidth_deg]

        for i in range(no_points):
            # Parametric ellipse parameter
            t = 2 * np.pi * i / no_points

            # Angular offsets
            theta_u = (beamwidth_rad[0]/2) * np.cos(t)
            theta_v = (beamwidth_rad[1]/2) * np.sin(t)

            # Create rotation matrices
            R_u = rotation_matrix_axis_angle(v_axis, theta_u) if abs(theta_u) > 1e-10 else np.eye(3)
            R_v = rotation_matrix_axis_angle(u_axis, theta_v) if abs(theta_v) > 1e-10 else np.eye(3)

            # Apply combined rotation
            R_combined = R_v @ R_u  # Note: order matters!
            beam_dir_i = R_combined @ n_sr_eci

            beam_directions.append(beam_dir_i)

        return beam_directions

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
    def __prevent_longitude_wraparound(lons, lats):
        # Find where the difference between consecutive longitudes exceeds 180 degrees -> avoid plotting lines from -179 to 179 deg longitude
        wraparound = np.abs(np.diff(lons)) > 180
        # Replace the longitude and latitude at the wraparound point with NaN
        lons[1:][wraparound] = np.nan
        lats[1:][wraparound] = np.nan
        return lons, lats

    def eci_to_geodetic(self, eci_pos, time_utc):
        """The ONE function you need"""
        # Ensure time is UTC
        if hasattr(time_utc, 'replace') and time_utc.tzinfo is None:
            time_utc = time_utc.replace(tzinfo=UTC)

        # Calculate Earth rotation
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
        dt_days = (time_utc - j2000).total_seconds() / 86400.0
        gmst_hours = (18.697374558 + 24.06570982441908 * dt_days) % 24.0
        gmst_radians = gmst_hours * np.pi / 12.0

        # Rotate from ECI to ECEF
        cos_gmst = np.cos(gmst_radians)
        sin_gmst = np.sin(gmst_radians)
        R = np.array([[ cos_gmst,  sin_gmst, 0],
                      [-sin_gmst,  cos_gmst, 0],
                      [        0,         0, 1]])
        ecef_pos = R @ eci_pos

        # Convert to lat/lon/alt
        x, y, z = ecef_pos
        r = np.sqrt(x*x + y*y + z*z)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        alt = r - Constants.R_EARTH_MEAN

        return lat, lon, alt
