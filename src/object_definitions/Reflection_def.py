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
                - refl_ECI_vec: Reflection points         (vector containing ECEF coordinates of point reflections) # TODO ECI or ECEF???
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
            vec2 = vec / s   # Normalize the vector

        theta = np.arccos(np.dot(plane, z))
        c2 = np.cos(theta)
        s2 = np.sin(theta)

        Sv = self._skew_matrix(vec2)
        Rot = np.eye(3) + s2 * Sv + Sv @ Sv * (1 - c2)
        return Rot

    def _compute_footprint_ellipse(self, index, no_footprint_points, beamwidth_deg=[10.0, 5.0], q_sat=np.array([1.0, 0.0, 0.0, 0.0]), wavelength=None):
        """
        Computes the elliptical footprint size of the reflection based on the gain and wavelength.
        INPUTS:
        - gain:           Gain of the antenna (in dB)
        - wavelength:     Wavelength of the signal (in meters)
        OUTPUTS:
        - footprint_size: Size of the footprint (in meters) on earths surface
        """
        # Get Satellite and Reflection positions
        sat_pos = self.sat_pos_vec[:, index]
        refl_pos = self.refl_ECI_vec[index, :]
        # Calculations related to the vector/beam from satellite to reflection point
        r_sr_eci = refl_pos - sat_pos # Vector from satellite to reflection point in ECI coordinates
        n_sr_eci = r_sr_eci / np.linalg.norm(r_sr_eci) # Unit vector in the direction "satellite to reflection point"
        l_sr = np.linalg.norm(r_sr_eci) # Length of the vector from satellite to reflection point

        # Attitude and coordinate system of the satellite-orbit
        v_s_e = self.sat_vel_vec[:, index] # Satellite velocity vector in ECI coordinates
        x_orbit_in_eci = v_s_e / np.linalg.norm(v_s_e) # X-axis of the "satellite-orbit" coordinate system, decomposed in ECI coordinates
        z_orbit_in_eci = -sat_pos / np.linalg.norm(sat_pos) # Z-axis of the "satellite-orbit" coordinate system, pointing towards the center of the Earth, decomposed in ECI coordinates
        y_orbit_in_eci = np.cross(z_orbit_in_eci, x_orbit_in_eci)
        y_orbit_in_eci = y_orbit_in_eci / np.linalg.norm(y_orbit_in_eci) # Y-axis of the "satellite-orbit" coordinate system, orthogonal to X and Z axes, decomposed in ECI coordinates
        # Ensure orthogonality (Gram-Schmidt process)
        x_orbit_in_eci = np.cross(y_orbit_in_eci, z_orbit_in_eci)
        x_orbit_in_eci = x_orbit_in_eci / np.linalg.norm(x_orbit_in_eci)

        R_sat = fn.quat_to_rotation_matrix(q_sat)
        R_orbit_to_eci = np.column_stack([x_orbit_in_eci, y_orbit_in_eci, z_orbit_in_eci])
#        R_body_to_eci = R_orbit_to_eci @ R_sat

#        n_antenna_body = np.array([0, 0, 1])
#        n_antenna_eci = R_body_to_eci @ n_antenna_body

        # ========== RAY-SPHERE INTERSECTION ==========
        # Create coordinate system for beam cone
        # Create perpendicular axes for the beam cone (orthogonal to the beam direction)
        sat_vel_projected = v_s_e - np.dot(v_s_e, n_sr_eci) * n_sr_eci

        # Handle edge case where velocity is parallel to the antenna beam (this should never happen)
        if np.linalg.norm(sat_vel_projected) < 1e-6:
            # Create arbitrary perpendicular vector
            if abs(n_sr_eci[2]) < 0.9:
                temp_vec = np.array([0, 0, 1])  # Use z-axis
            else:
                temp_vec = np.array([1, 0, 0])  # Use x-axis
            sat_vel_projected = temp_vec - np.dot(temp_vec, n_sr_eci) * n_sr_eci

        # Ellipse coordinate axes in the beam-perpendicular plane
        ellipse_x_axis = sat_vel_projected / np.linalg.norm(sat_vel_projected)  # Perpendicular to beam, (satellite motion component)
        ellipse_y_axis = np.cross(n_sr_eci, ellipse_x_axis)                     # Perpendicular to beam and ellipse_x_axis (cross-track component)
        ellipse_y_axis = ellipse_y_axis / np.linalg.norm(ellipse_y_axis)
        ellipse_z_axis = n_sr_eci                                               # Along the beam direction

        # ========== CALCULATE ELLIPSE DIMENSIONS ==========
        # Semi-axes of ellipse in the beam-perpendicular plane
        semi_axis_x = l_sr * np.tan(np.radians(beamwidth_deg[0]) / 2.0)  # Flight direction
        semi_axis_y = l_sr * np.tan(np.radians(beamwidth_deg[1]) / 2.0)  # Cross-track

        # ========== GENERATE ELLIPSE POINTS ==========
        theta = np.linspace(0, 2*np.pi, no_footprint_points)
        footprint_points = np.zeros((no_footprint_points, 3))

        # Generate ellips points in the beam-perpendicular plane
        for i in range(no_footprint_points):
            # Parametric ellipse in the beam-perpendicular plane
            ellipse_point = (semi_axis_x * np.cos(theta[i]) * ellipse_x_axis +
                            semi_axis_y * np.sin(theta[i]) * ellipse_y_axis)

            # Position in 3D space (centered at reflection point)
            point_3d = refl_pos + ellipse_point

            # ========== PROJECT TO EARTH'S SURFACE ==========

            # Method 1: Ray-sphere intersection (most accurate)
            # Ray from satellite through this ellipse point to Earth
            ray_direction = (point_3d - sat_pos) / np.linalg.norm(point_3d - sat_pos)

            # Solve: |sat_pos + t * ray_direction|² = R_earth²
            a = 1.0  # |ray_direction|² = 1
            b = 2.0 * np.dot(sat_pos, ray_direction)
            c = np.dot(sat_pos, sat_pos) - Constants.R_EARTH_MEAN**2

            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / 2.0
                t2 = (-b + np.sqrt(discriminant)) / 2.0

                # Choose the intersection closer to reflection point
                intersection1 = sat_pos + t1 * ray_direction
                intersection2 = sat_pos + t2 * ray_direction

                if np.linalg.norm(intersection1 - refl_pos) < np.linalg.norm(intersection2 - refl_pos):
                    footprint_points[i, :] = intersection1
                else:
                    footprint_points[i, :] = intersection2
            else:
                # Fallback: radial projection (shouldn't happen for reasonable geometry)
                distance_from_center = np.linalg.norm(point_3d)
                footprint_points[i, :] = point_3d * (Constants.R_EARTH_MEAN / distance_from_center)

        # DEBUG: Plot the footprint points
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        print("=== GNSS-R Footprint Debug Info ===")
        print(f"Satellite position: {sat_pos}")
        print(f"Reflection position: {refl_pos}")
        print(f"Satellite velocity (v_s_e): {v_s_e}")
        print(f"Beam direction (n_sr_eci): {n_sr_eci}")
        print(f"Slant range (l_sr): {l_sr:.1f} m")
        print(f"Satellite velocity projected: {sat_vel_projected}")
        print(f"Projection magnitude: {np.linalg.norm(sat_vel_projected):.1f} m/s")
        print(f"Ellipse X-axis: {ellipse_x_axis}")
        print(f"Ellipse Y-axis: {ellipse_y_axis}")
        print(f"Semi-axis X: {semi_axis_x:.1f} m")
        print(f"Semi-axis Y: {semi_axis_y:.1f} m")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Overall 3D geometry
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')

        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = Constants.R_EARTH_MEAN * np.outer(np.cos(u), np.sin(v))
        y_earth = Constants.R_EARTH_MEAN * np.outer(np.sin(u), np.sin(v))
        z_earth = Constants.R_EARTH_MEAN * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='lightblue')

        # Key points
        ax1.scatter(*sat_pos, color='red', s=100, label='Satellite')
        ax1.scatter(*refl_pos, color='green', s=100, label='Reflection Point')

        # Vectors from satellite
        ax1.quiver(*sat_pos, *(n_sr_eci * l_sr * 0.3), 
                  color='purple', linewidth=3, label='Beam Direction (n_sr_eci)')
        ax1.quiver(*sat_pos, *(v_s_e * 0.0001), 
                  color='blue', linewidth=3, label='Satellite Velocity (v_s_e)')
        ax1.quiver(*sat_pos, *(sat_vel_projected * 0.0001), 
                  color='cyan', linewidth=3, label='Velocity Projected')

        # Ellipse axes from reflection point
        ax1.quiver(*refl_pos, *(ellipse_x_axis * semi_axis_x), 
                  color='orange', linewidth=2, label='Ellipse X-axis')
        ax1.quiver(*refl_pos, *(ellipse_y_axis * semi_axis_y), 
                  color='red', linewidth=2, label='Ellipse Y-axis')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('Overall GNSS-R Geometry')

        # Plot 2: Zoom on reflection area
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')

        # Local Earth surface around reflection
        refl_area_size = max(semi_axis_x, semi_axis_y) * 3
        n_points = 20

        # Create local tangent plane grid
        local_x = np.linspace(-refl_area_size, refl_area_size, n_points)
        local_y = np.linspace(-refl_area_size, refl_area_size, n_points)
        X_local, Y_local = np.meshgrid(local_x, local_y)

        # Surface normal at reflection point
        surface_normal = refl_pos / np.linalg.norm(refl_pos)

        # Create approximate local surface
        surface_points = np.zeros((n_points, n_points, 3))

        for i in range(n_points):
            for j in range(n_points):
                local_point = (refl_pos + 
                              X_local[i,j] * ellipse_x_axis + 
                              Y_local[i,j] * ellipse_y_axis)
                # Project to Earth surface
                dist = np.linalg.norm(local_point)
                surface_point = local_point * (Constants.R_EARTH_MEAN / dist)
                surface_points[i,j,:] = surface_point

        ax2.plot_surface(surface_points[:,:,0], surface_points[:,:,1], surface_points[:,:,2], 
                        alpha=0.5, color='lightgreen')

        # Reflection point and ellipse
        ax2.scatter(*refl_pos, color='green', s=150, label='Reflection Point')

        # Plot footprint points
        ax2.scatter(footprint_points[:,0], footprint_points[:,1], footprint_points[:,2], 
                   color='red', s=30, label='Footprint Boundary')

        # Ellipse axes
        ax2.quiver(*refl_pos, *(ellipse_x_axis * semi_axis_x), 
                  color='orange', linewidth=3, label=f'X-axis ({beamwidth_deg[0]}°)')
        ax2.quiver(*refl_pos, *(ellipse_y_axis * semi_axis_y), 
                  color='red', linewidth=3, label=f'Y-axis ({beamwidth_deg[1]}°)')

        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.legend()
        ax2.set_title('Footprint Detail View')

        # Plot 3: Vector decomposition diagram
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')

        # Origin at satellite
        ax3.scatter(*sat_pos, color='black', s=100, label='Satellite')

        # Original velocity
        vel_scale = 0.0001
        ax3.quiver(*sat_pos, *(v_s_e * vel_scale), 
                  color='blue', linewidth=4, label='Original Velocity (v_s_e)')

        # Beam direction
        beam_scale = l_sr * 0.2
        ax3.quiver(*sat_pos, *(n_sr_eci * beam_scale), 
                  color='purple', linewidth=3, label='Beam Direction (n_sr_eci)')

        # Parallel component (to be removed)
        parallel_component = np.dot(v_s_e, n_sr_eci) * n_sr_eci
        ax3.quiver(*sat_pos, *(parallel_component * vel_scale), 
                  color='red', linewidth=3, linestyle='--', label='Parallel Component')

        # Projected velocity (result)
        ax3.quiver(*sat_pos, *(sat_vel_projected * vel_scale), 
                  color='green', linewidth=4, label='Projected Velocity')

        # Show the decomposition
        end_parallel = sat_pos + parallel_component * vel_scale
        end_original = sat_pos + v_s_e * vel_scale
        ax3.plot([end_parallel[0], end_original[0]], 
                [end_parallel[1], end_original[1]], 
                [end_parallel[2], end_original[2]], 
                'gray', linestyle=':', linewidth=2, label='Decomposition')

        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.legend()
        ax3.set_title('Vector Decomposition')

        # Plot 4: Coordinate system verification
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')

        # Show orthogonality of ellipse axes
        ax4.scatter(*refl_pos, color='green', s=150, label='Reflection Point')

        axis_length = max(semi_axis_x, semi_axis_y)
        ax4.quiver(*refl_pos, *(ellipse_x_axis * axis_length), 
                  color='red', linewidth=4, label='Ellipse X-axis')
        ax4.quiver(*refl_pos, *(ellipse_y_axis * axis_length), 
                  color='blue', linewidth=4, label='Ellipse Y-axis')
        ax4.quiver(*refl_pos, *(n_sr_eci * axis_length), 
                  color='purple', linewidth=4, label='Beam Direction (Z)')

        # Verify orthogonality
        dot_xy = np.dot(ellipse_x_axis, ellipse_y_axis)
        dot_xz = np.dot(ellipse_x_axis, n_sr_eci)
        dot_yz = np.dot(ellipse_y_axis, n_sr_eci)

        ax4.text2D(0.05, 0.95, f'X·Y = {dot_xy:.6f}', transform=ax4.transAxes)
        ax4.text2D(0.05, 0.90, f'X·Z = {dot_xz:.6f}', transform=ax4.transAxes)
        ax4.text2D(0.05, 0.85, f'Y·Z = {dot_yz:.6f}', transform=ax4.transAxes)

        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.legend()
        ax4.set_title('Coordinate System Verification')

        # Plot 5: Ellipse in beam-perpendicular plane
        ax5 = fig.add_subplot(2, 3, 5)

        # Project everything onto the ellipse plane for 2D view
        ellipse_2d_x = []
        ellipse_2d_y = []

        for i in range(no_footprint_points):
            # Get ellipse point relative to reflection
            ellipse_point = (semi_axis_x * np.cos(theta[i]) * ellipse_x_axis + 
                           semi_axis_y * np.sin(theta[i]) * ellipse_y_axis)

            # Project onto ellipse coordinate system
            x_coord = np.dot(ellipse_point, ellipse_x_axis)
            y_coord = np.dot(ellipse_point, ellipse_y_axis)

            ellipse_2d_x.append(x_coord)
            ellipse_2d_y.append(y_coord)

        ax5.plot(ellipse_2d_x, ellipse_2d_y, 'b-', linewidth=2, label='Ellipse Boundary')
        ax5.scatter([0], [0], color='green', s=100, label='Reflection Point')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.axvline(x=0, color='orange', linestyle='--', alpha=0.5)

        ax5.set_xlabel(f'X-axis: Flight Direction ({beamwidth_deg[0]}°)')
        ax5.set_ylabel(f'Y-axis: Cross-track ({beamwidth_deg[1]}°)')
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
        ax5.legend()
        ax5.set_title('Ellipse in Beam-Perpendicular Plane')

        # Plot 6: Ray tracing verification
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')

        # Show a few ray traces
        n_rays = 8
        ray_indices = np.linspace(0, no_footprint_points-1, n_rays, dtype=int)

        for i, idx in enumerate(ray_indices):
            # Ellipse point in 3D
            ellipse_point = (semi_axis_x * np.cos(theta[idx]) * ellipse_x_axis + 
                           semi_axis_y * np.sin(theta[idx]) * ellipse_y_axis)
            point_3d = refl_pos + ellipse_point

            # Ray from satellite through ellipse point
            ray_direction = (point_3d - sat_pos) / np.linalg.norm(point_3d - sat_pos)

            # Show ray
            ray_end = sat_pos + ray_direction * l_sr * 1.2
            ax6.plot([sat_pos[0], ray_end[0]], 
                    [sat_pos[1], ray_end[1]], 
                    [sat_pos[2], ray_end[2]], 
                    'gray', alpha=0.6, linewidth=1)

            # Show intersection with Earth
            ax6.scatter(*footprint_points[idx,:], color='red', s=30)

        # Earth surface (simplified)
        ax6.plot_surface(x_earth, y_earth, z_earth, alpha=0.2, color='lightblue')
        ax6.scatter(*sat_pos, color='red', s=100, label='Satellite')
        ax6.scatter(*refl_pos, color='green', s=100, label='Reflection Point')

        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_zlabel('Z (m)')
        ax6.legend()
        ax6.set_title('Ray-Sphere Intersection Verification')

        plt.tight_layout()
        plt.show(block=True)


        return footprint_points

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
