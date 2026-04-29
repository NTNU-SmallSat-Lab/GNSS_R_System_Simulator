import numpy as np
import pymap3d
from scipy.optimize import minimize

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.Satellite_def import Satellite


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
        self.link_budget = {}

    def compute_reflection_coordinates(
        self,
        GNSS_state: Satellite,
        Sat_state: Satellite,
        times: np.ndarray,
        antenna: RxAntenna | None = None,
        frequency_hz: float | None = None,
        link_budget_settings: dict | lb.BistaticLinkBudgetSettings | None = None,
        q_sat: np.ndarray | None = None,
    ):
        if not isinstance(times, np.ndarray):
            times = fn.to_datetime64_seconds(times)

        if frequency_hz is None:
            frequency_hz = Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
        if q_sat is None:
            q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # Preallocate outputs for efficiency
        num_positions = Sat_state.positions_eci.shape[1]
        theta_vec = np.full((num_positions, 3), np.nan)
        specular_point_vec = np.full((num_positions, 3), np.nan)
        specular_point_ecef_vec = None

        # Calculate the angles (theta) between GPS and satellite positions
        dot_product = np.einsum("ij,ij->j", GNSS_state.positions_eci, Sat_state.positions_eci)
        magnitude_a = np.linalg.norm(GNSS_state.positions_eci, axis=0)
        magnitude_b = np.linalg.norm(Sat_state.positions_eci, axis=0)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        theta = np.degrees(np.arccos(cos_theta))

        # Precompute line-of-sight (LOS) and valid theta indices for efficiency
        los_mask = np.array([self._check_los_satellites(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i]) for i in range(num_positions)])
        valid_indices = (theta < self.cutoff_angle) & los_mask
        valid_index_array = np.where(valid_indices)[0]

        # Loop through satellite positions with valid LOS to compute reflection points
        if self.earth_model == "elliptical" and valid_index_array.size > 0:
            specular_point_ecef_vec = np.full((num_positions, 3), np.nan)
            valid_times = times[valid_index_array]

            gnss_pos_ecef = np.column_stack(
                pymap3d.eci2ecef(
                    GNSS_state.positions_eci[0, valid_index_array],
                    GNSS_state.positions_eci[1, valid_index_array],
                    GNSS_state.positions_eci[2, valid_index_array],
                    valid_times,
                )
            )
            sat_pos_ecef = np.column_stack(
                pymap3d.eci2ecef(
                    Sat_state.positions_eci[0, valid_index_array],
                    Sat_state.positions_eci[1, valid_index_array],
                    Sat_state.positions_eci[2, valid_index_array],
                    valid_times,
                )
            )

            specular_init_guess_ecef = np.full((valid_index_array.size, 3), np.nan)
            specular_init_guess_eci = np.full((valid_index_array.size, 3), np.nan)
            for local_idx, i in enumerate(valid_index_array):
                try:
                    spec_init_guess_eci, _, _ = self.get_reflection_point_ff(GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i])
                    if np.isfinite(spec_init_guess_eci).all():
                        specular_init_guess_eci[local_idx] = spec_init_guess_eci
                except Exception:
                    pass

            valid_init_guess = np.isfinite(specular_init_guess_eci[:, 0])
            if valid_init_guess.any():
                specular_init_guess_ecef[valid_init_guess, :] = np.column_stack(
                    pymap3d.eci2ecef(
                        specular_init_guess_eci[valid_init_guess, 0],
                        specular_init_guess_eci[valid_init_guess, 1],
                        specular_init_guess_eci[valid_init_guess, 2],
                        valid_times[valid_init_guess],
                    )
                )

            for local_idx, i in enumerate(valid_index_array):
                specular_point_ecef, theta_degrees2, theta_degrees3 = self._find_specular_point_ellipse_ecef(
                    gnss_pos_ecef[local_idx],
                    sat_pos_ecef[local_idx],
                    initial_guess_ecef=specular_init_guess_ecef[local_idx],
                )
                theta_vec[i, :] = [theta_degrees2, theta_degrees3, theta[i]]
                specular_point_ecef_vec[i, :] = specular_point_ecef

            valid_ecef_points = ~np.isnan(specular_point_ecef_vec[:, 0])
            if valid_ecef_points.any():
                specular_point_vec[valid_ecef_points, :] = np.column_stack(
                    pymap3d.ecef2eci(
                        specular_point_ecef_vec[valid_ecef_points, 0],
                        specular_point_ecef_vec[valid_ecef_points, 1],
                        specular_point_ecef_vec[valid_ecef_points, 2],
                        times[valid_ecef_points],
                    )
                )
        else:
            for i in valid_index_array:
                reflection_point, theta_2_circ, theta_3_circ = self.get_reflection_point_ff(GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i])
                theta_vec[i, :] = [theta_2_circ, theta_3_circ, theta[i]]
                specular_point_vec[i, :] = reflection_point

        # Compute geodetic coordinates for valid specular points
        valid_specular_points = ~np.isnan(specular_point_vec[:, 0])
        latlon_ref = np.full((num_positions, 2), np.nan)
        if valid_specular_points.any():
            if self.earth_model == "elliptical" and specular_point_ecef_vec is not None:
                valid_ecef_points = ~np.isnan(specular_point_ecef_vec[:, 0])
                lat, lon, _ = pymap3d.ecef2geodetic(
                    specular_point_ecef_vec[valid_ecef_points, 0],
                    specular_point_ecef_vec[valid_ecef_points, 1],
                    specular_point_ecef_vec[valid_ecef_points, 2],
                )
                latlon_ref[valid_ecef_points, :] = np.column_stack((lat, lon))
            else:
                lat, lon, _ = pymap3d.eci2geodetic(
                    specular_point_vec[valid_specular_points, 0],
                    specular_point_vec[valid_specular_points, 1],
                    specular_point_vec[valid_specular_points, 2],
                    times[valid_specular_points],
                )
                latlon_ref[valid_specular_points, :] = np.column_stack((lat, lon))
            lons, lats = self._prevent_longitude_wraparound(latlon_ref[:, 1], latlon_ref[:, 0])
        else:
            lons = latlon_ref[:, 1]
            lats = latlon_ref[:, 0]

        incidence_deg = np.full(num_positions, np.nan, dtype=float)
        for i in range(num_positions):
            angle_candidates = theta_vec[i, :2]
            finite_mask = np.isfinite(angle_candidates)
            if finite_mask.any():
                incidence_deg[i] = float(np.mean(angle_candidates[finite_mask]))

        link_budget = lb.compute_reflection_link_budget(
            tx_positions_eci=GNSS_state.positions_eci,
            rx_positions_eci=Sat_state.positions_eci,
            rx_velocities_eci=Sat_state.velocities_eci,
            reflection_positions_eci=specular_point_vec,
            incidence_deg=incidence_deg,
            antenna=antenna,
            frequency_hz=frequency_hz,
            settings=link_budget_settings,
            q_sat=q_sat,
        )
        power = link_budget["received_power_w"]

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
        self.link_budget = link_budget
        self.power = power  # Received reflection power [W]

    def get_reflection_point_ff(self, gnssPos, satPos):
        """
        Calculate the reflection point between a satellite and a GNSS satellite position
        for a spherical Earth, using the same method as in FreeFlyer (the `_ff` suffix
        denotes FreeFlyer).
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
        Rot = self.get_rotation_3D_to_plane_coords(satCoord, GPSCoord)

        # Rotate satellite and GPS coordinates
        satCoord1 = Rot @ satCoord
        GPSCoord1 = Rot @ GPSCoord

        # Find 2D reflection point
        N, A, B = self.get_reflection_point2d(satCoord1.flatten(), GPSCoord1.flatten())

        # Transform reflection point to 3D (z = 0)
        N = np.array([N[0], N[1], 0])

        # Convert reflection point back to ECEF/ECI coordinates
        ReflectionPoint = Rot.T @ N

        # Calculate angles between vectors
        vec_1 = ReflectionPoint - np.array([0, 0, 0])
        vec_2 = satPos - ReflectionPoint
        vec_3 = gnssPos - ReflectionPoint

        theta_degrees2 = self._angle_between_vectors(vec_1, vec_2)
        theta_degrees3 = self._angle_between_vectors(vec_1, vec_3)

        return ReflectionPoint, theta_degrees2, theta_degrees3

    def find_specular_point_ellipse(self, Tx_eci, Rx_eci, time_i):
        # Check if transmitting and receiving satellites have a clear line of sight (if so there is no LOS => no reflection)
        if not self._check_los_satellites(Rx_eci, Tx_eci):
            # No line of sight, return NaN values
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # The WGS84 ellipsoid is body-fixed, so the specular geometry is solved in ECEF.
        Tx_ecef = np.array(pymap3d.eci2ecef(Tx_eci[0], Tx_eci[1], Tx_eci[2], time_i), dtype=float)
        Rx_ecef = np.array(pymap3d.eci2ecef(Rx_eci[0], Rx_eci[1], Rx_eci[2], time_i), dtype=float)

        S_ecef, theta_degrees2, theta_degrees3 = self._find_specular_point_ellipse_ecef(Tx_ecef, Rx_ecef)
        if np.isnan(S_ecef).any():
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        S_eci = np.array(pymap3d.ecef2eci(S_ecef[0], S_ecef[1], S_ecef[2], time_i), dtype=float)

        return S_eci, theta_degrees2, theta_degrees3

    def _find_specular_point_ellipse_ecef(self, Tx_ecef, Rx_ecef, initial_guess_ecef=None):
        # Radius at the poles for elliptical Earth model
        a = Constants.RADIUS_EARTH_EQUAT
        b = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)

        Tx_solve = np.asarray(Tx_ecef, dtype=float)
        Rx_solve = np.asarray(Rx_ecef, dtype=float)

        # Ellipsoid constraint
        def ellipsoid_constraint(S):
            return (S[0] ** 2 / a**2 + S[1] ** 2 / a**2 + S[2] ** 2 / b**2) - 1

        def ellipsoid_normal(S):
            S = np.array(S)
            n_surface = np.array([S[0] / a**2, S[1] / a**2, S[2] / b**2])
            return n_surface / np.linalg.norm(n_surface)

        def unit_vector(vec):
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                raise ZeroDivisionError("Cannot normalize near-zero vector.")
            return vec / norm

        # Reflection constraint
        def reflection_residual(S):
            """
            For specular reflection, mirror the incoming propagation direction
            about the local ellipsoid normal and compare it to the outgoing
            propagation direction.
            """
            S = np.array(S, dtype=float)
            v_in = unit_vector(S - Tx_solve)  # Incoming propagation direction, Tx -> S
            v_out = unit_vector(Rx_solve - S)  # Outgoing propagation direction, S -> Rx
            n_surface = ellipsoid_normal(S)

            v_reflected = v_in - 2.0 * np.dot(v_in, n_surface) * n_surface
            cos_incident = -np.dot(v_in, n_surface)
            cos_reflected = np.dot(v_out, n_surface)

            return v_reflected - v_out, cos_incident, cos_reflected

        def objective_function(S):
            try:
                S = np.array(S, dtype=float)
                return float(np.linalg.norm(S - Tx_solve) + np.linalg.norm(Rx_solve - S))
            except (ZeroDivisionError, np.linalg.LinAlgError):
                # Avoid this point if the reflection error calculation fails
                return 1e10

        if initial_guess_ecef is not None and np.isfinite(initial_guess_ecef).all():
            spec_init_guess = np.asarray(initial_guess_ecef, dtype=float)
        else:
            # Initial guess from the spherical FF solution in the same ECEF frame.
            try:
                spec_init_guess, _, _ = self.get_reflection_point_ff(Tx_solve, Rx_solve)
                if np.any(np.isnan(spec_init_guess)):
                    spec_init_guess = (Tx_solve + Rx_solve) / 2
            except Exception:
                spec_init_guess = (Tx_solve + Rx_solve) / 2

        # Ensure the initial guess is on the ellipsoid
        scale_denominator = (spec_init_guess[0] / a) ** 2 + (spec_init_guess[1] / a) ** 2 + (spec_init_guess[2] / b) ** 2
        if not np.isfinite(scale_denominator) or scale_denominator <= 0.0:
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan
        spec_init_guess = spec_init_guess / np.sqrt(scale_denominator)

        # Run optimization to find the specular point
        result = minimize(
            objective_function,
            spec_init_guess,
            constraints={"type": "eq", "fun": ellipsoid_constraint},
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 500, "disp": False},
        )

        S = result.x

        # coerce exactly onto ellipsoid (avoid tiny drift)
        scale = 1.0 / np.sqrt((S[0] / a) ** 2 + (S[1] / a) ** 2 + (S[2] / b) ** 2)
        S = scale * S

        residual, cos_inc, cos_refl = reflection_residual(S)
        residual_norm = np.linalg.norm(residual)

        if not (cos_inc > 0.0 and cos_refl > 0.0 and residual_norm <= 5e-2):
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        theta_degrees2 = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
        theta_degrees3 = np.degrees(np.arccos(np.clip(cos_refl, -1, 1)))

        return S, theta_degrees2, theta_degrees3

    def get_rotation_3D_to_plane_coords(self, satCoord, GPSCoord):  # noqa: N802
        # Calculate the rotation matrix to transform from 3D coordinates to plane coordinates
        plane = np.cross(satCoord.flatten(), GPSCoord.flatten())
        plane = plane / np.linalg.norm(plane)
        z = np.array([0, 0, 1])

        vec = np.cross(plane, z)
        s = np.linalg.norm(vec)

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

    def _compute_footprint(self, index: int, antenna: RxAntenna, frequency: float, q_sat=None, earth_model="spherical") -> tuple:
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
