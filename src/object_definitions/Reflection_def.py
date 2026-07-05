import numpy as np
import pymap3d
from scipy.optimize import least_squares

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.Satellite_def import Satellite


class Reflection:
    """Specular reflections for one (Tx GNSS, Rx LEO) satellite pair over a time window.

    One ``Reflection`` holds every specular reflection produced by a single transmitting GNSS satellite
    and a single receiving LEO satellite across the propagated time vector. The specular point is solved
    per ``earth_model`` -- a closed-form spherical solver or a two-angle ellipsoid stationarity solve -- and,
    when a link-budget configuration is supplied, the received power is evaluated at each reflection.

    Parameters
    ----------
    tx_satID : Any
        Transmitting (GNSS) satellite identifier.
    rx_satID : Any
        Receiving (LEO) satellite identifier.
    earth_model : {"spherical", "elliptical"}, optional
        Surface model for the specular solver (default "spherical").
    max_central_angle_deg : float, keyword-only
        Maximum Tx-Rx geocentric central angle [deg]; a coarse geometric pre-filter. Required.
    gnss_const : str, optional
        GNSS constellation label (GPS, GLONASS, GALILEO, BEIDOU).

    Attributes
    ----------
    time_vec : numpy.ndarray
        UTC timestamps of the samples.
    refl_ECI_vec : numpy.ndarray
        Specular points in ECI [m], shape (n, 3); NaN where no valid reflection.
    refl_ECEF_vec : numpy.ndarray or None
        Specular points in ECEF [m] (elliptical model only; None for spherical).
    refl_LL_vec : numpy.ndarray
        Specular points as (latitude, longitude) [deg], shape (n, 2).
    refl_ang_vec : numpy.ndarray
        Incidence angle at the reflection point wrt the local surface normal [deg], shape (n,).
    angle_diagnostics : numpy.ndarray
        Solver diagnostics [theta_in, theta_out, central_angle] per sample, shape (n, 3).
    link_budget : dict
        Link-budget output from ``compute_reflection_link_budget`` (empty in deferred mode).
    power : numpy.ndarray or None
        Received reflection power [W] per sample (None in deferred mode).
    """

    def __init__(
        self,
        tx_satID=None,
        rx_satID=None,
        earth_model=None,
        *,
        max_central_angle_deg,
        gnss_const=None,
    ):
        """Validate the earth model and satellite IDs, then initialise the empty reflection store."""
        # ENTRY point to the Object (Initialization)
        if earth_model is None:
            # If not specified, use the 'spherical' earth model
            print("Earth model not specified - Using default ('spherical')")
            earth_model = "spherical"
        elif not Reflection._is_valid_earth_model(earth_model):
            raise ValueError(f"Invalid earth_model {earth_model!r}. Use either 'spherical' or 'elliptical'.")

        if tx_satID is None or rx_satID is None:
            raise ValueError("Transmitting and Receiving satellite IDs are required to initialize a Reflection object.")
        try:
            max_central_angle_deg = float(max_central_angle_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_central_angle_deg must be finite and in (0, 180] deg.") from exc
        if not np.isfinite(max_central_angle_deg) or not (0.0 < max_central_angle_deg <= 180.0):
            raise ValueError("max_central_angle_deg must be finite and in (0, 180] deg.")
        self.initialization(earth_model, max_central_angle_deg, tx_satID, rx_satID, gnss_const)

    def initialization(self, earth_model, max_central_angle_deg, tx_satID, rx_satID, gnss_const):
        """Initialise the empty reflection-store attributes from the validated inputs."""
        self.earth_model = earth_model  # 'spherical' or 'elliptical'
        self.gnss_const = gnss_const  # Could be [GPS, GLONASS, GALILEO, BEIDOU]
        self.max_central_angle_deg = float(max_central_angle_deg)
        self.tx_satID = tx_satID
        self.rx_satID = rx_satID
        self.time_vec = []
        self.refl_ECI_vec = []
        self.refl_LL_vec = None
        self.refl_ECEF_vec = None     # ECEF reflection points (None for spherical); set in compute_reflection_coordinates
        self.gnss_pos_vec = []
        self.gnss_vel_vec = []
        self.sat_pos_vec = []
        self.sat_vel_vec = []
        self.refl_ang_vec = []        # incidence angle wrt local surface normal [deg], shape (n,)
        self.angle_diagnostics = []   # solver diagnostics [theta_in, theta_out, central_angle], shape (n, 3)
        self.link_budget = {}

    def compute_reflection_coordinates(
        self,
        GNSS_state: Satellite,
        Sat_state: Satellite,
        times: np.ndarray,
        tx_eirp_dbw: float,
        antenna: RxAntenna | None = None,
        frequency_hz: float | None = None,
        link_budget_settings: dict | lb.BistaticLinkBudgetSettings | None = None,
        q_sat: np.ndarray | None = None,
        compute_link_budget: bool = True,
    ):
        """Solve the specular point at every sample and (optionally) evaluate the link budget.

        Fills the reflection attributes in place. The specular geometry is solved per ``earth_model``
        (spherical closed form or elliptical stationary-path solve); samples failing the central-angle gate or
        the line-of-sight test are left as NaN.

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
            Receiver antenna (None => isotropic 0 dBi).
        frequency_hz : float or None, optional
            Carrier frequency [Hz]; defaults to GPS L1.
        link_budget_settings : dict or BistaticLinkBudgetSettings or None, optional
            Rx-system link-budget settings.
        q_sat : numpy.ndarray or None, optional
            Attitude quaternion [w, x, y, z] (default identity).
        compute_link_budget : bool, optional
            If False, skip the link budget (deferred per-event mode); default True.
        """
        times = fn.to_utc_datetime_array(times)

        if frequency_hz is None:
            frequency_hz = Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
        if q_sat is None:
            q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # Preallocate outputs for efficiency
        num_positions = Sat_state.positions_eci.shape[1]
        theta_vec = np.full((num_positions, 3), np.nan)
        specular_point_vec = np.full((num_positions, 3), np.nan)
        specular_point_ecef_vec = None

        # Central angle between Tx and Rx position vectors (rotation-invariant — ECI or ECEF works).
        theta = fn.angle_between_vectors(GNSS_state.positions_eci, Sat_state.positions_eci)

        # Precompute line-of-sight (LOS) and valid theta indices for efficiency
        if self.earth_model == "elliptical":
            los_mask = np.array(
                [
                    self._check_los_ellipsoid(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i])
                    for i in range(num_positions)
                ]
            )
        else:
            los_mask = np.array(
                [
                    self._check_los_satellites(
                        Sat_state.positions_eci[:, i],
                        GNSS_state.positions_eci[:, i],
                        earth_radius=Constants.R_EARTH_MEAN,
                    )
                    for i in range(num_positions)
                ]
            )
        valid_indices = (theta < self.max_central_angle_deg) & los_mask
        valid_index_array = np.where(valid_indices)[0]

        # Loop through satellite positions with valid LOS to compute reflection points
        if self.earth_model == "elliptical" and valid_index_array.size > 0:
            specular_point_ecef_vec = np.full((num_positions, 3), np.nan)
            valid_times = times[valid_index_array]
            # pymap3d expects a Python list/sequence of datetimes, not an
            # ndarray; tolist() unwraps the object array to plain datetimes.
            valid_times_list = list(valid_times)

            gnss_pos_ecef = np.column_stack(
                pymap3d.eci2ecef(
                    GNSS_state.positions_eci[0, valid_index_array],
                    GNSS_state.positions_eci[1, valid_index_array],
                    GNSS_state.positions_eci[2, valid_index_array],
                    valid_times_list,
                )
            )
            sat_pos_ecef = np.column_stack(
                pymap3d.eci2ecef(
                    Sat_state.positions_eci[0, valid_index_array],
                    Sat_state.positions_eci[1, valid_index_array],
                    Sat_state.positions_eci[2, valid_index_array],
                    valid_times_list,
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
                        list(valid_times[valid_init_guess]),
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
                        list(times[valid_ecef_points]),
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
                    list(times[valid_specular_points]),
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

        if compute_link_budget:
            link_budget = lb.compute_reflection_link_budget(
                tx_positions_eci=GNSS_state.positions_eci,
                rx_positions_eci=Sat_state.positions_eci,
                rx_velocities_eci=Sat_state.velocities_eci,
                reflection_positions_eci=specular_point_vec,
                incidence_deg=incidence_deg,
                antenna=antenna,
                frequency_hz=frequency_hz,
                tx_eirp_dbw=tx_eirp_dbw,
                settings=link_budget_settings,
                q_sat=q_sat,
                surface_normals_eci=(
                    lb.ellipsoid_surface_normals_eci(specular_point_ecef_vec, times)
                    if self.earth_model == "elliptical" and specular_point_ecef_vec is not None
                    else None
                ),
                tx_velocities_eci=GNSS_state.velocities_eci,
            )
            power = link_budget["received_power_w"]
        else:
            # Deferred mode: the link budget is computed per detection event by the caller.
            link_budget = {}
            power = None

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
        self.refl_ang_vec = incidence_deg  # Incidence angle wrt local surface normal [deg], shape (n,)
        self.angle_diagnostics = theta_vec  # Solver diagnostics [theta_in, theta_out, central_angle], shape (n, 3)
        self.refl_ECEF_vec = specular_point_ecef_vec  # ECEF reflection points (None for spherical); for deferred per-event budget
        self.link_budget = link_budget
        self.power = power  # Received reflection power [W]

    def get_reflection_point_ff(self, gnssPos, satPos, earth_radius=Constants.R_EARTH_MEAN):
        """Specular point on a spherical Earth via the FreeFlyer method (the ``_ff`` suffix).

        Rotates the Tx/Rx geometry into the orbital plane, solves the 2D specular point, and rotates back.

        Parameters
        ----------
        gnssPos : numpy.ndarray
            GNSS (Tx) satellite position [m], shape (3,).
        satPos : numpy.ndarray
            Receiver (LEO) satellite position [m], shape (3,).
        earth_radius : float, optional
            Sphere radius [m] (default ``Constants.R_EARTH_MEAN``).

        Returns
        -------
        tuple
            (reflection_point, theta_degrees2, theta_degrees3): the 3D specular point (NaNs if no
            line of sight), the incidence angle and the reflected angle from the local normal [deg].
        """
        if not self._check_los_satellites(np.asarray(satPos).flatten(), np.asarray(gnssPos).flatten(), earth_radius=earth_radius):
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # Convert positions to column vectors
        satCoord = np.array(satPos).reshape(-1, 1)
        txCoord = np.array(gnssPos).reshape(-1, 1)

        # Get rotation matrix
        Rot = self.get_rotation_3D_to_plane_coords(satCoord, txCoord)

        # Rotate satellite and GPS coordinates
        satCoord1 = Rot @ satCoord
        txCoord1 = Rot @ txCoord

        # Find 2D reflection point
        N, A, B = self.get_reflection_point2d(satCoord1.flatten(), txCoord1.flatten(), earth_radius=earth_radius)
        if N is None:
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

        # Transform reflection point to 3D (z = 0)
        N = np.array([N[0], N[1], 0])

        # Convert reflection point back to ECEF/ECI coordinates
        ReflectionPoint = Rot.T @ N

        # Calculate angles between vectors
        vec_1 = ReflectionPoint - np.array([0, 0, 0])
        vec_2 = satPos - ReflectionPoint
        vec_3 = gnssPos - ReflectionPoint

        theta_degrees2 = fn.angle_between_vectors(vec_1, vec_2)
        theta_degrees3 = fn.angle_between_vectors(vec_1, vec_3)

        return ReflectionPoint, theta_degrees2, theta_degrees3

    def _find_specular_point_ellipse_ecef(self, Tx_ecef, Rx_ecef, initial_guess_ecef=None):
        """Find the visible WGS-84 specular point from the stationary-path condition.

        The surface is parameterised by two angular variables, so every iterate lies exactly on the
        ellipsoid. Solving the two tangent components of ``grad(|S-T| + |R-S|) = 0`` is substantially
        better conditioned than minimising metre-scale Cartesian ranges with an equality constraint.
        """
        a = Constants.RADIUS_EARTH_EQUAT
        b = Constants.RADIUS_EARTH_EQUAT * (1 - Constants.EARTH_FLATTENING)

        Tx_solve = np.asarray(Tx_ecef, dtype=float)
        Rx_solve = np.asarray(Rx_ecef, dtype=float)
        invalid_result = (np.array([np.nan, np.nan, np.nan]), np.nan, np.nan)
        if Tx_solve.shape != (3,) or Rx_solve.shape != (3,):
            return invalid_result
        if not (np.isfinite(Tx_solve).all() and np.isfinite(Rx_solve).all()):
            return invalid_result

        def surface_point(latitude_param, longitude_param):
            cos_lat = np.cos(latitude_param)
            return np.array(
                [
                    a * cos_lat * np.cos(longitude_param),
                    a * cos_lat * np.sin(longitude_param),
                    b * np.sin(latitude_param),
                ]
            )

        def ellipsoid_normal(S):
            """
            Returns the normal vector to the ellipsoid at the given point.
            Raises ZeroDivisionError if the point is at the origin.
            """
            n_surface = np.array([S[0] / a**2, S[1] / a**2, S[2] / b**2])
            normal_norm = np.linalg.norm(n_surface)
            if normal_norm <= 0.0:
                raise ZeroDivisionError("Cannot construct an ellipsoid normal at the origin.")
            return n_surface / normal_norm

        def unit_vector(vec):
            """
            Returns the unit vector of the input vector. Raises ZeroDivisionError for near-zero vectors.
            """
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                raise ZeroDivisionError("Cannot normalize near-zero vector.")
            return vec / norm

        def reflection_residual(S):
            v_in = unit_vector(S - Tx_solve)  # Incoming propagation direction, Tx -> S
            v_out = unit_vector(Rx_solve - S)  # Outgoing propagation direction, S -> Rx
            n_surface = ellipsoid_normal(S)

            v_reflected = v_in - 2.0 * np.dot(v_in, n_surface) * n_surface
            cos_incident = -np.dot(v_in, n_surface)
            cos_reflected = np.dot(v_out, n_surface)

            return v_reflected - v_out, cos_incident, cos_reflected

        def tangent_stationarity(angles):
            """
            Returns the tangent components of the gradient of the total path length |S-T| + |R-S| at the
            given latitude and longitude parameters. Raises ZeroDivisionError if the surface point is at the origin.
            """
            latitude_param, longitude_param = angles
            S = surface_point(latitude_param, longitude_param)
            try:
                n_tx2s = unit_vector(S - Tx_solve)
                n_s2rx = unit_vector(Rx_solve - S)
            except ZeroDivisionError:
                return np.array([1.0e3, 1.0e3])

            # The surface gradient of total path length has no tangent component at a specular point.
            grad_path = n_tx2s - n_s2rx
            sin_lat = np.sin(latitude_param)
            cos_lat = np.cos(latitude_param)
            sin_lon = np.sin(longitude_param)
            cos_lon = np.cos(longitude_param)
            tangent_lat = np.array([-a * sin_lat * cos_lon, -a * sin_lat * sin_lon, b * cos_lat])
            tangent_lon = np.array([-a * cos_lat * sin_lon, a * cos_lat * cos_lon, 0.0])

            tangent_lat_norm = np.linalg.norm(tangent_lat)
            tangent_lon_norm = np.linalg.norm(tangent_lon)
            if tangent_lat_norm <= 0.0:
                return np.array([1.0e3, 1.0e3])
            tangent_lat /= tangent_lat_norm

            # Longitude is singular at a pole. Any orthogonal tangent direction is valid there.
            if tangent_lon_norm <= 1e-12:
                tangent_lon = np.cross(ellipsoid_normal(S), tangent_lat)
                tangent_lon_norm = np.linalg.norm(tangent_lon)
            if tangent_lon_norm <= 0.0:
                return np.array([1.0e3, 1.0e3])
            tangent_lon /= tangent_lon_norm
            return np.array([np.dot(grad_path, tangent_lat), np.dot(grad_path, tangent_lon)])

        if initial_guess_ecef is not None and np.isfinite(initial_guess_ecef).all():
            spec_init_guess = np.asarray(initial_guess_ecef, dtype=float)
        else:
            # Initial guess from the spherical FF solution in the same ECEF frame; use the
            # equatorial radius so the guess sits on the ellipsoid's largest dimension and
            # rescaling to the ellipsoid below is a near-identity (helps nonlinear-solver convergence).
            try:
                spec_init_guess, _, _ = self.get_reflection_point_ff(Tx_solve, Rx_solve, earth_radius=a)
                if np.any(np.isnan(spec_init_guess)):
                    spec_init_guess = (Tx_solve + Rx_solve) / 2
            except Exception:
                spec_init_guess = (Tx_solve + Rx_solve) / 2

        scaled_guess = np.array([spec_init_guess[0] / a, spec_init_guess[1] / a, spec_init_guess[2] / b])
        scaled_norm = np.linalg.norm(scaled_guess)
        if not np.isfinite(scaled_norm) or scaled_norm <= 0.0:
            return invalid_result
        scaled_guess /= scaled_norm
        latitude_0 = float(np.arcsin(np.clip(scaled_guess[2], -1.0, 1.0)))
        longitude_0 = float(np.arctan2(scaled_guess[1], scaled_guess[0]))

        # Centre the allowed 2*pi longitude interval on the seed so the +/-pi seam cannot trap
        # a solution just across the conventional longitude boundary.
        result = least_squares(
            tangent_stationarity,
            np.array([latitude_0, longitude_0]),
            bounds=(
                np.array([-np.pi / 2.0 + 1e-12, longitude_0 - np.pi]),
                np.array([np.pi / 2.0 - 1e-12, longitude_0 + np.pi]),
            ),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=200,
        )

        if not result.success or not np.isfinite(result.x).all():
            return invalid_result
        S = surface_point(result.x[0], result.x[1])
        try:
            residual, cos_inc, cos_refl = reflection_residual(S)
        except ZeroDivisionError:
            return invalid_result
        residual_norm = np.linalg.norm(residual)

        if not (cos_inc > 0.0 and cos_refl > 0.0 and residual_norm <= 5e-2):
            return invalid_result

        theta_degrees2 = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
        theta_degrees3 = np.degrees(np.arccos(np.clip(cos_refl, -1, 1)))

        return S, theta_degrees2, theta_degrees3

    def get_rotation_3D_to_plane_coords(self, satCoord, txCoord):  # noqa: N802
        """Rotation matrix mapping the Tx-Rx orbital plane onto the z = 0 plane."""
        # Calculate the rotation matrix to transform from 3D coordinates to plane coordinates
        plane = np.cross(satCoord.flatten(), txCoord.flatten())
        plane_norm = np.linalg.norm(plane)
        if plane_norm <= 1e-14:
            # Same-side parallel position vectors have a valid nadir reflection but do not define a
            # unique plane. Pick any stable plane that contains their common radial direction.
            radial = satCoord.flatten().astype(float)
            radial_norm = np.linalg.norm(radial)
            if radial_norm <= 0.0:
                raise ValueError("Cannot define a reflection plane from a zero satellite position.")
            radial /= radial_norm
            fallback = np.eye(3)[int(np.argmin(np.abs(radial)))]
            plane = np.cross(radial, fallback)
            plane_norm = np.linalg.norm(plane)
        plane = plane / plane_norm
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

        Sv = fn.skew_matrix(vec2)
        Rot = np.eye(3) + s2 * Sv + Sv @ Sv * (1 - c2)
        return Rot

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

    @staticmethod
    def get_reflection_point2d(satCoord1, txCoord1, earth_radius=Constants.R_EARTH_MEAN):
        """Specular point in the 2D orbital-plane frame (after rotation).

        Solves the quartic that places the reflection on the circle while obeying the law of reflection.

        Parameters
        ----------
        satCoord1 : numpy.ndarray
            Receiver coordinates in the 2D plane (x, y) [m].
        txCoord1 : numpy.ndarray
            GNSS coordinates in the 2D plane (x, y) [m].
        earth_radius : float, optional
            Sphere radius [m] (default ``Constants.R_EARTH_MEAN``).

        Returns
        -------
        tuple
            (N, A, B): the 2D reflection point [m] (None if no real positive root), and the normalised
            receiver and GNSS vectors.
        """
        # Calculates the reflection point in 2D
        A = satCoord1[:2] / earth_radius
        B = txCoord1[:2] / earth_radius

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
        roots = roots[np.abs(roots.imag) < 1e-9].real  # Keep only real roots; drop complex conjugate pairs

        for x in roots:
            if x > 0:
                y = (1 + x - 2 * u * x**2) / (1 + 2 * v * x)
                if y > 0:
                    N = x * A + y * B
                    return earth_radius * N, A, B
        return None, None, None

    @staticmethod
    def _check_los_satellites(r1, r2, earth_radius=Constants.R_EARTH_MEAN) -> bool:
        """
        =========================================================================================================
        Check whether a finite chord between r1 and r2 clears a spherical Earth.

        INPUTS:
            r1: [np.array, shape (3,)]   First point [m]
            r2: [np.array, shape (3,)]   Second point [m]

        OUTPUT:
            has_los: [bool]   True if the segment's closest point clears the sphere
        =========================================================================================================
        """
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
        if r1.shape != (3,) or r2.shape != (3,):
            return False
        if not (np.isfinite(r1).all() and np.isfinite(r2).all() and np.isfinite(earth_radius) and earth_radius > 0.0):
            return False

        d = r2 - r1
        d_sq = float(np.dot(d, d))
        if d_sq <= 0.0:
            return bool(np.linalg.norm(r1) >= earth_radius)

        # Exact closest point on the finite chord; sampling can miss a narrow grazing intersection.
        t_closest = float(np.clip(-np.dot(r1, d) / d_sq, 0.0, 1.0))
        closest_distance = np.linalg.norm(r1 + t_closest * d)
        return bool(closest_distance >= earth_radius)

    @staticmethod
    def _check_los_ellipsoid(
        r1,
        r2,
        a=Constants.RADIUS_EARTH_EQUAT,
        b=Constants.RADIUS_EARTH_EQUAT * (1.0 - Constants.EARTH_FLATTENING),
    ) -> bool:
        """Return whether a finite chord clears an axisymmetric ellipsoid.

        Scaling ``(x, y, z)`` by ``(a, a, b)`` maps the ellipsoid to the unit sphere, so closest
        approach on the finite scaled segment is an exact occultation test.
        """
        r1 = np.asarray(r1, dtype=float).reshape(-1)
        r2 = np.asarray(r2, dtype=float).reshape(-1)
        if r1.shape != (3,) or r2.shape != (3,):
            return False
        if not (np.isfinite(r1).all() and np.isfinite(r2).all() and np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
            return False

        scale = np.array([a, a, b], dtype=float)
        p1 = r1 / scale
        p2 = r2 / scale
        segment = p2 - p1
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 0.0:
            return bool(np.linalg.norm(p1) >= 1.0)

        t_closest = float(np.clip(-np.dot(p1, segment) / segment_norm_sq, 0.0, 1.0))
        closest_point = p1 + t_closest * segment
        return bool(np.linalg.norm(closest_point) >= 1.0)

    @staticmethod
    def _is_valid_earth_model(earth_model) -> bool:
        # Check if earth_model is either 'spherical' or 'elliptical'
        valid_models = ["spherical", "elliptical"]
        return earth_model in valid_models

    @staticmethod
    def _prevent_longitude_wraparound(lons, lats):
        # Find where the difference between consecutive longitudes exceeds 180 degrees -> avoid plotting lines from -179 to 179 deg longitude
        wraparound = np.abs(np.diff(lons)) > 180
        # Replace the longitude and latitude at the wraparound point with NaN
        lons[1:][wraparound] = np.nan
        lats[1:][wraparound] = np.nan
        return lons, lats
