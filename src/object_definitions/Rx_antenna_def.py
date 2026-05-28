import ast
import json
import math

from matplotlib import pyplot as plt
import numpy as np

from object_definitions.Constants_def import Constants


class RxAntenna:
    def __init__(self, settings: dict = None):
        """
        =========================================================================================================
        RxAntenna class:
            - Receiver antenna object initialized from a settings dictionary parsed from the [ANTENNA]
              INI section.

            INPUTS:
                settings: [dict]   Antenna configuration with keys:
                                       antenna_pattern_files: dict {frequency_mhz -> json filepath}.
                                                              Each JSON is an unsteered (boresight) pattern
                                                              used for the footprint contour and as the
                                                              boresight directivity baseline.
                                       polarizations:         list (e.g. ["LHCP", "RHCP"])
                                       bands:                 list (e.g. ["L1", "L5"])
                                       footprint_dB:          float, dB contour for the footprint
                                       array_geometry:        optional diagnostic-only dict with keys
                                                              elements_along_x, elements_along_y, d_x, d_y.
                                                              Fixed-nadir gain queries use only the JSON pattern.
        =========================================================================================================
        """
        if settings is None:
            raise ValueError("Settings dictionary must be provided.")
        self.antenna_lookuptable = {}
        self.footprint_dB_contour = settings.get("footprint_dB")
        if self.footprint_dB_contour is None:
            self.footprint_dB_contour = -3.0
        if self.footprint_dB_contour >= 0.0:
            raise ValueError("footprint_dB must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
        self.max_fov_off_boresight_deg = self._parse_max_fov_off_boresight(settings.get("max_fov_angle_deg"))

        antenna_patterns_temp = {}
        pattern_files = settings.get("antenna_pattern_files") or {}
        for frequency, filepath in pattern_files.items():
            with open(filepath) as f:
                antenna_data = json.load(f)

            mesh = np.array(antenna_data['data'])
            freq = antenna_data['frequency']
            beta_m = antenna_data['azimuth_pointing']
            beta_n = antenna_data['elevation_pointing']

            checks = [
                (mesh is None or mesh.size == 0, "Missing or empty mesh data"),
                (freq is None, "Missing frequency data"),
                (beta_m is None, "Missing azimuth pointing data"),
                (beta_n is None, "Missing elevation pointing data"),
            ]
            for condition, message in checks:
                if condition:
                    raise ValueError(f"{message} for {frequency} MHz pattern.")

            if not math.isclose(freq, frequency, abs_tol=1e-6):
                raise ValueError(f"Frequency mismatch for pattern at {frequency} MHz: file content reports {freq} MHz.")
            if not (math.isclose(beta_m, 0.0, abs_tol=0.01) and math.isclose(beta_n, 0.0, abs_tol=0.01)):
                raise ValueError(
                    f"Pattern for {frequency} MHz must be unsteered (azimuth_pointing = elevation_pointing = 0); "
                    f"got ({beta_m}, {beta_n})."
                )

            antenna_patterns_temp[frequency] = {
                "frequency": frequency,
                "radiation_pattern_dB": mesh,
            }
        self.polarizations = settings.get("polarizations")
        self.bands = settings.get("bands")
        self.footprint_resolution = 64
        self.R_antenna_to_body = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        self._array_geometry = settings.get("array_geometry")

        self._initialize(antenna_patterns_temp)

    def _initialize(self, antenna_patterns_temp):
        """
        =========================================================================================================
        Initialize antenna properties (per-frequency directivity, footprint contour, frequency list).

        INPUTS:
            antenna_patterns_temp: [dict]   {frequency_mhz -> {frequency, radiation_pattern_dB}}
        =========================================================================================================
        """
        for frequency, antenna_pattern in antenna_patterns_temp.items():
            directivity, directivity_dB, contour_vectors = self._calc_antenna_footprint(antenna_pattern)
            self.antenna_lookuptable[frequency] = {
                'directivity': directivity,
                'directivity_dB': directivity_dB,
                'vector_bundle': contour_vectors,
                'frequency': frequency,
                'radiation_pattern_dB': antenna_pattern["radiation_pattern_dB"],
            }

        self.frequencies = self._get_frequency_from_band()

    def _get_frequency_from_band(self):
        """
        =========================================================================================================
        Get the antenna frequencies (MHz) from the configured bands.

        OUTPUT:
            frequencies: [list[float]]   Center frequencies in MHz for each configured band
        =========================================================================================================
        """
        frequencies = []

        for band in self.bands:
            if band in Constants.BANDS_CENTER_FREQUENCIES_Hz:
                freq_Mz = Constants.BANDS_CENTER_FREQUENCIES_Hz[band] / 1e6  # Convert Hz to MHz
                if any(math.isclose(key, freq_Mz, abs_tol=1e-6) for key in self.antenna_lookuptable.keys()):
                    frequencies.append(freq_Mz)
                else:
                    raise ValueError(f"No radiation pattern available for frequency {freq_Mz} MHz (band {band}).")

        return frequencies

    def _calc_antenna_footprint(self, antenna_pattern):
        """
        =========================================================================================================
        Calculate the antenna footprint with vectors radiating from the antenna position outwards
        in the antenna frame.

        INPUTS:
            antenna_pattern: [dict]   Pattern entry with radiation_pattern_dB

        OUTPUT:
            result: [tuple]   (directivity, directivity_dB, contour_vectors)
        =========================================================================================================
        """
        rad_pattern_spherical_dB = antenna_pattern["radiation_pattern_dB"]
        # Convert to linear scale
        rad_pattern_spherical_lin = 10**(rad_pattern_spherical_dB/10) # Convert radiation pattern to linear scale (assumption: Power Pattern)

        theta_rad = np.linspace(0, 180, rad_pattern_spherical_dB.shape[0]) * np.pi / 180 # 0-180°
        phi_rad = np.linspace(0, 360, rad_pattern_spherical_dB.shape[1]) * np.pi / 180   # 0-360°

        theta_mesh, phi_mesh = np.meshgrid(theta_rad, phi_rad, indexing='ij')

        directivity, directivity_dB = self._calculate_directivity(rad_pattern_spherical_lin, theta_rad, phi_rad, theta_mesh)

        # Find angle with self.footprint_dB_contour
        contur_vectors = self._find_contour_vectors(rad_pattern_spherical_dB, theta_rad, phi_rad, theta_mesh, phi_mesh)

        # DEBUG, PLOT ANTENNA RADIATION PATTERN
        debug = False
        if debug:
            plotting_object = PlotRadiationPattern(spherical_coordinates=(rad_pattern_spherical_lin, theta_mesh, phi_mesh), arrow=(0.0, 0.0))
            plotting_object.add_vector_bundle_to_plot(contur_vectors)
            plotting_object.show()

        # Do resampling of the footprint to a lower resolution (self.footprint_resolution)
        contur_vectors = self._smooth_antenna_footprint(contur_vectors)

        ##DEBUG Visualize the antenna footprint in 3D
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.quiver(0, 0, 0, contur_vectors[:, 0], contur_vectors[:, 1], contur_vectors[:, 2])
        # plt.show(block=True)

        return directivity, directivity_dB, contur_vectors

    def _smooth_antenna_footprint(self, contour_vectors):
        """
        =========================================================================================================
        Smooth the antenna footprint by resampling on a 1m projection plane in front of the antenna.

        INPUTS:
            contour_vectors: [np.ndarray, shape (N, 3)]   Original contour unit vectors in antenna frame

        OUTPUT:
            resampled_vectors: [np.ndarray, shape (footprint_resolution, 3)]   Resampled contour vectors
        =========================================================================================================
        """
        # Project the footprint onto a "plane" 1m in front of the antenna
        nadir_n = np.array([0, 0, 1])  # Nadir pointing vector in antenna coordinate system

        # Original azimuth angles for the contour vectors
        num_original_points = len(contour_vectors)
        original_azimuths = np.linspace(0, 2*np.pi, num_original_points, endpoint=False)

        # Target azimuth angles for resampled footprint
        target_azimuths = np.linspace(0, 2*np.pi, self.footprint_resolution, endpoint=False)

        # Convert contour vectors to polar coordinates on the 1m plane
        original_rhos = []

        for vector in contour_vectors:
            # Angle between nadir and vector
            off_nadir_angle = np.arccos(np.clip(np.dot(nadir_n, vector), -1.0, 1.0))
            # Radial distance on plane 1m in front of the antenna
            rho = np.tan(off_nadir_angle)
            original_rhos.append(rho)

        original_rhos = np.array(original_rhos)

        # Interpolate rho values at target azimuth angles (handle wraparound by extending the arrays)
        extended_azimuths = np.concatenate([original_azimuths - 2*np.pi, original_azimuths, original_azimuths + 2*np.pi]) # Extend the azimuth range from -2π to +2π for interpolation
        extended_rhos = np.concatenate([original_rhos, original_rhos, original_rhos])

        # Interpolate rho values for target azimuths
        resampled_rhos = np.interp(target_azimuths, extended_azimuths, extended_rhos)

        # Convert back to 3D vectors
        resampled_vectors = []
        for _, (azimuth, rho) in enumerate(zip(target_azimuths, resampled_rhos, strict=True)):
            # Cartesian coordinates on the plane
            x = rho * np.cos(azimuth)
            y = rho * np.sin(azimuth)
            z = 1.0

            # Convert to unit vector
            vector_length = np.sqrt(x**2 + y**2 + z**2)
            unit_vector = np.array([x, y, z]) / vector_length
            resampled_vectors.append(unit_vector)

        resampled_vectors = np.array(resampled_vectors)

        return resampled_vectors
        ## DEBUG SHOW THE SPLINE 1m IN FRONT OF THE ANTENNA
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Plot original footprint
        #original_x = original_rhos * np.cos(original_azimuths)
        #original_y = original_rhos * np.sin(original_azimuths)
        #ax1.plot(original_x, original_y, 'ro-', label=f'Original ({len(contour_vectors)} points)')
        #ax1.set_xlabel('X (m)')
        #ax1.set_ylabel('Y (m)')
        #ax1.set_title('Original Antenna Footprint')
        #ax1.legend()
        #ax1.axis('equal')
        #ax1.grid(True)

        ## Plot resampled footprint
        #resampled_x = resampled_rhos * np.cos(target_azimuths)
        #resampled_y = resampled_rhos * np.sin(target_azimuths)
        #ax2.plot(resampled_x, resampled_y, 'bo-', label=f'Resampled ({self.footprint_resolution} points)')
        #ax2.set_xlabel('X (m)')
        #ax2.set_ylabel('Y (m)')
        #ax2.set_title('Resampled Antenna Footprint')
        #ax2.legend()
        #ax2.axis('equal')
        #ax2.grid(True)

        #plt.tight_layout()
        #plt.show(block=True)

    def _get_antenna_footprint(self, refl_dir_antenna_frame, frequency):
        """
        =========================================================================================================
        Get the antenna footprint contour rotated to point at the given reflection direction.

        INPUTS:
            refl_dir_antenna_frame: [np.ndarray, shape (3,)]   Reflection direction in antenna frame
            frequency:              [float]                    Frequency [MHz] (used for lookup)

        OUTPUT:
            footprint_vectors: [np.ndarray, shape (N, 3)]   Rotated footprint contour vectors
        =========================================================================================================
        """
        boresight = np.array([0.0, 0.0, 1.0])  # Nadir-pointing boresight in antenna coordinates
        refl_dir_antenna_frame = np.asarray(refl_dir_antenna_frame, dtype=float)
        if refl_dir_antenna_frame.shape != (3,):
            raise ValueError("Input 'refl_dir_antenna_frame' must be a 3-element vector.")
        if not np.isfinite(refl_dir_antenna_frame).all():
            raise ValueError("Input 'refl_dir_antenna_frame' must contain finite values.")

        direction_norm = np.linalg.norm(refl_dir_antenna_frame)
        if direction_norm <= 0.0:
            raise ValueError("Input 'refl_dir_antenna_frame' must be non-zero.")
        refl_dir_antenna_frame = refl_dir_antenna_frame / direction_norm

        # Footprint projection starts from the fixed nadir contour and rotates it
        # to the requested reflection direction for geometric footprint drawing only.
        footprint_lookup_key = self._resolve_unsteered_footprint_lookup_key(frequency)

        # Check if the direction is the same as boresight (no rotation needed)
        if np.allclose(refl_dir_antenna_frame, boresight):
            return self.antenna_lookuptable[footprint_lookup_key]["vector_bundle"]

        # Compute rotation axis and angle
        axis = np.cross(boresight, refl_dir_antenna_frame)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # refl_dir_antenna_frame is opposite to boresight, rotate 180 degrees around any perpendicular axis
            axis = np.array([1.0, 0.0, 0.0])  # Arbitrary perpendicular axis
            angle = np.pi
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(boresight, refl_dir_antenna_frame), -1.0, 1.0))

        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        rotated_footprint_vectors = self.antenna_lookuptable[footprint_lookup_key]["vector_bundle"] @ R.T
        return rotated_footprint_vectors

    def _resolve_unsteered_footprint_lookup_key(self, frequency_mhz: float) -> float:
        """
        =========================================================================================================
        Resolve the (single) unsteered pattern key for the requested frequency.

        Each frequency has exactly one stored pattern (boresight, unsteered). When the requested frequency does
        not exactly match, the closest available frequency is used.

        INPUTS:
            frequency_mhz: [float]   Target frequency [MHz]

        OUTPUT:
            key: [float]   Frequency [MHz] of the matched lookup entry
        =========================================================================================================
        """
        if not self.antenna_lookuptable:
            raise ValueError("Antenna lookup table is empty.")
        return min(self.antenna_lookuptable, key=lambda available_freq: abs(available_freq - float(frequency_mhz)))

    @staticmethod
    def _parse_max_fov_off_boresight(value) -> float | None:
        """
        =========================================================================================================
        Parse the fixed-antenna field-of-view limit.

        The INI accepts ``max_fov_angle_deg = [azimuth, max_off_boresight]`` (two-element list, the
        first element is currently ignored) or a scalar; the value used is the off-boresight half-angle.

        INPUTS:
            value: [float | list | tuple | str | None]   Configured FOV limit

        OUTPUT:
            limit: [float | None]   Maximum off-boresight angle in degrees, or None if unlimited
        =========================================================================================================
        """
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip() == "":
                return None
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid max_fov_angle_deg value: {value}") from exc

        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size == 0:
            return None
        limit = values[1] if values.size >= 2 else values[0]
        if limit < 0:
            raise ValueError("max_fov_angle_deg must not contain a negative off-boresight limit.")
        return float(limit)

    @staticmethod
    def _direction_to_spherical_angles_deg(direction: np.ndarray) -> tuple[float, float] | None:
        """
        =========================================================================================================
        Convert a unit look direction in antenna coordinates to pattern angles.

        INPUTS:
            direction: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (theta_deg, phi_deg) or None if invalid/behind antenna
        =========================================================================================================
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            return None

        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 0.0:
            return None
        direction = direction / direction_norm

        if direction[2] <= 0.0:
            return None

        theta_deg = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
        phi_deg = float((np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0)
        return theta_deg, phi_deg

    @staticmethod
    def _array_factor_db_at_direction(
        elements_along_x: int,
        elements_along_y: int,
        d_x: float,
        d_y: float,
        freq_hz: float,
        beta_x: float,
        beta_y: float,
        theta_rad,
        phi_rad,
    ):
        """
        =========================================================================================================
        Closed-form uniform planar-array factor in dB, normalized by the analytic peak (N_x * N_y).

        |sum_{n=0..N-1} exp(j n psi)| = |sin(N psi / 2) / sin(psi / 2)|, with the limit value N when psi -> 0.

        INPUTS:
            elements_along_x: [int]              Number of array elements along x
            elements_along_y: [int]              Number of array elements along y
            d_x:              [float]            Element spacing along x [m]
            d_y:              [float]            Element spacing along y [m]
            freq_hz:          [float]            Carrier frequency [Hz]
            beta_x:           [float]            Phase shift between adjacent x elements [rad]
            beta_y:           [float]            Phase shift between adjacent y elements [rad]
            theta_rad:        [float|np.ndarray] Polar angle from boresight [rad]
            phi_rad:          [float|np.ndarray] Azimuth angle [rad]

        OUTPUT:
            af_db: [float|np.ndarray]   Array-factor magnitude in dB, normalized to a 0 dB analytic peak
        =========================================================================================================
        """
        wavelength_m = Constants.SPEED_OF_LIGHT / freq_hz
        k = 2 * np.pi / wavelength_m

        theta_rad = np.asarray(theta_rad, dtype=float)
        phi_rad = np.asarray(phi_rad, dtype=float)

        psi_x = k * d_x * np.sin(theta_rad) * np.cos(phi_rad) + beta_x
        psi_y = k * d_y * np.sin(theta_rad) * np.sin(phi_rad) + beta_y

        # Use the closed-form ratio away from psi -> 0; substitute the limit value N at the peak.
        sin_half_x = np.sin(psi_x / 2.0)
        sin_half_y = np.sin(psi_y / 2.0)
        peak_tol = 1e-12
        af_x = np.where(
            np.abs(sin_half_x) < peak_tol,
            float(elements_along_x),
            np.sin(elements_along_x * psi_x / 2.0) / np.where(np.abs(sin_half_x) < peak_tol, 1.0, sin_half_x),
        )
        af_y = np.where(
            np.abs(sin_half_y) < peak_tol,
            float(elements_along_y),
            np.sin(elements_along_y * psi_y / 2.0) / np.where(np.abs(sin_half_y) < peak_tol, 1.0, sin_half_y),
        )

        epsilon = 1e-12  # avoid log(0) at array-factor nulls
        af_norm = np.abs(af_x * af_y) / (elements_along_x * elements_along_y)
        af_db = 20.0 * np.log10(af_norm + epsilon)
        return float(af_db) if af_db.ndim == 0 else af_db

    def compute_array_factor_db_on_demand(
        self,
        theta_deg: float,
        phi_deg: float,
        steering_az_deg: float,
        steering_el_deg: float,
        freq_mhz: float,
    ):
        """
        =========================================================================================================
        Diagnostic uniform planar-array factor in dB at (theta, phi) with exact steering applied.

        This helper is not used by the fixed-nadir link-budget gain path. It requires the RxAntenna
        to have been initialized explicitly with array_geometry.

        INPUTS:
            theta_deg:        [float]   Polar angle from boresight [deg]
            phi_deg:          [float]   Azimuth angle [deg]
            steering_az_deg:  [float]   Steering azimuth [deg]
            steering_el_deg:  [float]   Steering elevation (off-boresight) [deg]
            freq_mhz:         [float]   Carrier frequency [MHz]

        OUTPUT:
            af_db: [float]   Array-factor magnitude in dB (peak = 0 dB at the steering direction)
        =========================================================================================================
        """
        if getattr(self, "_array_geometry", None) is None:
            raise ValueError(
                "Array geometry not configured on RxAntenna. Pass an 'array_geometry' dict "
                "(elements_along_x, elements_along_y, d_x, d_y) in settings to enable the diagnostic AF helper."
            )
        geom = self._array_geometry
        freq_hz = float(freq_mhz) * 1e6
        wavelength_m = Constants.SPEED_OF_LIGHT / freq_hz
        k = 2 * np.pi / wavelength_m

        steer_az_rad = np.deg2rad(steering_az_deg)
        steer_el_rad = np.deg2rad(steering_el_deg)
        beta_x = -k * geom["d_x"] * np.sin(steer_el_rad) * np.cos(steer_az_rad)
        beta_y = -k * geom["d_y"] * np.sin(steer_el_rad) * np.sin(steer_az_rad)

        return self._array_factor_db_at_direction(
            geom["elements_along_x"],
            geom["elements_along_y"],
            geom["d_x"],
            geom["d_y"],
            freq_hz,
            beta_x,
            beta_y,
            np.deg2rad(theta_deg),
            np.deg2rad(phi_deg),
        )

    def get_gain_dbi_for_direction(
        self,
        direction_antenna_frame: np.ndarray,
        frequency_mhz: float,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
    ) -> float:
        """
        =========================================================================================================
        Get fixed-nadir absolute gain [dBi] for a given look direction in antenna coordinates.

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame
            frequency_mhz:           [float]                    Carrier frequency [MHz]
            azimuth_deg:             [float | None]             Retained for API compatibility; ignored.
            elevation_deg:           [float | None]             Retained for API compatibility; ignored.

        OUTPUT:
            gain_dbi: [float]   Absolute gain [dBi], or NaN if direction is invalid / behind the antenna
        =========================================================================================================
        """
        direction = np.asarray(direction_antenna_frame, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            return np.nan

        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 0.0:
            return np.nan
        direction = direction / direction_norm

        # If reflection is behind the antenna boresight hemisphere, return NaN.
        if direction[2] <= 0.0:
            return np.nan

        angles = self._direction_to_spherical_angles_deg(direction)
        if angles is None:
            return np.nan
        theta_deg, phi_deg = angles

        if self.max_fov_off_boresight_deg is not None and theta_deg > self.max_fov_off_boresight_deg:
            return np.nan

        unsteered_key = self._resolve_unsteered_footprint_lookup_key(frequency_mhz)
        pattern_db = self._sample_radiation_pattern_db(unsteered_key, theta_deg, phi_deg)
        if not np.isfinite(pattern_db):
            return np.nan

        return float(self.antenna_lookuptable[unsteered_key]["directivity_dB"] + pattern_db)

    def _sample_radiation_pattern_db(self, frequency_key: float, theta_deg: float, phi_deg: float) -> float:
        """Bilinearly sample the stored nadir-looking radiation pattern."""
        pattern = np.asarray(self.antenna_lookuptable[frequency_key]["radiation_pattern_dB"], dtype=float)
        n_theta, n_phi = pattern.shape
        if n_theta == 0 or n_phi == 0 or not (0.0 <= theta_deg <= 180.0):
            return np.nan

        theta_pos = theta_deg / 180.0 * (n_theta - 1) if n_theta > 1 else 0.0
        theta_0 = int(np.floor(theta_pos))
        theta_1 = min(theta_0 + 1, n_theta - 1)
        theta_frac = theta_pos - theta_0

        # Stored grid is np.linspace(0, 360, n_phi, endpoint=False), so bin spacing is 360/n_phi.
        # The sample after the last bin wraps periodically to bin 0.
        phi_pos = (phi_deg % 360.0) / 360.0 * n_phi if n_phi > 1 else 0.0
        phi_0 = int(np.floor(phi_pos))
        phi_1 = (phi_0 + 1) % n_phi
        phi_frac = phi_pos - phi_0

        value_00 = pattern[theta_0, phi_0]
        value_10 = pattern[theta_1, phi_0]
        value_01 = pattern[theta_0, phi_1]
        value_11 = pattern[theta_1, phi_1]

        value_0 = (1.0 - theta_frac) * value_00 + theta_frac * value_10
        value_1 = (1.0 - theta_frac) * value_01 + theta_frac * value_11
        return float((1.0 - phi_frac) * value_0 + phi_frac * value_1)

    @staticmethod
    def _calculate_directivity(rad_pattern_lin_spherical, theta_rad, phi_rad, theta_mesh):
        """
        =========================================================================================================
        Calculate the directivity of the antenna from its radiation pattern (antenna-gain).

        INPUTS:
            rad_pattern_lin_spherical: [np.ndarray]   Linear-scale spherical radiation pattern
            theta_rad:                 [np.ndarray]   Polar angle samples [rad]
            phi_rad:                   [np.ndarray]   Azimuth angle samples [rad]
            theta_mesh:                [np.ndarray]   Meshgrid of theta values

        OUTPUT:
            result: [tuple]   (directivity, directivity_dB)
        =========================================================================================================
        """
        # Calculate differential solid angle elements
        dtheta = np.gradient(theta_rad)
        dphi = np.gradient(phi_rad)
        dtheta_mesh, dphi_mesh = np.meshgrid(dtheta, dphi, indexing='ij')

        # Integrate: ∫∫ F(θ,φ) sin(θ) dθ dφ
        integrand = rad_pattern_lin_spherical * np.sin(theta_mesh) * dtheta_mesh * dphi_mesh
        total_power = np.sum(integrand)

        if total_power <= 0:
            raise ValueError("Antenna pattern integrates to non-positive total power; cannot compute directivity.")

        # Directivity = 4π / integral
        directivity = 4 * np.pi / total_power
        directivity_dB = 10 * np.log10(directivity)

        return directivity, directivity_dB

    def _find_contour_vectors(self, rad_pattern_spherical_dB, theta_rad, phi_rad, theta_mesh, phi_mesh):
        """
        =========================================================================================================
        Find unit vectors pointing from antenna center toward the target dB contour, with interpolation.

        Assumes normalized radiation pattern where peak = 0 dB and all other values <= 0 dB.

        INPUTS:
            rad_pattern_spherical_dB: [np.ndarray]   Radiation pattern [dB], shape (n_theta, n_phi)
            theta_rad:                [np.ndarray]   Polar angle samples [rad]
            phi_rad:                  [np.ndarray]   Azimuth angle samples [rad]
            theta_mesh:               [np.ndarray]   Meshgrid of theta values
            phi_mesh:                 [np.ndarray]   Meshgrid of phi values

        OUTPUT:
            contour_vectors: [np.ndarray, shape (n_phi, 3)]   Unit vectors at the dB contour
        =========================================================================================================
        """
        contour_vectors = []

        for phi_i, phi_val in enumerate(phi_rad):
            pattern_slice = rad_pattern_spherical_dB[:, phi_i]
            peak_idx = np.argmax(pattern_slice)  # Finds 0dB point

            # Search outward from peak (0dB) toward self.footprint_dB_contour
            theta = theta_rad[peak_idx]  # Default to peak

            for direction in [1, -1]:  # Forward, then backward
                start_idx = peak_idx + direction
                end_idx = len(pattern_slice) if direction == 1 else -1

                for i in range(start_idx, end_idx, direction):
                    if pattern_slice[i] <= self.footprint_dB_contour:
                        # Interpolate for sub-sample accuracy
                        prev_i = i - direction
                        if pattern_slice[prev_i] != pattern_slice[i]:
                            t = (self.footprint_dB_contour - pattern_slice[prev_i]) / (pattern_slice[i] - pattern_slice[prev_i])
                            theta = theta_rad[prev_i] + t * (theta_rad[i] - theta_rad[prev_i])
                        else:
                            theta = theta_rad[i]
                        break
                else:
                    continue  # No crossing in this direction
                break  # Found crossing, stop searching

            # Convert to unit vector
            contour_vectors.append([
                np.sin(theta) * np.cos(phi_val),
                np.sin(theta) * np.sin(phi_val),
                np.cos(theta)
            ])

        return np.array(contour_vectors)

class PlotRadiationPattern:
    def __init__(self, spherical_coordinates, arrow = None):
        """
        =========================================================================================================
        PlotRadiationPattern class:
            - Plot radiation pattern in 3D. Input must be in spherical coordinates; converted to Cartesian
              internally.

            INPUTS:
                spherical_coordinates: [tuple]   (rad_pattern_spherical, theta_mesh, phi_mesh)
                arrow:                 [tuple]   Optional (theta_deg, phi_deg) — draws an arrow
        =========================================================================================================
        """
        # Convert linear radiation pattern from spherical coordinates to cartesian coordinates
        rad_pattern_spherical = spherical_coordinates[0]
        theta_mesh = spherical_coordinates[1]
        phi_mesh = spherical_coordinates[2]
        x = rad_pattern_spherical * np.sin(theta_mesh) * np.cos(phi_mesh)
        y = rad_pattern_spherical * np.sin(theta_mesh) * np.sin(phi_mesh)
        z = rad_pattern_spherical * np.cos(theta_mesh)
        carthesian_coordinates = np.array([x, y, z]).T

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, antialiased=False)

        if arrow is not None:
            r = np.max(np.abs(carthesian_coordinates))
            theta = arrow[0] * np.pi / 180
            phi = arrow[1] * np.pi / 180
            arr_x = r * np.sin(theta) * np.cos(phi)
            arr_y = r * np.sin(theta) * np.sin(phi)
            arr_z = r * np.cos(theta)

            self.ax.quiver(0, 0, 0, arr_x, arr_y, arr_z, length=2)

        self._set_axis_properties()

    def _set_axis_properties(self, title='Antenna Radiation Pattern'):
        """
        =========================================================================================================
        Set consistent axis properties on the 3D plot.

        INPUTS:
            title: [str]   Plot title (default: 'Antenna Radiation Pattern')
        =========================================================================================================
        """
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)
        self.ax.set_title(title)

    def add_vector_bundle_to_plot(self, vector_bundle, color='blue', linewidth=1, alpha=0.7):
        """
        =========================================================================================================
        Add a bundle of vectors to the 3D plot.

        INPUTS:
            vector_bundle: [np.ndarray, shape (N, 3)]   Array of vectors
            color:         [str]                        Vector color (default: 'blue')
            linewidth:     [int]                        Vector line width (default: 1)
            alpha:         [float]                      Vector transparency (default: 0.7)
        =========================================================================================================
        """
        for i, vector in enumerate(vector_bundle):
            self.ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], 
                color=color, linewidth=linewidth, alpha=alpha)
#            self.ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
#                          length=1, color=color, linewidth=linewidth, alpha=alpha)

    def add_single_vector(self, vector, color='green', linewidth=2, label=None):
        """
        =========================================================================================================
        Add a single vector to the plot.

        INPUTS:
            vector:    [list | np.ndarray, shape (3,)]   3D vector [x, y, z]
            color:     [str]                             Vector color (default: 'green')
            linewidth: [int]                             Vector line width (default: 2)
            label:     [str | None]                      Optional label for the legend
        =========================================================================================================
        """
        self.ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                      length=1, color=color, linewidth=linewidth, label=label)

    def save_plot(self, filename, dpi=300):
        """
        =========================================================================================================
        Save the plot to file.

        INPUTS:
            filename: [str]   Output filename (with extension)
            dpi:      [int]   Resolution (default: 300)
        =========================================================================================================
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    def show(self):
        """
        =========================================================================================================
        Display the plot.
        =========================================================================================================
        """
        plt.show(block=True)
