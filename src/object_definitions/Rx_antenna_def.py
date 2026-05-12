import ast
import json
import math
import warnings

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
                                       antenna_pattern_files: dict {(az, el, freq) -> json filepath}
                                       polarizations:         list (e.g. ["LHCP", "RHCP"])
                                       bands:                 list (e.g. ["L1", "L5"])
                                       footprint_dB:          float, dB contour for the footprint
        =========================================================================================================
        """
        if settings is None:
            raise ValueError("Settings dictionary must be provided.")
        # Read in each "json" file (antenna pattern for all elevations and azimuths)
        self.antenna_lookuptable = {}
        self.steering_azimuth = []
        self.steering_elevation = []
        self.frequencies_radi_pattern = []
        self.bands = []
        self.footprint_dB_contour = settings.get("footprint_dB")
        if self.footprint_dB_contour is None:
            self.footprint_dB_contour = -3.0
        if self.footprint_dB_contour >= 0.0:
            raise ValueError("footprint_dB must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
        self.max_steering_off_boresight_deg = self._parse_max_steering_off_boresight(settings.get("max_steering_angle_deg"))

        antenna_patterns_temp = {}
        for antenna_az_el_fr in settings.get("antenna_pattern_files"):
            azimuth = antenna_az_el_fr[0]
            elevation = antenna_az_el_fr[1]
            frequency = antenna_az_el_fr[2]
            filepath = settings.get("antenna_pattern_files")[antenna_az_el_fr]

            # Read antenna pattern data from the JSON file
            with open(filepath) as f:
                antenna_data = json.load(f)

            mesh = np.array(antenna_data['data'])
            freq = antenna_data['frequency']
            beta_m = antenna_data['azimuth_pointing']
            beta_n = antenna_data['elevation_pointing']

            # Robustness checks (missing data)
            checks = [
                (mesh is None or mesh.size == 0, "Missing or empty mesh data"),
                (freq is None, "Missing frequency data"),
                (beta_m is None, "Missing azimuth pointing data"), 
                (beta_n is None, "Missing elevation pointing data")
            ]
            for condition, message in checks:
                if condition:
                    raise ValueError(f"{message} for {antenna_az_el_fr}.")

            # Robustness checks (value consistency)
            if freq != frequency:
                raise ValueError(f"Frequency mismatch for file and file-content {antenna_az_el_fr}. Expected {frequency}, got {freq}.")
            if not math.isclose(beta_n, elevation, abs_tol=0.01):
                raise ValueError(f"Elevation pointing mismatch for file and file-content {antenna_az_el_fr}. Expected {elevation}, got {beta_n}.")
            if not math.isclose(beta_m, azimuth, abs_tol=0.01):
                raise ValueError(f"Azimuth pointing mismatch for file and file-content {antenna_az_el_fr}. Expected {azimuth}, got {beta_m}.")

            # Dictionary with antenna properties read in
            antenna_patterns_temp[antenna_az_el_fr] = {
                "azimuth": azimuth,
                "elevation": elevation,
                "frequency": frequency,
                "radiation_pattern_dB": mesh
            }
            # Add the new frequency, elevation or azimuth
            if frequency not in self.frequencies_radi_pattern:
                self.frequencies_radi_pattern.append(frequency)
            if elevation not in self.steering_elevation:
                self.steering_elevation.append(elevation) 
            if azimuth not in self.steering_azimuth:
                self.steering_azimuth.append(azimuth)
        self.polarizations = settings.get("polarizations") # Antenna polarizations (e.g., ["LHCP", "RHCP"])
        self.bands = settings.get("bands") # Antenna bands
        self.footprint_resolution = 64  # Number of points to create a circle (higher resolution for smoother circles) | hardcoded to 64
        self.R_antenna_to_body = np.array([ # Rotation matrix from antenna frame to body-fixed frame
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

        self._initialize(antenna_patterns_temp)

    def _initialize(self, antenna_patterns_temp):
        """
        =========================================================================================================
        Initialize antenna properties (per-pattern directivity, footprint contour, frequency list).

        INPUTS:
            antenna_patterns_temp: [dict]   {(az, el, freq) -> {azimuth, elevation, frequency, radiation_pattern_dB}}
        =========================================================================================================
        """
        # Calculate "footprint" of the antenna at 1m distance using splines
        for antenna_key, antenna_pattern in antenna_patterns_temp.items():
            directivity, directivity_dB, contour_vectors = self._calc_antenna_footprint(antenna_pattern)

            # Store datasets in a lookup table
            self.antenna_lookuptable[antenna_key] = {
                'directivity': directivity,
                'directivity_dB': directivity_dB,
                'vector_bundle': contour_vectors,
                'frequency': antenna_pattern["frequency"],
                'elevation': antenna_pattern["elevation"],
                'azimuth': antenna_pattern["azimuth"],
                'radiation_pattern_dB': antenna_pattern["radiation_pattern_dB"]
            }

        # Initialize antenna frequecies
        self.frequencies = self._get_frequency_from_band()
        if self.max_steering_off_boresight_deg is not None:
            configured_pattern_elevations = [
                key[1]
                for key in self.antenna_lookuptable
                if any(math.isclose(key[2], frequency, abs_tol=1e-6) for frequency in self.frequencies)
            ]
            if configured_pattern_elevations:
                max_pattern_elevation = max(configured_pattern_elevations)
                if self.max_steering_off_boresight_deg > max_pattern_elevation + 1e-6:
                    warnings.warn(
                        "max_steering_angle_deg exceeds the generated antenna-pattern elevation range "
                        f"({self.max_steering_off_boresight_deg:.2f} deg > {max_pattern_elevation:.2f} deg). "
                        "Directions beyond the available pattern range will use the nearest pattern.",
                        stacklevel=2,
                    )
        self._validate_unsteered_footprint_patterns()

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
                # Check if a radiation pattern for this frequency is available:
                if any(key[2] == freq_Mz for key in self.antenna_lookuptable.keys()):
                    frequencies.append(freq_Mz)
                else:
                    raise ValueError(f"No radiation pattern available for frequency {freq_Mz} MHz (band {band}).")

        return frequencies

    def _validate_unsteered_footprint_patterns(self) -> None:
        """
        =========================================================================================================
        Validate that every configured antenna frequency has an unsteered footprint source pattern.

        Footprint projection starts from an elevation=0 pattern and rotates that contour toward the reflection
        direction. Without an elevation=0 pattern, the footprint would be based on an already-steered pattern and
        steering could be applied twice.
        =========================================================================================================
        """
        missing = []
        for frequency_mhz in self.frequencies:
            has_unsteered_pattern = any(
                math.isclose(key[2], frequency_mhz, abs_tol=1e-6) and math.isclose(key[1], 0.0, abs_tol=1e-6)
                for key in self.antenna_lookuptable
            )
            if not has_unsteered_pattern:
                missing.append(frequency_mhz)

        if missing:
            missing_str = ", ".join(f"{frequency:.2f} MHz" for frequency in missing)
            raise ValueError(
                "Missing unsteered antenna pattern for footprint calculation "
                f"at {missing_str}. Include 0 deg in [ANTENNA PATTERN].elevation_directivity."
            )

    def _calc_antenna_footprint(self, antenna_pattern):
        """
        =========================================================================================================
        Calculate the antenna footprint with vectors radiating from the antenna position outwards
        in the antenna frame.

        INPUTS:
            antenna_pattern: [dict]   Pattern entry with radiation_pattern_dB / elevation / azimuth / frequency

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
            plotting_object = PlotRadiationPattern(spherical_coordinates=(rad_pattern_spherical_lin, theta_mesh, phi_mesh), arrow = (antenna_pattern["elevation"], antenna_pattern["azimuth"]))
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

        # Footprint projection starts from the unsteered contour and rotates it
        # to the requested reflection direction. Steered patterns remain valid
        # for gain lookup, but using them here would apply steering twice.
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

    @staticmethod
    def _circular_abs_diff_deg(a: float, b: float) -> float:
        """
        =========================================================================================================
        Absolute smallest angular difference in degrees on a circle.

        INPUTS:
            a: [float]   First angle [deg]
            b: [float]   Second angle [deg]

        OUTPUT:
            diff: [float]   Smallest absolute circular difference in [0, 180] deg
        =========================================================================================================
        """
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _resolve_lookup_key(self, frequency_mhz: float, azimuth_deg: float | None = None, elevation_deg: float | None = None) -> tuple:
        """
        =========================================================================================================
        Resolve the closest antenna-pattern key for the requested frequency and steering angles.

        INPUTS:
            frequency_mhz: [float]         Target frequency [MHz]
            azimuth_deg:   [float | None]  Steering azimuth [deg] (default: 90.0 — unsteered convention)
            elevation_deg: [float | None]  Steering elevation [deg] (default: 0.0)

        OUTPUT:
            key: [tuple]   (azimuth, elevation, frequency) of the closest pattern
        =========================================================================================================
        """
        if not self.antenna_lookuptable:
            raise ValueError("Antenna lookup table is empty.")

        freq_target = float(frequency_mhz)
        az_target = 90.0 if azimuth_deg is None else float(azimuth_deg)
        el_target = 0.0 if elevation_deg is None else float(elevation_deg)

        keys = list(self.antenna_lookuptable.keys())
        return min(
            keys,
            key=lambda key: (
                abs(key[2] - freq_target),
                self._circular_abs_diff_deg(key[0], az_target),
                abs(key[1] - el_target),
            ),
        )

    def _resolve_unsteered_footprint_lookup_key(self, frequency_mhz: float) -> tuple:
        """
        =========================================================================================================
        Resolve the unsteered pattern key used as the base footprint contour.

        At zero off-boresight, steering azimuth is physically arbitrary. We therefore require elevation=0 for the
        requested frequency and then choose the available azimuth closest to 90 deg to keep the existing convention.

        INPUTS:
            frequency_mhz: [float]   Target frequency [MHz]

        OUTPUT:
            key: [tuple]   (azimuth, 0.0, frequency) footprint source pattern
        =========================================================================================================
        """
        if not self.antenna_lookuptable:
            raise ValueError("Antenna lookup table is empty.")

        closest_frequency = min({key[2] for key in self.antenna_lookuptable}, key=lambda available_freq: abs(available_freq - float(frequency_mhz)))
        unsteered_keys = [
            key
            for key in self.antenna_lookuptable
            if math.isclose(key[2], closest_frequency, abs_tol=1e-6) and math.isclose(key[1], 0.0, abs_tol=1e-6)
        ]
        if not unsteered_keys:
            raise ValueError(
                "Missing unsteered antenna pattern for footprint calculation "
                f"at {closest_frequency:.2f} MHz. Include 0 deg in [ANTENNA PATTERN].elevation_directivity."
            )

        return min(unsteered_keys, key=lambda key: self._circular_abs_diff_deg(key[0], 90.0))

    @staticmethod
    def _parse_max_steering_off_boresight(value) -> float | None:
        """
        =========================================================================================================
        Parse the configured max steering angle into an off-boresight limit.

        The INI uses ``max_steering_angle_deg = [azimuth, elevation]``. Azimuth is circular, so the
        elevation/off-boresight component is the useful field for rejecting impossible steering requests.

        INPUTS:
            value: [float | list | tuple | str | None]   Configured steering limit

        OUTPUT:
            limit: [float | None]   Maximum off-boresight steering angle in degrees, or None if unlimited
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
                raise ValueError(f"Invalid max_steering_angle_deg value: {value}") from exc

        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size == 0:
            return None
        limit = values[1] if values.size >= 2 else values[0]
        if limit < 0:
            raise ValueError("max_steering_angle_deg must not contain a negative off-boresight limit.")
        return float(limit)

    @staticmethod
    def _direction_to_spherical_angles_deg(direction: np.ndarray) -> tuple[float, float] | None:
        """
        =========================================================================================================
        Convert a unit look direction in antenna coordinates to pattern angles.

        INPUTS:
            direction: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (azimuth_deg, elevation_deg/off-boresight) or None if invalid/behind antenna
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

        elevation_deg = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
        azimuth_deg = float((np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0)
        return azimuth_deg, elevation_deg

    def resolve_steering_angles_for_direction(self, direction_antenna_frame: np.ndarray) -> tuple[float, float] | None:
        """
        =========================================================================================================
        Resolve the automatic steering command for a look direction in antenna coordinates.

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (azimuth_deg, elevation_deg) used to select the closest generated pattern,
                                     or None if the direction is invalid or outside the steering limit
        =========================================================================================================
        """
        angles = self._direction_to_spherical_angles_deg(direction_antenna_frame)
        if angles is None:
            return None

        azimuth_deg, elevation_deg = angles
        if self.max_steering_off_boresight_deg is not None and elevation_deg > self.max_steering_off_boresight_deg:
            return None
        return azimuth_deg, elevation_deg

    @staticmethod
    def _sample_radiation_pattern_db(pattern_db: np.ndarray, theta_deg: float, phi_deg: float) -> float:
        """
        =========================================================================================================
        Nearest-neighbor sample of a radiation pattern at (theta, phi).

        INPUTS:
            pattern_db: [np.ndarray, shape (n_theta, n_phi)]   Radiation pattern [dB]
            theta_deg:  [float]   Polar angle [deg], 0..180
            phi_deg:    [float]   Azimuth angle [deg], 0..360

        OUTPUT:
            value: [float]   Pattern value [dB] at the nearest grid point
        =========================================================================================================
        """
        n_theta, n_phi = pattern_db.shape
        theta_idx = int(np.clip(np.round(theta_deg * (n_theta - 1) / 180.0), 0, n_theta - 1))
        phi_idx = int(np.clip(np.round((phi_deg % 360.0) * (n_phi - 1) / 360.0), 0, n_phi - 1))
        return float(pattern_db[theta_idx, phi_idx])

    def get_gain_dbi_for_direction(
        self,
        direction_antenna_frame: np.ndarray,
        frequency_mhz: float,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
    ) -> float:
        """
        =========================================================================================================
        Get absolute gain [dBi] for a given look direction in antenna coordinates.

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame
            frequency_mhz:           [float]                    Carrier frequency [MHz]
            azimuth_deg:             [float | None]             Steering azimuth [deg] (optional)
            elevation_deg:           [float | None]             Steering elevation [deg] (optional)

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
        phi_deg, theta_deg = angles

        if azimuth_deg is None or elevation_deg is None:
            steering_angles = self.resolve_steering_angles_for_direction(direction)
            if steering_angles is None:
                return np.nan
            auto_azimuth_deg, auto_elevation_deg = steering_angles
            steering_azimuth_deg = auto_azimuth_deg if azimuth_deg is None else float(azimuth_deg)
            steering_elevation_deg = auto_elevation_deg if elevation_deg is None else float(elevation_deg)
        else:
            steering_azimuth_deg = float(azimuth_deg)
            steering_elevation_deg = float(elevation_deg)

        if self.max_steering_off_boresight_deg is not None and steering_elevation_deg > self.max_steering_off_boresight_deg:
            return np.nan

        lookup_key = self._resolve_lookup_key(frequency_mhz, azimuth_deg=steering_azimuth_deg, elevation_deg=steering_elevation_deg)
        lookup = self.antenna_lookuptable[lookup_key]
        relative_gain_db = self._sample_radiation_pattern_db(lookup["radiation_pattern_dB"], theta_deg, phi_deg)

        # Pattern is relative to peak (0 dB), so add directivity offset for absolute gain.
        return float(lookup["directivity_dB"] + relative_gain_db)

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
                arrow:                 [tuple]   Optional (elevation_deg, azimuth_deg) — draws an arrow
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
            el = arrow[0] * np.pi / 180
            az = arrow[1] * np.pi / 180
            arr_x = r * np.sin(el) * np.cos(az)
            arr_y = r * np.sin(el) * np.sin(az)
            arr_z = r * np.cos(el)

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
