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
                                       antenna_pattern_files: dict {(azimuth, elevation, frequency_mhz) -> json filepath}.
                                                              One steered pattern per grid steering direction; the
                                                              elevation=0 patterns also serve as the unsteered
                                                              footprint contour / boresight directivity baseline.
                                       polarizations:         list (e.g. ["LHCP", "RHCP"])
                                       bands:                 list (e.g. ["L1", "L5"])
                                       footprint_dB:          float, dB contour for the footprint
                                       max_steering_angle_deg: [azimuth, max_off_boresight] steering limit (deg)
        =========================================================================================================
        """
        if settings is None:
            raise ValueError("Settings dictionary must be provided.")
        self.antenna_lookuptable = {}
        self.steering_azimuth = []
        self.steering_elevation = []
        self.frequencies_radi_pattern = []
        self.footprint_dB_contour = settings.get("footprint_dB")
        if self.footprint_dB_contour is None:
            self.footprint_dB_contour = -3.0
        if self.footprint_dB_contour >= 0.0:
            raise ValueError("footprint_dB must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
        self.max_steering_off_boresight_deg = self._parse_max_steering_off_boresight(settings.get("max_steering_angle_deg"))

        antenna_patterns_temp = {}
        pattern_files = settings.get("antenna_pattern_files") or {}
        for antenna_az_el_fr, filepath in pattern_files.items():
            azimuth = antenna_az_el_fr[0]
            elevation = antenna_az_el_fr[1]
            frequency = antenna_az_el_fr[2]

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
                    raise ValueError(f"{message} for {antenna_az_el_fr}.")

            if not math.isclose(freq, frequency, abs_tol=1e-6):
                raise ValueError(f"Frequency mismatch for {antenna_az_el_fr}: file content reports {freq} MHz.")
            if not math.isclose(beta_n, elevation, abs_tol=0.01):
                raise ValueError(f"Elevation pointing mismatch for {antenna_az_el_fr}: file content reports {beta_n} deg.")
            if not math.isclose(beta_m, azimuth, abs_tol=0.01):
                raise ValueError(f"Azimuth pointing mismatch for {antenna_az_el_fr}: file content reports {beta_m} deg.")

            antenna_patterns_temp[antenna_az_el_fr] = {
                "azimuth": azimuth,
                "elevation": elevation,
                "frequency": frequency,
                "radiation_pattern_dB": mesh,
            }
            if frequency not in self.frequencies_radi_pattern:
                self.frequencies_radi_pattern.append(frequency)
            if elevation not in self.steering_elevation:
                self.steering_elevation.append(elevation)
            if azimuth not in self.steering_azimuth:
                self.steering_azimuth.append(azimuth)
        self.polarizations = settings.get("polarizations")
        self.bands = settings.get("bands")
        self.footprint_resolution = 64
        self.R_antenna_to_body = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

        self._initialize(antenna_patterns_temp)

    def _initialize(self, antenna_patterns_temp):
        """
        =========================================================================================================
        Initialize antenna properties (per-pattern directivity, footprint contour, frequency list, steering grid).

        INPUTS:
            antenna_patterns_temp: [dict]   {(az, el, freq) -> {azimuth, elevation, frequency, radiation_pattern_dB}}
        =========================================================================================================
        """
        for antenna_key, antenna_pattern in antenna_patterns_temp.items():
            directivity, directivity_dB, contour_vectors = self._calc_antenna_footprint(antenna_pattern)
            self.antenna_lookuptable[antenna_key] = {
                'directivity': directivity,
                'directivity_dB': directivity_dB,
                'vector_bundle': contour_vectors,
                'frequency': antenna_pattern["frequency"],
                'elevation': antenna_pattern["elevation"],
                'azimuth': antenna_pattern["azimuth"],
                'radiation_pattern_dB': antenna_pattern["radiation_pattern_dB"],
            }

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
        self._build_steering_grid()
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
                if any(math.isclose(key[2], freq_Mz, abs_tol=1e-6) for key in self.antenna_lookuptable.keys()):
                    frequencies.append(freq_Mz)
                else:
                    raise ValueError(f"No radiation pattern available for frequency {freq_Mz} MHz (band {band}).")

        return frequencies

    def _validate_unsteered_footprint_patterns(self) -> None:
        """
        =========================================================================================================
        Validate that every configured antenna frequency has an unsteered (elevation=0) footprint source pattern.

        Footprint projection starts from an elevation=0 pattern and rotates that contour toward the reflection
        direction. Without an elevation=0 pattern the footprint would be based on an already-steered pattern.
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

    def _build_steering_grid(self) -> None:
        """
        =========================================================================================================
        Pre-compute, per available frequency, sorted azimuth/elevation steering grids and a (az, el) -> key map.
        Consumed by _steering_bracket_keys_and_weights to bracket a target steering direction in O(log N).

        Stored on self._steering_grids: dict {freq_mhz: {"az_sorted", "el_sorted", "key_map"}}.
        =========================================================================================================
        """
        self._steering_grids = {}
        for freq in sorted({key[2] for key in self.antenna_lookuptable}):
            freq_keys = [k for k in self.antenna_lookuptable if math.isclose(k[2], freq, abs_tol=1e-6)]
            self._steering_grids[freq] = {
                "az_sorted": np.array(sorted({k[0] for k in freq_keys})),
                "el_sorted": np.array(sorted({k[1] for k in freq_keys})),
                "key_map": {(k[0], k[1]): k for k in freq_keys},
            }

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
        phi_rad = np.linspace(0, 360, rad_pattern_spherical_dB.shape[1], endpoint=False) * np.pi / 180   # 0-360° endpoint-exclusive

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

        # Footprint projection starts from the unsteered (elevation=0) contour and rotates it
        # to the requested reflection direction. The stored steered patterns are used for gain
        # lookup; using them here would apply the steering twice.
        footprint_lookup_key = self._resolve_unsteered_footprint_lookup_key(frequency)

        # Check if the direction is the same as boresight (no rotation needed)
        if np.allclose(refl_dir_antenna_frame, boresight):
            return self.antenna_lookuptable[footprint_lookup_key]["vector_bundle"]

        # Compute rotation axis and angle
        axis = np.cross(boresight, refl_dir_antenna_frame)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # refl_dir_antenna_frame is opposite to boresight; rotate 180 deg about any perpendicular axis
            axis = np.array([1.0, 0.0, 0.0])
            angle = np.pi
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(boresight, refl_dir_antenna_frame), -1.0, 1.0))

        # Rodrigues rotation matrix
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
        """Absolute smallest angular difference in degrees on a circle, in [0, 180]."""
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _steering_bracket_keys_and_weights(self, frequency_mhz: float, az_target: float, el_target: float) -> list:
        """
        =========================================================================================================
        Return the four steering pattern keys + bilinear weights bracketing (az_target, el_target) for the
        closest available frequency. Azimuth is circular (wraps at 360 deg); elevation is clamped to the
        configured pattern range. Some weights may be 0 when the target sits on a grid line or outside the grid.

        INPUTS:
            frequency_mhz: [float]   Target frequency [MHz] (snapped to closest available)
            az_target:     [float]   Target steering azimuth [deg]
            el_target:     [float]   Target steering elevation / off-boresight [deg]

        OUTPUT:
            corners: [list[tuple]]   [(pattern_key, weight), ...] of length 4; weights sum to 1.
        =========================================================================================================
        """
        if not self._steering_grids:
            raise ValueError("Antenna steering grid is empty.")

        closest_freq = min(self._steering_grids.keys(), key=lambda f: abs(f - float(frequency_mhz)))
        grid = self._steering_grids[closest_freq]
        az_sorted = grid["az_sorted"]
        el_sorted = grid["el_sorted"]
        key_map = grid["key_map"]

        # Elevation: linear axis with edge clamping
        el_clamped = float(np.clip(el_target, el_sorted[0], el_sorted[-1]))
        if el_sorted.size == 1 or el_clamped <= el_sorted[0]:
            el_lo_idx = el_hi_idx = 0
            w_el = 0.0
        elif el_clamped >= el_sorted[-1]:
            el_lo_idx = el_hi_idx = el_sorted.size - 1
            w_el = 0.0
        else:
            el_hi_idx = int(np.searchsorted(el_sorted, el_clamped, side="right"))
            el_lo_idx = el_hi_idx - 1
            w_el = (el_clamped - el_sorted[el_lo_idx]) / (el_sorted[el_hi_idx] - el_sorted[el_lo_idx])

        # Azimuth: circular axis (wraparound across the 0/360 seam)
        az_wrapped = float(az_target) % 360.0
        if az_sorted.size == 1:
            az_lo_idx = az_hi_idx = 0
            w_az = 0.0
        else:
            az_hi_raw = int(np.searchsorted(az_sorted, az_wrapped, side="right"))
            if az_hi_raw == 0 or az_hi_raw == az_sorted.size:
                az_lo_idx = az_sorted.size - 1
                az_hi_idx = 0
                az_lo_val = az_sorted[az_lo_idx]
                az_hi_val = az_sorted[az_hi_idx] + 360.0
                az_eff = az_wrapped if az_wrapped >= az_lo_val else az_wrapped + 360.0
                w_az = (az_eff - az_lo_val) / (az_hi_val - az_lo_val)
            else:
                az_hi_idx = az_hi_raw
                az_lo_idx = az_hi_idx - 1
                w_az = (az_wrapped - az_sorted[az_lo_idx]) / (az_sorted[az_hi_idx] - az_sorted[az_lo_idx])

        az_lo, az_hi = az_sorted[az_lo_idx], az_sorted[az_hi_idx]
        el_lo, el_hi = el_sorted[el_lo_idx], el_sorted[el_hi_idx]

        return [
            (key_map[(az_lo, el_lo)], (1.0 - w_az) * (1.0 - w_el)),
            (key_map[(az_hi, el_lo)], w_az * (1.0 - w_el)),
            (key_map[(az_lo, el_hi)], (1.0 - w_az) * w_el),
            (key_map[(az_hi, el_hi)], w_az * w_el),
        ]

    def _resolve_unsteered_footprint_lookup_key(self, frequency_mhz: float) -> tuple:
        """
        =========================================================================================================
        Resolve the unsteered (elevation=0) pattern key used as the base footprint contour.

        At zero off-boresight the steering azimuth is physically arbitrary, so we require elevation=0 for the
        requested frequency and then pick the available azimuth closest to 90 deg to keep a stable convention.

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

        The INI uses ``max_steering_angle_deg = [azimuth, max_off_boresight]``. Azimuth is circular, so the
        elevation / off-boresight component is the useful field for rejecting impossible steering requests.

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

        The beam steers to point at the reflection direction; requests beyond the max steering angle are rejected.

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (azimuth_deg, elevation_deg) used to select / interpolate steered patterns,
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
    def _bilinear_pattern_sample_linear(pattern_db: np.ndarray, theta_deg: float, phi_deg: float) -> float:
        """
        =========================================================================================================
        Bilinear-interpolated pattern sample at (theta, phi), returned in linear (not dB) units.

        Theta grid: endpoint-inclusive [0, 180], step = 180 / (n_theta - 1).
        Phi grid: endpoint-exclusive [0, 360), step = 360 / n_phi, with wraparound across the seam (matches the
        AntennaPatternGenerator convention). Interpolation is in linear power so values near pattern nulls are
        not biased upward by log-space averaging.

        INPUTS:
            pattern_db: [np.ndarray, shape (n_theta, n_phi)]   Radiation pattern [dB]
            theta_deg:  [float]                                Polar angle [deg], 0..180
            phi_deg:    [float]                                Azimuth angle [deg], any (wraps mod 360)

        OUTPUT:
            value: [float]   Bilinear-interpolated pattern value in linear units (peak == 1.0)
        =========================================================================================================
        """
        n_theta, n_phi = pattern_db.shape

        theta_clamped = float(np.clip(theta_deg, 0.0, 180.0))
        theta_pos = theta_clamped * (n_theta - 1) / 180.0
        t_lo = int(np.floor(theta_pos))
        t_hi = min(t_lo + 1, n_theta - 1)
        wt = theta_pos - t_lo

        phi_pos = (float(phi_deg) % 360.0) * n_phi / 360.0
        phi_floor = np.floor(phi_pos)
        p_lo = int(phi_floor) % n_phi
        p_hi = (p_lo + 1) % n_phi
        wp = phi_pos - phi_floor

        # Vectorised dB -> linear at the 4 corner cells, then weighted sum.
        corners_db = pattern_db[[t_lo, t_lo, t_hi, t_hi], [p_lo, p_hi, p_lo, p_hi]]
        corners_lin = np.power(10.0, corners_db / 10.0)
        weights = np.array([(1.0 - wt) * (1.0 - wp), (1.0 - wt) * wp, wt * (1.0 - wp), wt * wp])
        return float(np.dot(corners_lin, weights))

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

        The beam auto-steers to the look direction (capped by max_steering_off_boresight_deg). Gain is the
        bilinear interpolation, in linear power, of the four stored steered patterns bracketing the steering
        command; each contributes its own directivity (so scan loss enters through the per-direction directivity).

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame
            frequency_mhz:           [float]                    Carrier frequency [MHz]
            azimuth_deg:             [float | None]             Steering azimuth override [deg] (optional)
            elevation_deg:           [float | None]             Steering elevation override [deg] (optional)

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

        # Bilinear interpolation of absolute gain across the 4 steering patterns bracketing
        # (steering_azimuth_deg, steering_elevation_deg), evaluated in linear power.
        corners = self._steering_bracket_keys_and_weights(frequency_mhz, steering_azimuth_deg, steering_elevation_deg)
        total_linear = 0.0
        for key, weight in corners:
            if weight <= 0.0:
                continue
            lookup = self.antenna_lookuptable[key]
            pattern_linear = self._bilinear_pattern_sample_linear(
                lookup["radiation_pattern_dB"], theta_deg, phi_deg
            )
            directivity_linear = 10.0 ** (lookup["directivity_dB"] / 10.0)
            total_linear += weight * directivity_linear * pattern_linear

        if total_linear <= 0.0:
            return np.nan
        return float(10.0 * np.log10(total_linear))

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
        for _i, vector in enumerate(vector_bundle):
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
