import json
import math

from matplotlib import pyplot as plt
import numpy as np

from object_definitions.Constants_def import Constants


class RxAntenna:
    """Receiver antenna: holds the phased-array patterns and returns gain along a look direction.

    Built from the ``[ANTENNA]`` settings. It loads the generated radiation-pattern files, steers the beam
    analytically toward a requested direction, and reports the gain there (in dBi). It also owns the
    footprint contour used to project the beam onto the ground.

    Parameters
    ----------
    settings : dict
        Antenna configuration with keys: ``antenna_pattern_files`` (``{(across_track, along_track,
        frequency_mhz) -> json path}``), ``polarizations`` (e.g. ``["LHCP", "RHCP"]``), ``bands``
        (e.g. ``["L1", "L5"]``), and ``footprint_dB`` (the footprint contour level, below 0 dB).

    Notes
    -----
    The per-axis maximum steering angles are derived from the generated steering grid (the largest
    ``|across_track|`` and ``|along_track|``), not set directly.
    """

    def __init__(self, settings: dict = None):
        """Load the pattern files, validate them, and derive the steering grid and footprint baseline."""
        if settings is None:
            raise ValueError("Settings dictionary must be provided.")
        self.antenna_lookuptable = {}
        self.steering_across_track = []
        self.steering_along_track = []
        self.frequencies_radi_pattern = []
        self.footprint_dB_contour = settings.get("footprint_dB")
        if self.footprint_dB_contour is None:
            raise ValueError("footprint_dB must be provided in the antenna settings (set 'footprint_db' in the [ANTENNA] section of the configuration file).")
        if self.footprint_dB_contour >= 0.0:
            raise ValueError("footprint_dB must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
        # Per-axis max steering angles are derived from the generated steering grid in _initialize.
        self.max_across_track_steering_deg = None
        self.max_along_track_steering_deg = None

        antenna_patterns_temp = {}
        pattern_files = settings.get("antenna_pattern_files") or {}
        for antenna_key, filepath in pattern_files.items():
            across_track = antenna_key[0]
            along_track = antenna_key[1]
            frequency = antenna_key[2]

            with open(filepath) as f:
                antenna_data = json.load(f)

            mesh = np.array(antenna_data['data'])
            freq = antenna_data['frequency']
            beta_m = antenna_data['across_track_pointing']
            beta_n = antenna_data['along_track_pointing']

            checks = [
                (mesh is None or mesh.size == 0, "Missing or empty mesh data"),
                (freq is None, "Missing frequency data"),
                (beta_m is None, "Missing across_track pointing data"),
                (beta_n is None, "Missing along_track pointing data"),
            ]
            for condition, message in checks:
                if condition:
                    raise ValueError(f"{message} for {antenna_key}.")

            if not math.isclose(freq, frequency, abs_tol=1e-6):
                raise ValueError(f"Frequency mismatch for {antenna_key}: file content reports {freq} MHz.")
            if not math.isclose(beta_n, along_track, abs_tol=0.01):
                raise ValueError(f"Along-track pointing mismatch for {antenna_key}: file content reports {beta_n} deg.")
            if not math.isclose(beta_m, across_track, abs_tol=0.01):
                raise ValueError(f"Across-track pointing mismatch for {antenna_key}: file content reports {beta_m} deg.")

            antenna_patterns_temp[antenna_key] = {
                "across_track": across_track,
                "along_track": along_track,
                "frequency": frequency,
                "radiation_pattern_dB": mesh,
            }
            if frequency not in self.frequencies_radi_pattern:
                self.frequencies_radi_pattern.append(frequency)
            if along_track not in self.steering_along_track:
                self.steering_along_track.append(along_track)
            if across_track not in self.steering_across_track:
                self.steering_across_track.append(across_track)
        self.polarizations = settings.get("polarizations")
        self.bands = settings.get("bands")
        self.footprint_resolution = 64
        self.R_antenna_to_body = Constants.R_antenna_to_body

        self._initialize(antenna_patterns_temp)

    def _initialize(self, antenna_patterns_temp):
        """
        =========================================================================================================
        Initialize antenna properties (per-pattern directivity, footprint contour, frequency list, steering grid).

        INPUTS:
            antenna_patterns_temp: [dict]   {(across, along, freq) -> {across_track, along_track, frequency, radiation_pattern_dB}}
        =========================================================================================================
        """
        for antenna_key, antenna_pattern in antenna_patterns_temp.items():
            directivity, directivity_dB, contour_vectors = self._calc_antenna_footprint(antenna_pattern)
            self.antenna_lookuptable[antenna_key] = {
                'directivity': directivity,
                'directivity_dB': directivity_dB,
                'vector_bundle': contour_vectors,
                'frequency': antenna_pattern["frequency"],
                'along_track': antenna_pattern["along_track"],
                'across_track': antenna_pattern["across_track"],
                'radiation_pattern_dB': antenna_pattern["radiation_pattern_dB"],
            }

        self.frequencies = self._get_frequency_from_band()
        # The reachable steering range is exactly the generated grid extent, per axis.
        self.max_across_track_steering_deg = max(abs(key[0]) for key in self.antenna_lookuptable)
        self.max_along_track_steering_deg = max(abs(key[1]) for key in self.antenna_lookuptable)
        self._build_steering_grid()
        self._validate_boresight_steering_patterns()

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

    def _validate_boresight_steering_patterns(self) -> None:
        """
        =========================================================================================================
        Validate that every configured antenna frequency has a boresight (across_track=0, along_track=0) pattern.

        The generated steering grid must include the true zero-steering command so nadir gain and footprint
        interpolation have a physical reference point instead of being inferred only from tilted beams.
        =========================================================================================================
        """
        missing = []
        for frequency_mhz in self.frequencies:
            has_boresight_pattern = any(
                math.isclose(key[2], frequency_mhz, abs_tol=1e-6)
                and math.isclose(key[0], 0.0, abs_tol=1e-6)
                and math.isclose(key[1], 0.0, abs_tol=1e-6)
                for key in self.antenna_lookuptable
            )
            if not has_boresight_pattern:
                missing.append(frequency_mhz)

        if missing:
            missing_str = ", ".join(f"{frequency:.2f} MHz" for frequency in missing)
            raise ValueError(
                "Missing boresight antenna pattern (across_track=0, along_track=0) for footprint calculation "
                f"at {missing_str}. Include 0 in both across_track_steering and along_track_steering."
            )

    def _build_steering_grid(self) -> None:
        """
        =========================================================================================================
        Pre-compute, per available frequency, sorted across_track/along_track steering grids and a (across, along) -> key map.
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

        # Steering-command-centred contour ring; already at footprint_resolution, no further resampling needed.
        contur_vectors = self._find_contour_vectors(
            rad_pattern_spherical_dB,
            antenna_pattern.get("across_track", 0.0),
            antenna_pattern.get("along_track", 0.0),
        )

        debug = False
        if debug:
            plotting_object = PlotRadiationPattern(spherical_coordinates=(rad_pattern_spherical_lin, theta_mesh, phi_mesh), arrow=(0.0, 0.0))
            plotting_object.add_vector_bundle_to_plot(contur_vectors)
            plotting_object.show()

        return directivity, directivity_dB, contur_vectors

    def _get_antenna_footprint(self, refl_dir_antenna_frame, frequency):
        """
        =========================================================================================================
        Get the steering-interpolated antenna footprint contour for the given reflection direction.

        INPUTS:
            refl_dir_antenna_frame: [np.ndarray, shape (3,)]   Reflection direction in antenna frame
            frequency:              [float]                    Frequency [MHz] (used for lookup)

        OUTPUT:
            footprint_vectors: [np.ndarray, shape (N, 3)]   Interpolated steered footprint contour vectors
        =========================================================================================================
        """
        refl_dir_antenna_frame = np.asarray(refl_dir_antenna_frame, dtype=float)
        if refl_dir_antenna_frame.shape != (3,):
            raise ValueError("Input 'refl_dir_antenna_frame' must be a 3-element vector.")
        if not np.isfinite(refl_dir_antenna_frame).all():
            raise ValueError("Input 'refl_dir_antenna_frame' must contain finite values.")

        direction_norm = np.linalg.norm(refl_dir_antenna_frame)
        if direction_norm <= 0.0:
            raise ValueError("Input 'refl_dir_antenna_frame' must be non-zero.")
        refl_dir_antenna_frame = refl_dir_antenna_frame / direction_norm

        steering_angles = self.resolve_steering_angles_for_direction(refl_dir_antenna_frame)
        if steering_angles is None:
            return np.empty((0, 3), dtype=float)

        steering_across_track_deg, steering_along_track_deg = steering_angles
        corners = self._steering_bracket_keys_and_weights(frequency, steering_across_track_deg, steering_along_track_deg)

        interpolated_vectors = None
        expected_shape = None
        for key, weight in corners:
            if weight <= 0.0:
                continue
            contour_vectors = np.asarray(self.antenna_lookuptable[key]["vector_bundle"], dtype=float)
            if expected_shape is None:
                expected_shape = contour_vectors.shape
                interpolated_vectors = np.zeros(expected_shape, dtype=float)
            elif contour_vectors.shape != expected_shape:
                raise ValueError("All steered footprint vector bundles must have the same shape.")
            interpolated_vectors += weight * contour_vectors

        if interpolated_vectors is None:
            return np.empty((0, 3), dtype=float)

        vector_norms = np.linalg.norm(interpolated_vectors, axis=1, keepdims=True)
        valid_vectors = np.isfinite(interpolated_vectors).all(axis=1) & (vector_norms[:, 0] > 0.0)
        if not valid_vectors.all():
            raise ValueError("Interpolated footprint vector bundle contains invalid or zero-length vectors.")
        return interpolated_vectors / vector_norms

    @staticmethod
    def _bracket_linear(sorted_vals: np.ndarray, target: float) -> tuple[int, int, float]:
        """
        Bracket a target value within a sorted 1-D grid axis, clamping at the edges.

        Returns (lo_idx, hi_idx, weight) where the interpolated value is
        (1 - weight) * sorted_vals[lo_idx] + weight * sorted_vals[hi_idx]. Targets outside the grid
        clamp to the nearest edge (weight 0).
        """
        n = sorted_vals.size
        if n == 1:
            return 0, 0, 0.0
        clamped = float(np.clip(target, sorted_vals[0], sorted_vals[-1]))
        if clamped <= sorted_vals[0]:
            return 0, 0, 0.0
        if clamped >= sorted_vals[-1]:
            return n - 1, n - 1, 0.0
        hi_idx = int(np.searchsorted(sorted_vals, clamped, side="right"))
        lo_idx = hi_idx - 1
        weight = (clamped - sorted_vals[lo_idx]) / (sorted_vals[hi_idx] - sorted_vals[lo_idx])
        return lo_idx, hi_idx, weight

    def _steering_bracket_keys_and_weights(self, frequency_mhz: float, az_target: float, el_target: float) -> list:
        """
        =========================================================================================================
        Return the four steering pattern keys + bilinear weights bracketing (az_target, el_target) for the
        closest available frequency. Both axes are plain bounded linear axes (two-plane scan), clamped at the
        grid edges. Some weights may be 0 when the target sits on a grid line or outside the grid.

        INPUTS:
            frequency_mhz: [float]   Target frequency [MHz] (snapped to closest available)
            az_target:     [float]   Target steering across_track (y-axis tilt) [deg]
            el_target:     [float]   Target steering along_track (x-axis tilt) [deg]

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

        az_lo_idx, az_hi_idx, w_az = self._bracket_linear(az_sorted, az_target)
        el_lo_idx, el_hi_idx, w_el = self._bracket_linear(el_sorted, el_target)

        az_lo, az_hi = az_sorted[az_lo_idx], az_sorted[az_hi_idx]
        el_lo, el_hi = el_sorted[el_lo_idx], el_sorted[el_hi_idx]

        return [
            (key_map[(az_lo, el_lo)], (1.0 - w_az) * (1.0 - w_el)),
            (key_map[(az_hi, el_lo)], w_az * (1.0 - w_el)),
            (key_map[(az_lo, el_hi)], (1.0 - w_az) * w_el),
            (key_map[(az_hi, el_hi)], w_az * w_el),
        ]

    @staticmethod
    def _direction_to_steering_angles_deg(direction: np.ndarray) -> tuple[float, float] | None:
        """
        =========================================================================================================
        Convert a unit look direction in antenna coordinates to two-plane steering angles.

        Sine-space: across_track = asin(y-component), along_track = asin(x-component); boresight is +z.

        INPUTS:
            direction: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (across_track_deg, along_track_deg) or None if invalid / behind the antenna
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

        across_track_deg = float(np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0))))    # across_track steers the y-axis
        along_track_deg = float(np.degrees(np.arcsin(np.clip(direction[0], -1.0, 1.0))))  # along_track steers the x-axis
        return across_track_deg, along_track_deg

    def resolve_steering_angles_for_direction(self, direction_antenna_frame: np.ndarray) -> tuple[float, float] | None:
        """
        =========================================================================================================
        Resolve the automatic two-plane steering command for a look direction in antenna coordinates.

        The beam steers to point at the reflection direction; requests beyond the per-axis max steering angle
        are rejected.

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame

        OUTPUT:
            angles: [tuple | None]   (across_track_deg, along_track_deg) used to select / interpolate steered patterns,
                                     or None if the direction is invalid or outside the steering limits
        =========================================================================================================
        """
        angles = self._direction_to_steering_angles_deg(direction_antenna_frame)
        if angles is None:
            return None

        across_track_deg, along_track_deg = angles
        # 1e-9 deg tolerance so a direction exactly at the limit is not rejected by float round-off
        # (e.g. degrees(asin(sin(40 deg))) can come back as 40.0000000001).
        if self.max_across_track_steering_deg is not None and abs(across_track_deg) > self.max_across_track_steering_deg + 1e-9:
            return None
        if self.max_along_track_steering_deg is not None and abs(along_track_deg) > self.max_along_track_steering_deg + 1e-9:
            return None
        return across_track_deg, along_track_deg

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
            phi_deg:    [float]                                Across-track angle [deg], any (wraps mod 360)

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
        across_track_deg: float | None = None,
        along_track_deg: float | None = None,
    ) -> float:
        """
        =========================================================================================================
        Get absolute gain [dBi] for a given look direction in antenna coordinates.

        The beam auto-steers to the look direction (two-plane, capped per axis by max_across_track_steering_deg /
        max_along_track_steering_deg). Gain is the bilinear interpolation, in linear power, of the four stored
        steered patterns bracketing the steering command; each contributes its own directivity (so scan loss
        enters through the per-direction directivity).

        INPUTS:
            direction_antenna_frame: [np.ndarray, shape (3,)]   Look direction in antenna frame
            frequency_mhz:           [float]                    Carrier frequency [MHz]
            across_track_deg:             [float | None]             Steering across_track override [deg] (optional)
            along_track_deg:           [float | None]             Steering along_track override [deg] (optional)

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

        # Pattern-sampling angles (spherical), used to read the stored pattern over (theta, phi).
        theta_deg = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
        phi_deg = float((np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0)

        if across_track_deg is None or along_track_deg is None:
            steering_angles = self.resolve_steering_angles_for_direction(direction)
            if steering_angles is None:
                return np.nan
            auto_across_track_deg, auto_along_track_deg = steering_angles
            steering_across_track_deg = auto_across_track_deg if across_track_deg is None else float(across_track_deg)
            steering_along_track_deg = auto_along_track_deg if along_track_deg is None else float(along_track_deg)
        else:
            steering_across_track_deg = float(across_track_deg)
            steering_along_track_deg = float(along_track_deg)

        if self.max_across_track_steering_deg is not None and abs(steering_across_track_deg) > self.max_across_track_steering_deg + 1e-9:
            return np.nan
        if self.max_along_track_steering_deg is not None and abs(steering_along_track_deg) > self.max_along_track_steering_deg + 1e-9:
            return np.nan

        # Bilinear interpolation of absolute gain across the 4 steering patterns bracketing
        # (steering_across_track_deg, steering_along_track_deg), evaluated in linear power.
        corners = self._steering_bracket_keys_and_weights(frequency_mhz, steering_across_track_deg, steering_along_track_deg)
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

        The power integral covers the front hemisphere (theta <= 90 deg) only: the array is mounted on
        the Earth-facing spacecraft panel over a ground plane, so the rear half-space is shielded and the
        isotropic-element array factor's mirror back lobe (|AF| depends on theta only via sin(theta))
        carries no physical power.

        INPUTS:
            rad_pattern_lin_spherical: [np.ndarray]   Linear-scale spherical radiation pattern
            theta_rad:                 [np.ndarray]   Polar angle samples [rad]
            phi_rad:                   [np.ndarray]   Across-track angle samples [rad]
            theta_mesh:                [np.ndarray]   Meshgrid of theta values

        OUTPUT:
            result: [tuple]   (directivity, directivity_dB)
        =========================================================================================================
        """
        # Calculate differential solid angle elements
        dtheta = np.gradient(theta_rad)
        dphi = np.gradient(phi_rad)
        dtheta_mesh, dphi_mesh = np.meshgrid(dtheta, dphi, indexing='ij')

        # Integrate: int int F(theta, phi) sin(theta) dtheta dphi over the shielded-panel front hemisphere
        front_hemisphere = theta_mesh <= np.pi / 2.0                # boresight half-space, theta <= 90 deg
        integrand = rad_pattern_lin_spherical * np.sin(theta_mesh) * dtheta_mesh * dphi_mesh
        total_power = np.sum(integrand[front_hemisphere])

        if total_power <= 0:
            raise ValueError("Antenna pattern integrates to non-positive total power; cannot compute directivity.")

        # Directivity = 4*pi / integral
        directivity = 4 * np.pi / total_power
        directivity_dB = 10 * np.log10(directivity)

        return directivity, directivity_dB

    def _find_contour_vectors(self, rad_pattern_spherical_dB, steering_across_track_deg, steering_along_track_deg):
        """
        =========================================================================================================
        Build the -|footprint_dB_contour| ring in a steering-command-centred local frame.

        The commanded front-lobe direction defines the axis n_peak. We sweep an off-peak across_track psi around
        n_peak in a tangent basis whose psi=0 direction is the projection of antenna +x onto the
        tangent plane, and for each psi walk the off-peak angle gamma outward from 0 until the
        pattern (bilinearly sampled in antenna-frame theta/phi) drops below footprint_dB_contour.

        For unsteered patterns (n_peak = +z) this collapses to the antenna-frame (theta, phi) walk;
        for steered patterns it yields a ring centred on the steered beam axis instead of the
        antenna-frame fan the previous per-phi-slice walk produced. The global pattern maximum is
        deliberately ignored because the pure array factor can have an equally strong back lobe.

        INPUTS:
            rad_pattern_spherical_dB: [np.ndarray]   Radiation pattern [dB], shape (n_theta, n_phi)
            steering_across_track_deg:     [float]        Commanded y-axis steering angle [deg]
            steering_along_track_deg:   [float]        Commanded x-axis steering angle [deg]

        OUTPUT:
            contour_vectors: [np.ndarray, shape (footprint_resolution, 3)]   Main-lobe-centred ring vectors
        =========================================================================================================
        """
        n_theta_pat = rad_pattern_spherical_dB.shape[0]

        sin_az = np.sin(np.radians(float(steering_across_track_deg)))
        sin_el = np.sin(np.radians(float(steering_along_track_deg)))
        z_squared = 1.0 - sin_az**2 - sin_el**2
        if z_squared <= 0.0:
            raise ValueError("Steering command points outside the antenna front hemisphere.")

        n_peak = np.array([
            sin_el,   # along_track steers the x-axis
            sin_az,   # across_track steers the y-axis
            np.sqrt(z_squared),
        ])

        # Tangent basis at n_peak: psi=0 is the projection of antenna +x onto the tangent plane,
        # so for n_peak = +z (unsteered) psi reduces to antenna-frame phi.
        e_x = np.array([1.0, 0.0, 0.0])
        tangent = e_x - np.dot(e_x, n_peak) * n_peak
        if np.linalg.norm(tangent) < 1e-9:
            # n_peak parallel to +x (would only happen at theta_p=90, phi_p=0); use +y as the reference.
            e_y = np.array([0.0, 1.0, 0.0])
            tangent = e_y - np.dot(e_y, n_peak) * n_peak
        e_psi0 = tangent / np.linalg.norm(tangent)
        e_psi90 = np.cross(n_peak, e_psi0)

        # Off-peak radial walk uses the same theta-resolution as the stored pattern.
        gamma_grid = np.linspace(0.0, np.pi, n_theta_pat)
        n_out = self.footprint_resolution
        psi_grid = np.linspace(0.0, 2.0 * np.pi, n_out, endpoint=False)
        theta_peak_deg = float(np.degrees(np.arccos(np.clip(n_peak[2], -1.0, 1.0))))
        phi_peak_deg = float((np.degrees(np.arctan2(n_peak[1], n_peak[0])) + 360.0) % 360.0)
        peak_lin = self._bilinear_pattern_sample_linear(rad_pattern_spherical_dB, theta_peak_deg, phi_peak_deg)
        peak_dB = 10.0 * np.log10(max(peak_lin, 1e-30))

        contour_vectors = np.zeros((n_out, 3), dtype=float)
        for j, psi in enumerate(psi_grid):
            u_psi = np.cos(psi) * e_psi0 + np.sin(psi) * e_psi90

            prev_dB = peak_dB
            prev_gamma = 0.0
            gamma_c = gamma_grid[-1]  # default if no crossing (would mean pattern stays above threshold)
            for gamma in gamma_grid[1:]:
                d = np.cos(gamma) * n_peak + np.sin(gamma) * u_psi
                theta_d_deg = float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))
                phi_d_deg = float((np.degrees(np.arctan2(d[1], d[0])) + 360.0) % 360.0)
                val_lin = self._bilinear_pattern_sample_linear(rad_pattern_spherical_dB, theta_d_deg, phi_d_deg)
                val_dB = 10.0 * np.log10(max(val_lin, 1e-30))

                if val_dB <= self.footprint_dB_contour:
                    if prev_dB != val_dB:
                        t = (self.footprint_dB_contour - prev_dB) / (val_dB - prev_dB)
                        gamma_c = prev_gamma + t * (gamma - prev_gamma)
                    else:
                        gamma_c = gamma
                    break
                prev_dB = val_dB
                prev_gamma = gamma

            contour_vectors[j] = np.cos(gamma_c) * n_peak + np.sin(gamma_c) * u_psi

        return contour_vectors

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
