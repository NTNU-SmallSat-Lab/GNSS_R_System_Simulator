import json
from pathlib import Path
import warnings

import numpy as np

from object_definitions.Constants_def import Constants

# Metadata file written next to the generated JSONs. Used to invalidate the
# cache when any pattern-affecting parameter changes (analogous to the
# ShipTrajectory metadata JSON used to skip route replanning).
METADATA_FILENAME = "antenna_pattern_metadata.json"
PATTERN_FILE_PATTERN = "array_factor_*.json"


class ArrayFactor:
    def __init__(self, elements_along_x, elements_along_y, freq_hz, d_x, d_y, beta_x, beta_y):
        """Phased-array factor: psi = k d sin(theta) [cos|sin](phi) + beta, summed over elements."""
        wavelength_m = Constants.SPEED_OF_LIGHT / freq_hz

        self.elements_along_x = elements_along_x
        self.elements_along_y = elements_along_y
        self.k = 2 * np.pi / wavelength_m
        self.d_x = d_x
        self.d_y = d_y
        self.beta_x = beta_x
        self.beta_y = beta_y

    def calculate_array_factor(self, theta, phi):
        """Compute the (theta x phi) array factor pattern in dB, normalized to a 0 dB peak."""
        psi_x = self.k * self.d_x * np.sin(theta) * np.cos(phi) + self.beta_x
        psi_y = self.k * self.d_y * np.sin(theta) * np.sin(phi) + self.beta_y

        x_elems = np.arange(self.elements_along_x).reshape(-1, 1, 1)
        y_elems = np.arange(self.elements_along_y).reshape(-1, 1, 1)

        af = np.sum(np.exp(1j * x_elems * psi_x), axis=0) * np.sum(np.exp(1j * y_elems * psi_y), axis=0)

        epsilon = 1e-12  # avoid log(0) at array-factor nulls
        af_dB = 20 * np.log10(np.abs(af) + epsilon)
        return af_dB - af_dB.max()


class AntennaPatternGenerator:
    """Generate (or reuse cached) phased-array radiation-pattern JSON files for the receiver antenna."""

    def __init__(
        self,
        elements_along_x,
        elements_along_y,
        d_x,
        d_y,
        freqs,
        azimuth_steering,
        elevation_steering,
        output_folder,
        theta_resolution=180,
        phi_resolution=360,
    ):
        """
        AntennaPatternGenerator (two-plane / sine-space steering):

        INPUTS:
            elements_along_x:   [int]         Number of array elements along x
            elements_along_y:   [int]         Number of array elements along y
            d_x:                [float]       Element spacing along x [m]
            d_y:                [float]       Element spacing along y [m]
            freqs:              [list[float]] Frequencies to simulate [MHz]
            azimuth_steering:   [list]        [start_deg, end_deg, n_steps] beam tilt sweep along the array x-axis
            elevation_steering: [list]        [start_deg, end_deg, n_steps] beam tilt sweep along the array y-axis
            output_folder:      [str]         Path to write generated JSON files
            theta_resolution:   [int]         Number of theta samples in each pattern (0..180 deg), default 180
            phi_resolution:     [int]         Number of phi samples in each pattern (0..360 deg), default 360
        """
        self.elements_x = int(elements_along_x)
        self.elements_y = int(elements_along_y)
        self.d_x = float(d_x)
        self.d_y = float(d_y)
        self.freqs = [float(f) for f in freqs]
        self.azimuth_steering = list(azimuth_steering)
        self.elevation_steering = list(elevation_steering)
        self.output_folder = Path(output_folder)
        self.theta_resolution = int(theta_resolution)
        self.phi_resolution = int(phi_resolution)

        # Two-plane sine-space constraint: the steered beam direction is
        # (sin az, sin el, sqrt(1 - sin^2 az - sin^2 el)), so the worst-case corner of the
        # steering rectangle must stay inside the unit disk or the beam points below the horizon.
        max_az_rad = np.deg2rad(max(abs(self.azimuth_steering[0]), abs(self.azimuth_steering[1])))
        max_el_rad = np.deg2rad(max(abs(self.elevation_steering[0]), abs(self.elevation_steering[1])))
        if np.sin(max_az_rad) ** 2 + np.sin(max_el_rad) ** 2 > 1.0 + 1e-9:
            raise ValueError(
                "Steering sweep corner is unreachable: sin^2(max_azimuth) + sin^2(max_elevation) > 1. "
                "Reduce azimuth_steering / elevation_steering extents."
            )

    def _steering_sweeps_deg(self):
        """Return (azimuth_sweep_deg, elevation_sweep_deg) arrays from the [start, end, n_steps] specs."""
        azimuth_sweep = np.linspace(self.azimuth_steering[0], self.azimuth_steering[1], self.azimuth_steering[2])
        elevation_sweep = np.linspace(self.elevation_steering[0], self.elevation_steering[1], self.elevation_steering[2])
        return azimuth_sweep, elevation_sweep

    # Bump this whenever the generator's output convention changes in a way that affects how the
    # consumer (Rx_antenna_def.py) must sample the patterns. Each bump invalidates every cached
    # antenna_pattern_metadata.json on disk and forces a full regenerate.
    #   v1: initial release.
    #   v2: azimuth grid switched from endpoint-inclusive [0..360] to endpoint-exclusive [0..360),
    #       matching the bilinear sampler convention in _bilinear_pattern_sample_linear.
    #   v3: fixed-nadir pattern generation. Generator emits ONE unsteered (az=0, el=0)
    #       pattern per frequency, and RxAntenna samples that JSON pattern directly.
    #   v4: pattern coordinate names renamed elevation/azimuth -> theta/phi. Signature dict
    #       keys changed, invalidating every existing cached metadata file.
    #   v5: steerable antenna restored. Generator emits one steered pattern per
    #       (freq, steering-azimuth, steering-elevation) grid point; RxAntenna bilinearly
    #       interpolates gain across the four nearest grid steering directions.
    #   v6: two-plane (sine-space) steering. azimuth/elevation are signed beam-tilt angles
    #       along the array x/y axes (beta_x=-k dx sin az, beta_y=-k dy sin el); both grids are
    #       plain bounded linear axes (no circular azimuth). Signature keys changed.
    _GENERATOR_FORMAT_VERSION = 6

    def _signature(self):
        """Dict of every parameter that affects the generated JSONs; used as a cache key."""
        return {
            "generator_format_version": self._GENERATOR_FORMAT_VERSION,
            "elements_along_x": self.elements_x,
            "elements_along_y": self.elements_y,
            "d_x": self.d_x,
            "d_y": self.d_y,
            "freqs": sorted(self.freqs),
            "azimuth_steering": self.azimuth_steering,
            "elevation_steering": self.elevation_steering,
            "theta_resolution": self.theta_resolution,
            "phi_resolution": self.phi_resolution,
        }

    def _metadata_path(self):
        """Path to the metadata JSON written alongside the generated patterns."""
        return self.output_folder / METADATA_FILENAME

    def _expected_filenames(self):
        """Yield the filename for every (frequency, steering-azimuth, steering-elevation) generate() will produce."""
        azimuth_sweep, elevation_sweep = self._steering_sweeps_deg()
        for freq in self.freqs:
            for az_deg in azimuth_sweep:
                for el_deg in elevation_sweep:
                    yield f"array_factor_{freq}MHz_{az_deg:.2f}az_{el_deg:.2f}el.json"

    def _all_files_exist(self):
        """True if every expected pattern JSON is present on disk."""
        filenames = list(self._expected_filenames())
        if not filenames:
            raise ValueError(
                "AntennaPatternGenerator: no output files expected — check that freqs, "
                "azimuth_steering, and elevation_steering are set correctly."
            )
        existing = sum(1 for name in filenames if (self.output_folder / name).exists())
        print(f"[AntennaPatternGenerator] {existing}/{len(filenames)} pattern files found in '{self.output_folder.resolve()}'.")
        return existing == len(filenames)

    def _warn_unexpected_pattern_files(self):
        """Warn when generator-owned output_folder contains array-factor files outside the current sweep."""
        expected = set(self._expected_filenames())
        existing = {path.name for path in self.output_folder.glob(PATTERN_FILE_PATTERN)}
        unexpected = sorted(existing - expected)
        if not unexpected:
            return

        preview = ", ".join(unexpected[:5])
        suffix = "" if len(unexpected) <= 5 else f", ... ({len(unexpected)} total)"
        warnings.warn(
            "Antenna pattern folder contains unexpected array_factor_*.json files. "
            "The pattern folder is generator-owned; remove stale files before relying on cached patterns. "
            f"Unexpected files: {preview}{suffix}",
            stacklevel=2,
        )

    def _load_cached_signature(self):
        """Return the signature recorded by a prior generate() run, or None if missing/unreadable."""
        path = self._metadata_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[AntennaPatternGenerator] Could not read cached metadata ({e}); will regenerate.")
            return None

    def _purge_existing_patterns(self):
        """Remove every array_factor_*.json + the metadata file from output_folder."""
        if not self.output_folder.exists():
            return
        for stale in self.output_folder.glob(PATTERN_FILE_PATTERN):
            try:
                stale.unlink()
            except OSError as e:
                print(f"[AntennaPatternGenerator] Could not delete '{stale}': {e}")
        meta = self._metadata_path()
        if meta.exists():
            try:
                meta.unlink()
            except OSError as e:
                print(f"[AntennaPatternGenerator] Could not delete '{meta}': {e}")

    def _write_metadata(self):
        """Persist the current signature so the next run can decide whether to skip generation."""
        with open(self._metadata_path(), "w") as f:
            json.dump(self._signature(), f, indent=2)

    def generate(self):
        """Generate (or reuse cached) per-(freq, steering-az, steering-el) JSON pattern files based on a metadata signature."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

        current_signature = self._signature()
        cached_signature = self._load_cached_signature()

        if cached_signature == current_signature and self._all_files_exist():
            self._warn_unexpected_pattern_files()
            print("[AntennaPatternGenerator] Settings unchanged and all pattern files present — skipping generation.")
            return

        if cached_signature is None:
            print("[AntennaPatternGenerator] No cached metadata; generating pattern files.")
        elif cached_signature != current_signature:
            print("[AntennaPatternGenerator] Settings changed since last run; regenerating pattern files.")
        else:
            print("[AntennaPatternGenerator] Some pattern files are missing; regenerating.")

        # Wipe any partial / stale artefacts so the consumer-side glob sees only files
        # that match the current sweep.
        self._purge_existing_patterns()

        # Pattern grid (theta = polar 0..180 deg, phi = azimuthal 0..360 deg)
        theta = np.linspace(0, 180, self.theta_resolution).reshape(-1, 1) * np.pi / 180
        phi = np.linspace(0, 360, self.phi_resolution, endpoint=False).reshape(1, -1) * np.pi / 180

        # Steering sweep (two-plane): azimuth = tilt along array x-axis, elevation = tilt along y-axis.
        azimuth_sweep_deg, elevation_sweep_deg = self._steering_sweeps_deg()
        azimuth_sweep = azimuth_sweep_deg * np.pi / 180
        elevation_sweep = elevation_sweep_deg * np.pi / 180

        # Write one steered pattern per (freq, steering-az, steering-el). The progressive phase shift
        # beta steers the array peak to direction (sin az, sin el, .); RxAntenna bilinearly
        # interpolates gain between neighbouring grid steering directions.
        for freq in self.freqs:
            freq_hz = freq * 1e6
            k = 2 * np.pi / (Constants.SPEED_OF_LIGHT / freq_hz)

            for az_direction in azimuth_sweep:
                for el_direction in elevation_sweep:
                    beta_x = -k * self.d_x * np.sin(az_direction)
                    beta_y = -k * self.d_y * np.sin(el_direction)

                    az_deg = az_direction * 180 / np.pi
                    el_deg = el_direction * 180 / np.pi

                    af = ArrayFactor(self.elements_x, self.elements_y, freq_hz, self.d_x, self.d_y, beta_x, beta_y)
                    pattern_db = af.calculate_array_factor(theta, phi)

                    out_dict = {
                        "frequency": freq,
                        "azimuth_pointing": az_deg,
                        "elevation_pointing": el_deg,
                        "data": pattern_db.tolist(),
                    }

                    filename = f"array_factor_{freq}MHz_{az_deg:.2f}az_{el_deg:.2f}el.json"
                    with open(self.output_folder / filename, "w") as f:
                        # Compact dump — the data array is large (~64k floats); pretty-printing is wasteful.
                        json.dump(out_dict, f)

        self._write_metadata()
        self._warn_unexpected_pattern_files()
        print(f"[AntennaPatternGenerator] Done. Files written to '{self.output_folder.resolve()}'.")
