import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from object_definitions.Constants_def import Constants

BORESIGHT_PLOT_SUBDIR = "antennaRadiationPattern"

# Metadata file written next to the generated JSONs. Used to invalidate the
# cache when any pattern-affecting parameter changes (analogous to the
# ShipTrajectory metadata JSON used to skip route replanning).
METADATA_FILENAME = "antenna_pattern_metadata.json"
PATTERN_FILE_PATTERN = "array_factor_*.json"


class ArrayFactor:
    def __init__(self, elements_along_x, elements_along_y, freq_hz, d_x, d_y, beta_x, beta_y):
        """
        Phased-array factor: psi = k d sin(theta) [cos|sin](phi) + beta, summed over elements.
        """
        wavelength_m = Constants.SPEED_OF_LIGHT / freq_hz

        self.elements_along_x = elements_along_x
        self.elements_along_y = elements_along_y
        self.k = 2 * np.pi / wavelength_m
        self.d_x = d_x
        self.d_y = d_y
        self.beta_x = beta_x
        self.beta_y = beta_y

    def calculate_array_factor(self, theta, phi):
        """
        Compute the (theta x phi) array factor pattern in dB, normalized to a 0 dB peak.
        """
        psi_x = self.k * self.d_x * np.sin(theta) * np.cos(phi) + self.beta_x
        psi_y = self.k * self.d_y * np.sin(theta) * np.sin(phi) + self.beta_y

        x_elems = np.arange(self.elements_along_x).reshape(-1, 1, 1)
        y_elems = np.arange(self.elements_along_y).reshape(-1, 1, 1)

        af = np.sum(np.exp(1j * x_elems * psi_x), axis=0) * np.sum(np.exp(1j * y_elems * psi_y), axis=0)

        epsilon = 1e-12  # avoid log(0) at array-factor nulls
        af_dB = 20 * np.log10(np.abs(af) + epsilon)
        return af_dB - af_dB.max()


class AntennaPatternGenerator:
    """Generate (or reuse cached) phased-array radiation-pattern files for the receiver antenna.

    Uses two-plane (sine-space) steering: elevation tilts the beam about the array y-axis (peak moves along
    x^a, the flight direction) and azimuth about the x-axis (peak moves along y^a), so the beam direction is
    ``(sin el, sin az, sqrt(1 - sin^2 az - sin^2 el))`` and the steering rectangle must stay inside the unit
    disk (``sin^2 az + sin^2 el <= 1``).

    Parameters
    ----------
    elements_along_x, elements_along_y : int
        Number of array elements per axis.
    d_x, d_y : float
        Element spacing per axis [m].
    freqs : list of float
        Frequencies to simulate [MHz].
    azimuth_steering, elevation_steering : list
        ``[start_deg, end_deg, n_steps]`` beam-tilt sweep: azimuth tilts the peak along y^a (rotation about
        the x-axis), elevation along x^a (rotation about the y-axis).
    output_folder : str
        Where to write the generated pattern files.
    theta_resolution : int, optional
        Polar-angle samples over 0..180 deg off boresight (default 180).
    phi_resolution : int, optional
        Azimuth samples over 0..360 deg around boresight (default 360).
    """

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
        """Store the array geometry and steering sweeps; check the sine-space reachability constraint."""
        def _integer_parameter(name, value, minimum):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}") from exc
            if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
            return int(numeric)

        self.elements_x = _integer_parameter("elements_along_x", elements_along_x, 1)
        self.elements_y = _integer_parameter("elements_along_y", elements_along_y, 1)
        self.d_x = float(d_x)
        self.d_y = float(d_y)
        self.freqs = [float(f) for f in freqs]
        self.azimuth_steering = list(azimuth_steering)
        self.elevation_steering = list(elevation_steering)
        self.output_folder = Path(output_folder)
        self.theta_resolution = _integer_parameter("theta_resolution", theta_resolution, 2)
        self.phi_resolution = _integer_parameter("phi_resolution", phi_resolution, 3)

        if not (np.isfinite(self.d_x) and self.d_x > 0.0 and np.isfinite(self.d_y) and self.d_y > 0.0):
            raise ValueError("d_x and d_y must be finite and > 0 m")
        if not self.freqs or not np.isfinite(self.freqs).all() or any(f <= 0.0 for f in self.freqs):
            raise ValueError("freqs must contain finite, positive frequencies in MHz")
        for axis_name, spec in (
            ("azimuth_steering", self.azimuth_steering),
            ("elevation_steering", self.elevation_steering),
        ):
            if len(spec) != 3:
                raise ValueError(f"{axis_name} must be [start_deg, end_deg, n_steps]")
            try:
                endpoints = np.asarray(spec[:2], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{axis_name} endpoints must be finite and n_steps must be an integer >= 1") from exc
            if not np.isfinite(endpoints).all():
                raise ValueError(f"{axis_name} endpoints must be finite and n_steps must be >= 1")
            n_steps = _integer_parameter(f"{axis_name} n_steps", spec[2], 1)
            spec[:] = [float(endpoints[0]), float(endpoints[1]), n_steps]
            if max(abs(spec[0]), abs(spec[1])) >= 90.0:
                raise ValueError(f"{axis_name} must stay strictly inside the front hemisphere (-90, 90) deg")

        # Two-plane sine-space constraint: the steered beam direction is
        # (sin el, sin az, sqrt(1 - sin^2 az - sin^2 el)), so the worst-case corner of the
        # steering rectangle must stay inside the unit disk or the beam points below the horizon.
        max_az_rad = np.deg2rad(max(abs(self.azimuth_steering[0]), abs(self.azimuth_steering[1])))
        max_el_rad = np.deg2rad(max(abs(self.elevation_steering[0]), abs(self.elevation_steering[1])))
        if np.sin(max_az_rad) ** 2 + np.sin(max_el_rad) ** 2 > 1.0 + 1e-9:
            raise ValueError(
                "Steering sweep corner is unreachable: sin^2(max_azimuth) + sin^2(max_elevation) > 1. "
                "Reduce azimuth_steering / elevation_steering extents."
            )

    def _steering_sweeps_deg(self):
        """
        Return (azimuth_sweep_deg, elevation_sweep_deg) arrays from the [start, end, n_steps] specs.

        Each sweep is guaranteed to contain boresight (0 deg); see _axis_sweep_with_boresight.
        """
        azimuth_sweep = self._axis_sweep_with_boresight("azimuth", self.azimuth_steering)
        elevation_sweep = self._axis_sweep_with_boresight("elevation", self.elevation_steering)
        return azimuth_sweep, elevation_sweep

    @staticmethod
    def _axis_sweep_with_boresight(axis_name, spec):
        """
        Build a steering sweep [deg] guaranteed to include boresight (0 deg).

        linspace(start, end, n_steps) lands exactly on 0 only for particular (range, count) pairs --
        for a symmetric range that means an ODD count. The boresight pattern is required downstream
        (nadir gain + footprint reference), so when 0 is missing this raises the count by one (which
        fixes the common symmetric-even case while keeping the grid uniform) and, failing that, inserts
        0 explicitly. Either repair emits a warning so the configuration can be corrected.
        """
        start, end, n_steps = spec[0], spec[1], int(spec[2])
        sweep = np.linspace(start, end, n_steps)
        if np.any(np.isclose(sweep, 0.0, atol=1e-9)):
            return sweep

        bumped = np.linspace(start, end, n_steps + 1)
        if np.any(np.isclose(bumped, 0.0, atol=1e-9)):
            warnings.warn(
                f"[ANTENNA PATTERN] {axis_name}_steering = [{start}, {end}, {n_steps}] skips boresight "
                f"(0 deg): a symmetric range needs an odd n_steps. Raising n_steps to {n_steps + 1} so the "
                f"steering grid includes boresight.",
                stacklevel=2,
            )
            return bumped

        warnings.warn(
            f"[ANTENNA PATTERN] {axis_name}_steering = [{start}, {end}, {n_steps}] does not include "
            f"boresight (0 deg); inserting it. Use a range that lands on 0 (e.g. symmetric with odd "
            f"n_steps) to avoid a non-uniform steering grid.",
            stacklevel=2,
        )
        return np.unique(np.concatenate([sweep, [0.0]]))

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
    #   v7: azimuth/elevation axis association swapped. Elevation now steers the array x-axis
    #       (beta_x=-k dx sin el, peak along x = flight direction); azimuth steers the y-axis
    #       (beta_y=-k dy sin az). Beam direction (sin el, sin az, .). Invalidates cached patterns.
    _GENERATOR_FORMAT_VERSION = 7

    def _signature(self):
        """
        Dict of every parameter that affects the generated JSONs; used as a cache key.
        """
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
        """
        Path to the metadata JSON written alongside the generated patterns.
        """
        return self.output_folder / METADATA_FILENAME

    def _expected_filenames(self):
        """
        Yield the filename for every (frequency, steering-azimuth, steering-elevation) generate() will produce.
        """
        azimuth_sweep, elevation_sweep = self._steering_sweeps_deg()
        for freq in self.freqs:
            for az_deg in azimuth_sweep:
                for el_deg in elevation_sweep:
                    yield f"array_factor_{freq}MHz_{az_deg:.2f}az_{el_deg:.2f}el.json"

    def _all_files_exist(self):
        """
        True if every expected pattern JSON is present on disk.
        """
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
        """
        Warn when generator-owned output_folder contains array-factor files outside the current sweep.
        """
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
        """
        Return the signature recorded by a prior generate() run, or None if missing/unreadable.
        """
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
        """
        Remove every array_factor_*.json + the metadata file from output_folder.
        """
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
        """
        Persist the current signature so the next run can decide whether to skip generation.
        """
        with open(self._metadata_path(), "w") as f:
            json.dump(self._signature(), f, indent=2)

    def generate(self):
        """
        Generate (or reuse cached) per-(freq, steering-az, steering-el) JSON pattern files based on a metadata signature.
        """
        self.output_folder.mkdir(parents=True, exist_ok=True) # ensure the output folder exists before checking for cached files

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
        theta = np.linspace(0, 180, self.theta_resolution).reshape(-1, 1) * Constants.DEG2RAD
        phi = np.linspace(0, 360, self.phi_resolution, endpoint=False).reshape(1, -1) * Constants.DEG2RAD

        # Steering sweep (two-plane): elevation tilts along array x; azimuth tilts along array y.
        azimuth_sweep_deg, elevation_sweep_deg = self._steering_sweeps_deg()
        azimuth_sweep = azimuth_sweep_deg * Constants.DEG2RAD
        elevation_sweep = elevation_sweep_deg * Constants.DEG2RAD

        # Write one steered pattern per (freq, steering-az, steering-el). The progressive phase shift
        # beta steers the array peak to direction (sin el, sin az, .); RxAntenna bilinearly
        # interpolates gain between neighbouring grid steering directions.
        for freq in self.freqs:
            freq_hz = freq * Constants.MHZ_TO_HZ #[Hz]
            lambda_m = Constants.SPEED_OF_LIGHT / freq_hz # [m] wavelength
            k = 2 * np.pi / lambda_m # [rad/m] Angular wavenumber

            for az_direction in azimuth_sweep:
                for el_direction in elevation_sweep:
                    beta_x = -k * self.d_x * np.sin(el_direction)  # elevation steers the x-axis (rotation about y)
                    beta_y = -k * self.d_y * np.sin(az_direction)  # azimuth steers the y-axis (rotation about x)

                    az_deg = az_direction * Constants.RAD2DEG
                    el_deg = el_direction * Constants.RAD2DEG

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
        self._plot_boresight_pattern()
        print(f"[AntennaPatternGenerator] Done. Files written to '{self.output_folder.resolve()}'.")

    # Floor below which the 3D dB surface collapses to the origin. Anything below this is
    # considered "no signal" for visualization; values above scale linearly between floor and peak.
    _PLOT_DB_FLOOR = -40.0

    def _plot_boresight_pattern(self):
        """
        Render the boresight (az=0, el=0) array factor as a dB radiation pattern plot per frequency.

        For each configured frequency, writes one PNG to OUTPUT/antennaRadiationPattern/:
          - <freq>MHz.png — 2D polar cut in dB (phi=0 and phi=90) + 3D dB surface (clipped at floor)
        Power and voltage are numerically identical in dB (10*log10(|AF|^2) == 20*log10(|AF|)),
        so a single dB figure covers both. Called only when patterns are actually regenerated.
        """
        plot_dir = Path(Constants.output_folder_path) / BORESIGHT_PLOT_SUBDIR
        plot_dir.mkdir(parents=True, exist_ok=True)

        theta_1d = np.linspace(0.0, np.pi, self.theta_resolution)
        phi_1d = np.linspace(0.0, 2.0 * np.pi, self.phi_resolution, endpoint=False)
        theta_mesh, phi_mesh = np.meshgrid(theta_1d, phi_1d, indexing="ij")
        phi0_idx = 0
        phi90_idx = self.phi_resolution // 4

        for freq in self.freqs:
            freq_hz = freq * 1e6
            af = ArrayFactor(self.elements_x, self.elements_y, freq_hz, self.d_x, self.d_y, 0.0, 0.0)
            pattern_dB = af.calculate_array_factor(theta_1d.reshape(-1, 1), phi_1d.reshape(1, -1))

            self._render_pattern_figure(
                path=plot_dir / f"{freq}MHz.png",
                title=f"Boresight radiation pattern @ {freq} MHz",
                theta_1d=theta_1d,
                theta_mesh=theta_mesh,
                phi_mesh=phi_mesh,
                pattern_dB=pattern_dB,
                floor_dB=self._PLOT_DB_FLOOR,
                phi0_idx=phi0_idx,
                phi90_idx=phi90_idx,
            )
        print(f"[AntennaPatternGenerator] Boresight radiation plots written to '{plot_dir.resolve()}'.")

    @staticmethod
    def _render_pattern_figure(*, path, title, theta_1d, theta_mesh, phi_mesh, pattern_dB, floor_dB, phi0_idx, phi90_idx):
        """
        One 2-panel figure: polar cut and 3D surface, both in dB, with a shared floor.
        """
        fig = plt.figure(figsize=(14, 6))

        # 2D polar cut in dB: phi=0 (solid), phi=90 (dashed). Boresight at top, theta sweeps clockwise.
        ax1 = fig.add_subplot(1, 2, 1, projection="polar")
        ax1.plot(theta_1d, pattern_dB[:, phi0_idx], label="phi=0 deg")
        ax1.plot(theta_1d, pattern_dB[:, phi90_idx], "--", label="phi=90 deg")
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax1.set_ylim(floor_dB, 0.0)
        ax1.set_title("Radiation pattern [dB]")
        ax1.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
        ax1.grid(True)

        # 3D surface in dB: clip below the floor, then map [floor_dB, 0 dB] -> [0, 1] for the radius
        # so the surface visually represents the dB pattern. Face colour also encodes dB level.
        pattern_dB_clipped = np.maximum(pattern_dB, floor_dB)
        radius_3d = (pattern_dB_clipped - floor_dB) / (-floor_dB)
        x = radius_3d * np.sin(theta_mesh) * np.cos(phi_mesh)
        y = radius_3d * np.sin(theta_mesh) * np.sin(phi_mesh)
        z = radius_3d * np.cos(theta_mesh)
        face_colors = plt.cm.viridis(radius_3d)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(x, y, z, facecolors=face_colors, linewidth=0, antialiased=True, shade=False)
        ax2.set_xlabel("x (antenna frame)")
        ax2.set_ylabel("y (antenna frame)")
        ax2.set_zlabel("z (boresight)")
        ax2.set_title(f"Radiation pattern [dB], floor {floor_dB:g} dB")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
