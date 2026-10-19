import ast
import configparser as cp
import datetime  # Import the module itself first

# ✅ Monkey patch for Python < 3.11 compatibility
if not hasattr(datetime, "UTC"):
    from datetime import timezone
    datetime.UTC = timezone.utc  # noqa: UP017  (Python < 3.11 shim: datetime.UTC does not exist yet here)
from dataclasses import fields
from datetime import UTC, datetime
import json as json
import logging
import math
import os
import re
import sys
import warnings

# Python versions the simulator has been validated against. The project's
# pyproject.toml requires >=3.12.3; running on a newer minor (e.g. 3.13)
# has surfaced issues with debugpy and binary-wheel availability for
# scientific packages, so we warn the user explicitly.
_PYTHON_MIN = (3, 12, 3)
_PYTHON_TESTED_MAX_MINOR = (3, 12)

# Cartopy passes reflection arrays containing NaN gaps through pyproj; pyproj
# wraps those entries as masked and then warns on the round-trip back to NaN.
# The plot output is correct (NaNs become line breaks), so silence the noise.
warnings.filterwarnings(
    "ignore",
    message="Warning: converting a masked element to nan.",
    category=UserWarning,
    module=r"pyproj\.transformer",
)

# import numpy              as np
# import xmltodict          as xd
import yaml

from functions.link_budget import BistaticLinkBudgetSettings
from object_definitions.AntennaPatternGenerator_def import AntennaPatternGenerator
from object_definitions.Constants_def import Constants

## UPDATE THE SIMULATIOR VERSION HERE ##
__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _check_python_version() -> None:
    """
    Check the current Python version against the simulator's supported 'Python range' and issue warnings if it's outside that range!

    The simulator has been tested on Python 3.12.x, and the pyproject.toml
    specifies a minimum of 3.12.3. Running on versions below 3.12.3 may lead to compatibility issues, while running on versions above 3.12
    (e.g., 3.13) may cause issues with certain dependencies like debugpy and cartopy.
    """
    current = sys.version_info[:3]
    current_str = ".".join(map(str, current))
    if current < _PYTHON_MIN:
        min_str = ".".join(map(str, _PYTHON_MIN))
        warnings.warn(
            f"Python {current_str} is below the simulator's minimum supported version ({min_str}). "
            "Some features may fail; please upgrade.",
            stacklevel=2,
        )
    elif current[:2] > _PYTHON_TESTED_MAX_MINOR:
        tested_str = ".".join(map(str, _PYTHON_TESTED_MAX_MINOR))
        warnings.warn(
            f"Python {current_str} is newer than the simulator's tested range (<= {tested_str}.x). "
            "If you hit import errors (debugpy, cartopy, etc.), recreate the venv with Python 3.12.",
            stacklevel=2,
        )

def parse_antenna_angles(filename: str) -> tuple[float, float, float]:
    """Extract (steering-across-track, steering-along-track, frequency) in (deg, deg, MHz) from an
    array_factor_{freq}MHz_{across}across_{along}along.json filename. Across/along-track may be negative."""
    across_match = re.search(r'(-?\d+\.?\d*)across', filename)
    along_match = re.search(r'(-?\d+\.?\d*)along', filename)
    freq_match = re.search(r'(\d+\.?\d*)MHz', filename)
    if not (across_match and along_match and freq_match):
        raise ValueError(f"Could not parse across-track/along-track/frequency from filename: {filename}")
    return float(across_match.group(1)), float(along_match.group(1)), float(freq_match.group(1))


def build_selected_signals(antenna_bands, loaded_constellation_names) -> list[dict]:
    """
    Validate the configured bands and return the [{band, constellation, freq_hz, eirp_dbw, chip_rate_hz}] signal plan.

    Each band is a constellation-specific signal (L1=GPS, E1=Galileo, ...). Invalid bands are warned
    about and dropped (never raise): unknown band names, bands with no transmit-EIRP entry, and bands
    whose owning constellation is not among `loaded_constellation_names`.
    """
    selected_signals = []
    loaded_by_upper = {name.upper(): name for name in loaded_constellation_names}  # canonical name -> actual key
    for band in antenna_bands or []:
        if band not in Constants.BANDS_CENTER_FREQUENCIES_Hz:
            logger.warning("Band %r is not a known GNSS band; skipping it. Known bands: %s.", band, sorted(Constants.BANDS_CENTER_FREQUENCIES_Hz))
            continue
        owner = Constants.get_constellation_for_band(band)  # canonical constellation name, or None when no EIRP entry
        if owner is None:
            logger.warning("Band %r has no transmit-EIRP entry (no constellation defined for it); skipping it.", band)
            continue
        if owner not in loaded_by_upper:
            logger.warning(
                "Band %r belongs to constellation %s, which is not among the loaded GNSS constellations %s; skipping it.",
                band, owner, sorted(loaded_by_upper),
            )
            continue
        constellation = loaded_by_upper[owner]  # use the key as it appears in gnss_constellation for downstream lookups
        selected_signals.append(
            {
                "band": band,
                "constellation": constellation,
                "freq_hz": float(Constants.BANDS_CENTER_FREQUENCIES_Hz[band]),
                "eirp_dbw": Constants.get_tx_eirp_dbw_for_band(constellation, band),
                "chip_rate_hz": Constants.get_chip_rate_hz_for_band(constellation, band),
            }
        )
    return selected_signals


def load_simulator_settings(path: str | None = None) -> dict:
    """
    Load engine/numerical simulator settings from the YAML file, kept separate from the per-mission INI.

    Mission scenario lives in Configuration_*.ini; simulator settings (angular resolutions, numerical
    grids, ...) live in simulator_settings.yaml. Strict like the INI loader: the file must exist and any
    requested key must be present -- there are no hidden defaults.
    """
    settings_path = Constants.simulator_settings_path if path is None else path
    if not os.path.isfile(settings_path):
        raise FileNotFoundError(
            f"Simulator settings file not found: {settings_path}. It holds engine settings such as "
            "antenna_pattern.theta_resolution / phi_resolution."
        )
    with open(settings_path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Simulator settings file {settings_path} must contain a top-level mapping; got {type(data).__name__}.")
    return data


_ANTENNA_PATTERN_YAML_KEYS = ("theta_resolution", "phi_resolution", "along_track_steering_steps", "across_track_steering_steps")


def _antenna_pattern_settings_from_yaml() -> dict:
    """Read the antenna-pattern engine settings (pattern resolution + steering step counts) strictly from the simulator YAML."""
    antenna_pattern = load_simulator_settings().get("antenna_pattern")
    if not isinstance(antenna_pattern, dict):
        raise KeyError(f"Missing 'antenna_pattern' mapping in {Constants.simulator_settings_path}.")
    missing = [key for key in _ANTENNA_PATTERN_YAML_KEYS if key not in antenna_pattern]
    if missing:
        raise KeyError(f"Missing key(s) {missing} under 'antenna_pattern' in {Constants.simulator_settings_path}.")
    try:
        return {key: int(antenna_pattern[key]) for key in _ANTENNA_PATTERN_YAML_KEYS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid antenna_pattern setting in {Constants.simulator_settings_path}: {exc}") from None


def _simulation_step_size_from_yaml() -> int:
    """Read the propagation/sampling step size [s] strictly from the 'simulation' block of the simulator YAML."""
    simulation = load_simulator_settings().get("simulation")
    if not isinstance(simulation, dict) or "step_size_s" not in simulation:
        raise KeyError(f"Missing 'simulation.step_size_s' in {Constants.simulator_settings_path}.")
    try:
        return int(simulation["step_size_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid simulation.step_size_s in {Constants.simulator_settings_path}: {exc}") from None


_TLE_FIT_YAML_KEYS = ("position_weight_m", "velocity_weight_m_s")


def tle_fit_weights_from_yaml() -> tuple[float, float]:
    """Read the osculating->SGP4 mean-element fit residual weights (position [m], velocity [m/s]) strictly from the simulator YAML."""
    tle_fit = load_simulator_settings().get("tle_fit")
    if not isinstance(tle_fit, dict):
        raise KeyError(f"Missing 'tle_fit' mapping in {Constants.simulator_settings_path}.")
    missing = [key for key in _TLE_FIT_YAML_KEYS if key not in tle_fit]
    if missing:
        raise KeyError(f"Missing key(s) {missing} under 'tle_fit' in {Constants.simulator_settings_path}.")
    try:
        position_weight_m = float(tle_fit["position_weight_m"])
        velocity_weight_m_s = float(tle_fit["velocity_weight_m_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tle_fit setting in {Constants.simulator_settings_path}: {exc}") from None
    if not (math.isfinite(position_weight_m) and position_weight_m > 0.0 and math.isfinite(velocity_weight_m_s) and velocity_weight_m_s > 0.0):
        raise ValueError(
            f"tle_fit weights in {Constants.simulator_settings_path} must be finite and > 0; "
            f"got position_weight_m={position_weight_m}, velocity_weight_m_s={velocity_weight_m_s}."
        )
    return position_weight_m, velocity_weight_m_s


def _validate_antenna_steering(along_extent, across_extent, yaml_ap) -> None:
    """
    User-facing steering guards named for the config keys (the generator keeps antenna-frame backstops).

    Symmetric steering: each extent must be strictly inside (-90, 90) deg, the two-plane corner must satisfy
    sin^2 + sin^2 <= 1, and each step count must be odd so the symmetric sweep includes boresight (0 deg).
    """
    max_along = max(abs(float(along_extent[0])), abs(float(along_extent[1])))
    max_across = max(abs(float(across_extent[0])), abs(float(across_extent[1])))
    for cfg_key, max_deg in (("along_track", max_along), ("across_track", max_across)):
        if max_deg >= 90.0:
            raise ValueError(f"[ANTENNA PATTERN].{cfg_key}_steering must stay strictly within (-90, 90) deg; got extent up to {max_deg} deg.")
    if math.sin(math.radians(max_along)) ** 2 + math.sin(math.radians(max_across)) ** 2 > 1.0 + 1e-9:
        raise ValueError(
            "[ANTENNA PATTERN] steering corner is unreachable: sin^2(max along-track) + sin^2(max across-track) > 1. "
            "Reduce along_track_steering / across_track_steering."
        )
    for cfg_key in ("along_track", "across_track"):
        steps = yaml_ap[f"{cfg_key}_steering_steps"]
        if steps % 2 == 0:
            raise ValueError(
                f"simulator_settings.yaml antenna_pattern.{cfg_key}_steering_steps ({steps}) must be odd so the "
                f"symmetric [ANTENNA PATTERN].{cfg_key}_steering sweep includes boresight (0 deg); use {steps + 1}."
            )


def initialize(config_file_name="Configuration.ini"):
    """
    =========================================================================================================
    Read a scenario configuration file and initialize the simulation settings.
    The configuration file acts as the main input for the simulation.

    INPUTS:
        config_file_name: [str, optional]   The name of the configuration file to read. Default is 'Configuration.ini'.

    OUTPUT:
        result: [tuple]   (config: ConfigParser, simulation_settings: dict) — the configuration settings read from the file
                          and a dictionary containing the simulation settings.

    Configuration File Structure:
        [SIMULATION SETTINGS]
        startTime          = dd.MM.YYYY HH:mm:ss
        simulationDuration = simulation_duration_in_hours                               (only needed for area-coverage analysis)
        (stepSize is an engine setting -> simulator_settings.yaml: simulation.step_size_s)

        [MOVING TARGET]
        start_pos = (longitude, latitude)                                               (only needed for moving target analysis)
        end_pos   = (longitude, latitude)                                               (only needed for moving target analysis)
        speed     = Speed_in_knots                                                      (only needed for moving target analysis)

        [ANTENNA]
        (the antenna pattern folder is not an INI key; it is fixed in Constants.antenna_pattern_folder)
        antenna_beamwidth_3dB  = [x1, y1, x2, y2, ...] (from 0 to 90 degrees rotationally) (only needed for moving target analysis)
        bands                  = ["L1", "L2", "L2", "L5"]                                        (only needed for moving target analysis)

        [ORBITS]
        orbit1 = (inclination, eccentricity, raan, arg_periapsis, true_anomaly, semi_major_axis)
        orbit2 = ...

        [GNSS CONSTELLATIONS]
        gnss_tle_files = [file1.tle, file2.tle, ...]       (TLE files for GNSS constellations; constellation names are inferred from filenames)

        [LINK BUDGET] (required) and [OCEAN] (required)
        Receiver-system losses and sea-state inputs for the bistatic ocean RCS. See
        `BistaticLinkBudgetSettings` in `functions/link_budget.py` for the accepted keys
        and units. INI-strict: every key must be present, missing keys raise. The dataclass
        defaults are the programmatic-API default only; the loader never falls back to them.

        [SHIP] (required when [MOVING TARGET] is present)
        length_m    = above-waterline length [m]   (> 0)
        beam_m      = above-waterline width [m]    (> 0)
        freeboard_m = above-waterline height [m]   (> 0)
    =========================================================================================================
    """
    _check_python_version()

    config = cp.ConfigParser()
    config.read(config_file_name)

    #[SIMULATION SETTINGS]
    try:
        start_time_dt = datetime.strptime(config["SIMULATION SETTINGS"]["startTime"], "%d.%m.%Y %H:%M:%S")  # Start time: Year, Month, Day, Hour, Minute, Second
        start_time_dt = start_time_dt.replace(tzinfo=UTC)  # Add timezone information
    except KeyError:
        raise KeyError("The 'startTime' key is missing in the 'SIMULATION SETTINGS' section of the configuration file.")  # noqa: B904

    step_size = _simulation_step_size_from_yaml()  # Seconds; engine setting -> simulator YAML, not the mission INI

    sim_duration = int(config["SIMULATION SETTINGS"]["simulationDuration"]) if "simulationDuration" in config["SIMULATION SETTINGS"] else None  # Hours

    #[MOVING TARGET]
    try:
        moving_tgt_speed = float(config["MOVING TARGET"]["speed"]) if "MOVING TARGET" in config else None  # Speed of the moving target in knots
        start_coord_ll = (
            ast.literal_eval(config["MOVING TARGET"]["start_pos"]) if "MOVING TARGET" in config else None
        )  # Start coordinates of the moving target in (latitude, longitude) format
        end_coord_ll = (
            ast.literal_eval(config["MOVING TARGET"]["end_pos"]) if "MOVING TARGET" in config else None
        )  # End coordinates of the moving target in (latitude, longitude) format
    except KeyError:
        raise KeyError("The 'speed', 'start_pos', or 'end_pos' key is missing in the 'MOVING TARGET' section of the configuration file.")  # noqa: B904

    #[COVERAGE]
    try:
        grid_size_m = (
            int(config["COVERAGE"]["grid_size_meters"]) if ("COVERAGE" in config and "grid_size_meters" in config["COVERAGE"]) else None
        )  # Default grid size in meters
        coverage_file_path = (
            config["COVERAGE"]["shapefile_path"] if ("COVERAGE" in config and "shapefile_path" in config["COVERAGE"]) else None
        )  # Path to the coverage shapefile
    except KeyError:
        raise KeyError("The 'grid_size_meters' or 'shapefile_path' key is missing in the 'COVERAGE' section of the configuration file.")  # noqa: B904
    max_bistatic_bksc_angle_deg = None
    if "COVERAGE" in config and "max_bistatic_bksc_angle_deg" in config["COVERAGE"]:
        try:
            max_bistatic_bksc_angle_deg = float(config["COVERAGE"]["max_bistatic_bksc_angle_deg"])
        except ValueError:
            raise ValueError(
                f"Invalid value for 'max_bistatic_bksc_angle_deg' in [COVERAGE]: {config['COVERAGE']['max_bistatic_bksc_angle_deg']}"
            ) from None
        if not math.isfinite(max_bistatic_bksc_angle_deg) or not (0.0 < max_bistatic_bksc_angle_deg < 90.0):
            raise ValueError("[COVERAGE].max_bistatic_bksc_angle_deg must be finite and in (0, 90) deg.")

    # The antenna-pattern folder is a fixed data path (Constants.antenna_pattern_folder), not an INI
    # knob. It is "active" (non-None) only when an antenna is configured, so configs without an antenna
    # keep antenna_pattern_folder = None -- the flag main_simulation_areaCoverage.py uses to skip the antenna.
    antenna_pattern_folder = Constants.antenna_pattern_folder if ("ANTENNA" in config or "ANTENNA PATTERN" in config) else None

    # [ANTENNA].bands parsed up-front: the bands are the single source of truth for which carrier
    # frequencies the antenna patterns must cover. [ANTENNA PATTERN] below derives its frequencies from
    # them, so there is no separate freqs key to keep in sync (and no silent frequency snapping).
    # [ANTENNA] absent -> antenna_bands is None (no antenna configured).
    if "ANTENNA" in config:
        bands_raw = config["ANTENNA"].get("bands")
        antenna_bands = [b.strip().strip('"') for b in bands_raw[1:-1].split(",") if b.strip()] if bands_raw else []
    else:
        antenna_bands = None

    #[ANTENNA PATTERN]
    # Parameters for AntennaPatternGenerator. Parsed up-front so the generator
    # can run (and rebuild stale JSONs) before [ANTENNA] is consumed below.
    antenna_pattern_settings = None
    expected_antenna_pattern_files = None
    if "ANTENNA PATTERN" in config:
        section = config["ANTENNA PATTERN"]
        required_keys = ("elements_along_x", "elements_along_y", "d_x", "d_y", "across_track_steering", "along_track_steering")
        missing = [k for k in required_keys if k not in section]
        if missing:
            raise KeyError(f"Missing required key(s) in [ANTENNA PATTERN] section: {missing}")

        # Carrier frequencies are DERIVED from [ANTENNA].bands (the single source of truth), so a
        # generated pattern always exists for every selected band and the gain lookup never silently
        # snaps to a different frequency. Unknown band names are skipped here -- they warn-and-drop
        # later in build_selected_signals.
        band_freqs_mhz = sorted(
            {
                Constants.BANDS_CENTER_FREQUENCIES_Hz[b] / Constants.MHZ_TO_HZ
                for b in (antenna_bands or [])
                if b in Constants.BANDS_CENTER_FREQUENCIES_Hz
            }
        )
        if not band_freqs_mhz:
            raise ValueError(
                "[ANTENNA PATTERN] is configured but no carrier frequencies could be derived from "
                '[ANTENNA].bands. List at least one known band, e.g. bands = ["L1"].'
            )
        # Engine/numerical settings (pattern resolution + steering step counts) -> simulator YAML (strict);
        # the INI keeps only the mission-facing steering extent [min_deg, max_deg].
        yaml_ap = _antenna_pattern_settings_from_yaml()
        try:
            along_extent = ast.literal_eval(section["along_track_steering"])
            across_extent = ast.literal_eval(section["across_track_steering"])
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid steering extent in [ANTENNA PATTERN] section: {e}") from None
        for name, extent in (("along_track_steering", along_extent), ("across_track_steering", across_extent)):
            if not (isinstance(extent, (list, tuple)) and len(extent) == 2):
                raise ValueError(
                    f"[ANTENNA PATTERN] {name} must be [min_deg, max_deg]; the step count moved to "
                    f"simulator_settings.yaml (antenna_pattern.{name}_steps)."
                )
        _validate_antenna_steering(along_extent, across_extent, yaml_ap)
        try:
            antenna_pattern_settings = {
                "elements_along_x": int(section["elements_along_x"]),
                "elements_along_y": int(section["elements_along_y"]),
                "d_x": float(section["d_x"]),
                "d_y": float(section["d_y"]),
                "freqs": band_freqs_mhz,
                # generator steering keys are antenna-frame axes: across-track = array y, along-track = array x
                "across_track_steering": [across_extent[0], across_extent[1], yaml_ap["across_track_steering_steps"]],
                "along_track_steering": [along_extent[0], along_extent[1], yaml_ap["along_track_steering_steps"]],
                "theta_resolution": yaml_ap["theta_resolution"],
                "phi_resolution": yaml_ap["phi_resolution"],
                "output_folder": antenna_pattern_folder,
            }
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid value in [ANTENNA PATTERN] section: {e}") from None

        # Generate (or reuse cached) antenna-pattern JSONs before the [ANTENNA]
        # block reads them. AntennaPatternGenerator decides whether to skip
        # generation based on a metadata signature stored in the pattern folder.
        antenna_pattern_generator = AntennaPatternGenerator(**antenna_pattern_settings)
        antenna_pattern_generator.generate()
        expected_antenna_pattern_files = set(antenna_pattern_generator._expected_filenames())

    #[ANTENNA]
    try:
        antenna_patterns = {}  # Initialize the dictionary here (antenna_bands was parsed up-front, above)

        # If antenna is initialzied in the ini-file:
        if "ANTENNA" in config:
            if not os.path.isdir(antenna_pattern_folder):
                raise NotADirectoryError(
                    f"Antenna pattern folder '{antenna_pattern_folder}' (Constants.antenna_pattern_folder) "
                    "does not exist or is not a directory."
                )
            filepaths_antenna_pattern = []
            for file in sorted(os.listdir(antenna_pattern_folder)):
                # Pick up only the pattern JSONs; the AntennaPatternGenerator metadata file
                # (antenna_pattern_metadata.json) is a sibling and must be ignored here.
                if file.startswith("array_factor_") and file.endswith(".json"):
                    if expected_antenna_pattern_files is not None and file not in expected_antenna_pattern_files:
                        continue
                    filepaths_antenna_pattern.append(os.path.join(antenna_pattern_folder, file))

            if not filepaths_antenna_pattern:
                raise FileNotFoundError(
                    f"No antenna pattern files found in '{antenna_pattern_folder}'. "
                    "Expected files named 'array_factor_*.json'."
                )

            for file in filepaths_antenna_pattern:
                if not os.path.isfile(file):
                    raise FileNotFoundError(f"The specified antenna pattern file '{file}' does not exist. Please check the file path in the configuration file.")

                # Key each pattern by its (steering-across-track, steering-along-track, frequency) parsed from the filename.
                across_track, along_track, freq = parse_antenna_angles(os.path.basename(file))
                antenna_patterns[(across_track, along_track, freq)] = file
            antenna_patterns = dict(sorted(antenna_patterns.items()))
            antenna_polarizations = None  # polarizations removed from the INI for now (currently unused downstream)
        else:
            antenna_polarizations = None
            antenna_bands = None

        if "ANTENNA" in config:
            if "footprint_dB" not in config["ANTENNA"]:
                raise ValueError("The 'footprint_dB' key is missing in the 'ANTENNA' section of the configuration file. It must be set to a value below 0 dB.")
            footprint_dB_str = config["ANTENNA"]["footprint_dB"]
            try:
                footprint_dB = float(footprint_dB_str)
            except ValueError:
                raise ValueError(f"Invalid value for 'footprint_dB': {footprint_dB_str}. It must be a float.") from None
            if footprint_dB >= 0.0:
                raise ValueError("'footprint_dB' must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
        else:
            footprint_dB = None
    except KeyError as e:
        raise KeyError(f"A required key is missing in the 'ANTENNA' section of the configuration file: {e}") from None

    #[ORBITS]
    # Check how many orbits are specified in the configuration file
    if "ORBITS" not in config:
        raise KeyError("The 'ORBITS' section is missing in the configuration file.")
    orbits_input = []
    orbit_no = 1
    orbit_str = "orbit" + str(orbit_no)
    while orbit_str in config["ORBITS"]:
        orb_str = config["ORBITS"][orbit_str]
        orbit_values = ast.literal_eval(orb_str)
        orbits_input.append(
            [
                orbit_values[0],
                orbit_values[1],
                orbit_values[2],
                orbit_values[3],
                orbit_values[4],
                orbit_values[5],
                start_time_dt,
            ]
        )
        orbit_no += 1
        orbit_str = "orbit" + str(orbit_no)

    #[GNSS CONSTELLATIONS]
    if "GNSS CONSTELLATIONS" not in config:
        raise KeyError("The 'GNSS CONSTELLATIONS' section is missing in the configuration file.")
    try:
        const_string = config["GNSS CONSTELLATIONS"]["gnss_tle_files"]
    except KeyError:
        raise KeyError("The 'gnss_tle_files' key is missing in the 'GNSS CONSTELLATIONS' section of the configuration file.") from None

    tle_filenames = [s.strip() for s in const_string.strip("[]").split(",") if s.strip()]
    const_string_updated = []
    gnss_names = []
    unknown_files = []
    for filename in tle_filenames:
        match = re.search(r"(GPS|GLONASS|GALILEO|BEIDOU)", filename, re.IGNORECASE)
        if match:
            gnss_names.append(match.group(0))
            const_string_updated.append(os.path.join(Constants.tle_folderpath, filename))
        else:
            unknown_files.append(filename)

    if unknown_files:
        raise ValueError(
            f"Cannot identify GNSS constellation for TLE file(s) {unknown_files} in [GNSS CONSTELLATIONS].gnss_tle_files. "
            "Filename must contain one of: GPS, GLONASS, GALILEO, BEIDOU."
        )

    gnss_constellation = {
        const_name: {"tle_file": const_tle, "index": idx}
        for idx, (const_tle, const_name) in enumerate(zip(const_string_updated, gnss_names, strict=False))
    }

    if not gnss_constellation:
        raise ValueError(
            "[GNSS CONSTELLATIONS].gnss_tle_files is empty — the simulation needs at least one GNSS constellation."
        )

    #[ANTENNA] band selection -> per-(constellation, band) signal plan
    # Each configured band is a constellation-specific signal (L1=GPS, E1=Galileo, ...). Invalid
    # entries warn and drop instead of stopping, so a typo never aborts the run (see build_selected_signals).
    selected_signals = build_selected_signals(antenna_bands, gnss_constellation.keys())

    #[REFLECTION] (REQUIRED SECTION)
    # max_central_angle_deg is required always; max_incidence_angle_deg only when
    # reflection_type = backscatter. No fallback defaults — these knobs materially
    # change the result count and must be set explicitly per simulation.
    if "REFLECTION" not in config:
        raise ValueError("[REFLECTION] section is required in the configuration file.")
    section = config["REFLECTION"]
    reflection_type = section.get("reflection_type", "specular").strip('"').strip("'")
    if reflection_type not in {"specular", "backscatter"}:
        raise ValueError("[REFLECTION].reflection_type must be either 'specular' or 'backscatter'.")
    if "earth_model" not in section:
        raise ValueError("[REFLECTION].earth_model is required in the configuration file.")
    earth_model = section["earth_model"].strip('"').strip("'")
    if earth_model not in {"spherical", "elliptical"}:
        raise ValueError("[REFLECTION].earth_model must be either 'spherical' or 'elliptical'.")
    if "max_central_angle_deg" not in section:
        raise ValueError("[REFLECTION].max_central_angle_deg is required in the configuration file.")
    try:
        max_central_angle_deg = float(section["max_central_angle_deg"])
    except ValueError:
        raise ValueError(
            f"Invalid value for 'max_central_angle_deg' in [REFLECTION]: {section['max_central_angle_deg']}"
        ) from None
    if not math.isfinite(max_central_angle_deg) or not (0.0 < max_central_angle_deg <= 180.0):
        raise ValueError("[REFLECTION].max_central_angle_deg must be finite and in (0, 180] deg.")
    max_incidence_angle_deg = None
    if reflection_type == "backscatter":
        if "max_incidence_angle_deg" not in section:
            raise ValueError(
                "[REFLECTION].max_incidence_angle_deg is required when reflection_type = backscatter."
            )
        try:
            max_incidence_angle_deg = float(section["max_incidence_angle_deg"])
        except ValueError:
            raise ValueError(
                f"Invalid value for 'max_incidence_angle_deg' in [REFLECTION]: {section['max_incidence_angle_deg']}"
            ) from None
        if not math.isfinite(max_incidence_angle_deg) or not (0.0 < max_incidence_angle_deg <= 90.0):
            raise ValueError("[REFLECTION].max_incidence_angle_deg must be finite and in (0, 90] deg.")

    # Backscatter AREA COVERAGE (grid_size_meters marks an area-coverage config) additionally needs the
    # near-backscatter acceptance half-angle that sets the swath width around the BRP track.
    if reflection_type == "backscatter" and grid_size_m is not None and max_bistatic_bksc_angle_deg is None:
        raise ValueError(
            "[COVERAGE].max_bistatic_bksc_angle_deg is required for backscatter area coverage "
            "(reflection_type = backscatter with a gridded [COVERAGE] region)."
        )

    #[LINK BUDGET] (REQUIRED SECTION)
    # Receiver-system parameters. No fallback defaults; missing keys raise.
    # (Transmit EIRP is per-(constellation, band) and lives in Constants.GNSS_EIRP_DBW.)
    if "LINK BUDGET" not in config:
        raise ValueError("[LINK BUDGET] section is required in the configuration file.")
    section = config["LINK BUDGET"]
    link_budget = {}
    # Rx-system losses feed the link equation in both analysis modes -> always required.
    for key in ("system_losses_db", "polarization_losses_db"):
        if key not in section:
            raise ValueError(f"[LINK BUDGET].{key} is required in the configuration file.")
        try:
            link_budget[key] = float(section[key])
        except ValueError:
            raise ValueError(f"Invalid value for '{key}' in [LINK BUDGET]: {section[key]}") from None
    # Thermal-noise + detection-threshold settings drive SNR/SINR detection, which only the
    # moving-target analysis computes. Area coverage is geometric and ignores them, so they are
    # required only when [MOVING TARGET] is present; parsed leniently (if supplied) otherwise.
    detection_keys = ("noise_bandwidth_hz", "system_noise_temperature_k", "detection_sinr_threshold_db")
    moving_target_mode = "MOVING TARGET" in config
    for key in detection_keys:
        if key not in section:
            if moving_target_mode:
                raise ValueError(f"[LINK BUDGET].{key} is required when [MOVING TARGET] is present.")
            continue
        try:
            link_budget[key] = float(section[key])
        except ValueError:
            raise ValueError(f"Invalid value for '{key}' in [LINK BUDGET]: {section[key]}") from None
    for key in ("noise_bandwidth_hz", "system_noise_temperature_k"):
        if key in link_budget and link_budget[key] <= 0.0:
            raise ValueError(f"[LINK BUDGET].{key} must be > 0; got {link_budget[key]}")
    if "detection_sinr_threshold_db" in link_budget and not math.isfinite(link_budget["detection_sinr_threshold_db"]):
        raise ValueError(
            f"[LINK BUDGET].detection_sinr_threshold_db must be finite; got {link_budget['detection_sinr_threshold_db']}"
        )
    if "ocean_patch_integration_enabled" in section:
        try:
            link_budget["ocean_patch_integration_enabled"] = section.getboolean("ocean_patch_integration_enabled")
        except ValueError:
            raise ValueError(
                f"Invalid value for 'ocean_patch_integration_enabled' in [LINK BUDGET]: {section['ocean_patch_integration_enabled']}"
            ) from None
    if "ocean_patch_grid_size" in section:
        try:
            link_budget["ocean_patch_grid_size"] = int(section["ocean_patch_grid_size"])
        except ValueError:
            raise ValueError(f"Invalid value for 'ocean_patch_grid_size' in [LINK BUDGET]: {section['ocean_patch_grid_size']}") from None
        if link_budget["ocean_patch_grid_size"] < 3:
            raise ValueError(
                f"[LINK BUDGET].ocean_patch_grid_size must be >= 3; got {link_budget['ocean_patch_grid_size']}"
            )
        if link_budget["ocean_patch_grid_size"] > 101:
            raise ValueError(
                f"[LINK BUDGET].ocean_patch_grid_size must be <= 101; got {link_budget['ocean_patch_grid_size']}"
            )
    # Delay-Doppler resolution cell (moving-target detection only). Optional feature gate; when enabled,
    # the coherent integration time is required (it sets the Doppler-bin width of the cell). Mutual
    # exclusion with patch integration is enforced in BistaticLinkBudgetSettings.__post_init__.
    if "delay_doppler_cell_enabled" in section:
        try:
            link_budget["delay_doppler_cell_enabled"] = section.getboolean("delay_doppler_cell_enabled")
        except ValueError:
            raise ValueError(
                f"Invalid value for 'delay_doppler_cell_enabled' in [LINK BUDGET]: {section['delay_doppler_cell_enabled']}"
            ) from None
    if "coherent_integration_time_s" in section:
        try:
            link_budget["coherent_integration_time_s"] = float(section["coherent_integration_time_s"])
        except ValueError:
            raise ValueError(f"Invalid value for 'coherent_integration_time_s' in [LINK BUDGET]: {section['coherent_integration_time_s']}") from None
        if link_budget["coherent_integration_time_s"] <= 0.0:
            raise ValueError(
                f"[LINK BUDGET].coherent_integration_time_s must be > 0; got {link_budget['coherent_integration_time_s']}"
            )
    if "num_incoherent_looks" in section:
        try:
            link_budget["num_incoherent_looks"] = int(section["num_incoherent_looks"])
        except ValueError:
            raise ValueError(f"Invalid value for 'num_incoherent_looks' in [LINK BUDGET]: {section['num_incoherent_looks']}") from None
        if link_budget["num_incoherent_looks"] < 1:
            raise ValueError(
                f"[LINK BUDGET].num_incoherent_looks must be >= 1; got {link_budget['num_incoherent_looks']}"
            )

    #[OCEAN] (REQUIRED SECTION)
    # Sea-state and water-property inputs for the bistatic ocean RCS model. INI-strict.
    if "OCEAN" not in config:
        raise ValueError("[OCEAN] section is required in the configuration file.")
    section = config["OCEAN"]
    for key in (
        "ocean_wave_height_m",
        "ocean_wind_speed_m_s",
        "ocean_wind_direction_deg",
        "ocean_sea_temperature_c",
        "ocean_salinity_psu",
        "ocean_inverse_wave_age",
    ):
        if key not in section:
            raise ValueError(f"[OCEAN].{key} is required in the configuration file.")
        try:
            link_budget[key] = float(section[key])
        except ValueError:
            raise ValueError(f"Invalid value for '{key}' in [OCEAN]: {section[key]}") from None

    # Guard: parsed keys must stay in sync with BistaticLinkBudgetSettings. The detection_keys are
    # optional (absent for area coverage); every other field must be present.
    expected_fields = {f.name for f in fields(BistaticLinkBudgetSettings)}
    parsed_keys = set(link_budget.keys())
    unknown_keys = parsed_keys - expected_fields
    optional_link_budget_keys = set(detection_keys) | {
        "ocean_patch_integration_enabled",
        "ocean_patch_grid_size",
        "delay_doppler_cell_enabled",
        "coherent_integration_time_s",
        "num_incoherent_looks",
    }
    missing_required = (expected_fields - optional_link_budget_keys) - parsed_keys
    if unknown_keys or missing_required:
        raise RuntimeError(
            f"Parsed [LINK BUDGET]+[OCEAN] keys {sorted(parsed_keys)} are out of sync with "
            f"BistaticLinkBudgetSettings fields {sorted(expected_fields)} "
            f"(optional: {sorted(optional_link_budget_keys)}). Update the loader and the dataclass together."
        )

    #[SHIP] (REQUIRED with [MOVING TARGET])
    # Physical dimensions of the moving target, above waterline only. INI-strict.
    ship = None
    if "MOVING TARGET" in config:
        if "SHIP" not in config:
            raise ValueError("[SHIP] section is required when [MOVING TARGET] is present.")
        section = config["SHIP"]
        ship = {}
        for key in ("length_m", "beam_m", "freeboard_m"):
            if key not in section:
                raise ValueError(f"[SHIP].{key} is required in the configuration file.")
            try:
                value = float(section[key])
            except ValueError:
                raise ValueError(f"Invalid value for '{key}' in [SHIP]: {section[key]}") from None
            if not value > 0.0:
                raise ValueError(f"[SHIP].{key} must be > 0; got {value}")
            ship[key] = value

    #[PLOT] (OPTIONAL SECTION)
    detection_bbox_buffer_deg = 2.0
    if "PLOT" in config and "detection_bbox_buffer_deg" in config["PLOT"]:
        try:
            detection_bbox_buffer_deg = float(config["PLOT"]["detection_bbox_buffer_deg"])
        except ValueError:
            raise ValueError(
                f"Invalid value for 'detection_bbox_buffer_deg' in [PLOT]: {config['PLOT']['detection_bbox_buffer_deg']}"
            ) from None

    #[PLOT] animation knobs (moving-target only)
    # Animation only applies when [MOVING TARGET] is present. INI-strict:
    # create_animation must be set; its sub-knobs are required when it is True.
    create_animation = False
    animation_settings = None
    if "MOVING TARGET" in config:
        if "PLOT" not in config or "create_animation" not in config["PLOT"]:
            raise ValueError("[PLOT].create_animation is required when [MOVING TARGET] is present.")
        plot_section = config["PLOT"]
        try:
            create_animation = plot_section.getboolean("create_animation")
        except ValueError:
            raise ValueError(
                f"Invalid boolean for [PLOT].create_animation: {plot_section['create_animation']}"
            ) from None
        if create_animation:
            required_anim_keys = ("animation_filename", "animation_fps")
            missing_anim = [k for k in required_anim_keys if k not in plot_section]
            if missing_anim:
                raise ValueError(f"[PLOT] keys required when create_animation = True: {missing_anim}")
            try:
                animation_settings = {
                    "output_filename": plot_section["animation_filename"].strip().strip('"').strip("'"),
                    "fps": int(plot_section["animation_fps"]),
                }
            except ValueError as e:
                raise ValueError(f"Invalid value in [PLOT] animation settings: {e}") from None


    simulation_settings = {
        #[SIMULATION SETTINGS]
        "start_time": start_time_dt,
        "step_size": step_size,
        "sim_duration": sim_duration,
        "reflection_type": reflection_type,
        "earth_model": earth_model,
        "max_central_angle_deg": max_central_angle_deg,
        "max_incidence_angle_deg": max_incidence_angle_deg,
        #[ORBITS]
        "orbits": orbits_input,
        "No_satellites": len(orbits_input),
        #[GNSS CONSTELLATIONS]
        "GNSS_constellations": gnss_constellation,
        "selected_signals": selected_signals,
        #[COVERAGE]
        "shapefile_path": coverage_file_path,
        "grid_size_m": grid_size_m,
        "max_bistatic_bksc_angle_deg": max_bistatic_bksc_angle_deg,
        #[MOVING TARGET]
        "moving_target_speed": moving_tgt_speed,
        "start_coord_ll": start_coord_ll,
        "end_coord_ll": end_coord_ll,
        #[SHIP]
        "ship": ship,
        #[ANTENNA PATTERN]
        "antenna_pattern_settings": antenna_pattern_settings,
        #[ANTENNA]
        "antenna_pattern_folder": antenna_pattern_folder,
        "antenna_pattern_files": antenna_patterns,
        "polarizations": antenna_polarizations,
        "bands": antenna_bands,
        "footprint_dB": float(footprint_dB) if footprint_dB is not None else None,
        #[LINK BUDGET]
        "link_budget": link_budget,
        #[PLOT]
        "detection_bbox_buffer_deg": detection_bbox_buffer_deg,
        "create_animation": create_animation,
        "animation_settings": animation_settings,
    }
    return config, simulation_settings

# Export the initialize function
__all__ = ["build_selected_signals", "initialize"]
