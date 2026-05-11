import ast
import configparser as cp
import datetime  # Import the module itself first
# ✅ Monkey patch for Python < 3.11 compatibility
if not hasattr(datetime, "UTC"):
    from datetime import timezone
    datetime.UTC = timezone.utc
from dataclasses import fields
from datetime import datetime, timezone
import json as json
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
from functions.link_budget import BistaticLinkBudgetSettings
from object_definitions.AntennaPatternGenerator_def import AntennaPatternGenerator
from object_definitions.Constants_def import Constants

## UPDATE THE SIMULATIOR VERSION HERE ##
__version__ = "0.1.0"


def _check_python_version() -> None:
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

def parse_antenna_angles(filename):
    # Extract azimuth and elevation using regex
    az_match = re.search(r'(\d+\.?\d*)az', filename)
    el_match = re.search(r'(\d+\.?\d*)el', filename)
    freq_match = re.search(r'(\d+\.?\d*)MHz', filename)

    if az_match and el_match and freq_match:
        azimuth = float(az_match.group(1))
        elevation = float(el_match.group(1))
        freq = float(freq_match.group(1))

        return azimuth, elevation, freq
    else:
        raise ValueError(f"Could not parse azimuth/elevation from filename: {filename}")


def _normalize_path_for_compare(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _get_antenna_pattern_folder(config, pattern_section=None):
    """
    Resolve the antenna-pattern folder from configuration.

    [ANTENNA].pattern_folder is the canonical setting. The older
    [ANTENNA].antenna_filepaths and [ANTENNA PATTERN].output_folder keys are
    still accepted for compatibility.
    """
    antenna_folder = None
    if "ANTENNA" in config:
        antenna_section = config["ANTENNA"]
        if "pattern_folder" in antenna_section:
            antenna_folder = antenna_section["pattern_folder"].strip()
        elif "antenna_pattern_folder" in antenna_section:
            antenna_folder = antenna_section["antenna_pattern_folder"].strip()
            warnings.warn(
                "[ANTENNA].antenna_pattern_folder is deprecated; use [ANTENNA].pattern_folder instead.",
                stacklevel=2,
            )
        elif "antenna_filepaths" in antenna_section:
            antenna_folder = antenna_section["antenna_filepaths"].strip()
            warnings.warn(
                "[ANTENNA].antenna_filepaths is deprecated; use [ANTENNA].pattern_folder instead.",
                stacklevel=2,
            )

    pattern_folder = None
    if pattern_section is not None and "output_folder" in pattern_section:
        pattern_folder = pattern_section["output_folder"].strip()
        warnings.warn(
            "[ANTENNA PATTERN].output_folder is deprecated; use [ANTENNA].pattern_folder instead.",
            stacklevel=2,
        )

    if antenna_folder and pattern_folder and _normalize_path_for_compare(antenna_folder) != _normalize_path_for_compare(pattern_folder):
        raise ValueError(
            "Antenna pattern folder is configured in two places with different values. "
            "Use only [ANTENNA].pattern_folder."
        )

    return antenna_folder or pattern_folder


def initialize(config_file_name="Configuration.ini"):
    """
    =========================================================================================================
    Read the configuration file (default: 'Configuration.ini') and initialize the simulation settings.
    The configuration file acts as the main input for the simulation.

    INPUTS:
        config_file_name: [str, optional]   The name of the configuration file to read. Default is 'Configuration.ini'.

    OUTPUT:
        result: [tuple]   (config: ConfigParser, simulation_settings: dict) — the configuration settings read from the file
                          and a dictionary containing the simulation settings.

    Configuration File Structure:
        [SIMULATION SETTINGS]
        startTime          = dd.MM.YYYY HH:mm:ss
        stepSize           = step_size_in_seconds
        simulationDuration = simulation_duration_in_hours                               (only needed for area-coverage analysis)

        [MOVING TARGET]
        start_pos = (longitude, latitude)                                               (only needed for moving target analysis)
        end_pos   = (longitude, latitude)                                               (only needed for moving target analysis)
        speed     = Speed_in_knots                                                      (only needed for moving target analysis)

        [ANTENNA]
        pattern_folder         = path/to/generated_or_precomputed_patterns                       (only needed for moving target analysis)
        antenna_beamwidth_3dB  = [x1, y1, x2, y2, ...] (from 0 to 90 degrees rotationally) (only needed for moving target analysis)
        max_steering_angle_deg = [azimuth, elevation] (in degrees)                         (only needed for moving target analysis)
        polarizations          = ["LHCP", "RHCP", "VP", "HP"]                              (only needed for moving target analysis)
        bands                  = ["L1", "L2", "L2", "L5"]                                        (only needed for moving target analysis)

        [ORBITS]
        orbit1 = (inclination, eccentricity, raan, arg_perigee, mean_anomaly, apogee)
        orbit2 = ...

        [GNSS CONSTELLATIONS]
        gnss_tle_files   = [file1.tle, file2.tle, ...]    (TLE files for GNSS constellations)
        gnss_const_names = [const1, const2, ...]          (Names of the GNSS constellations, e.g., GPS, GLONASS, GALILEO, BEIDOU)

        [LINK BUDGET] (optional)
        See `BistaticLinkBudgetSettings` in `functions/link_budget.py` for the
        list of accepted keys, units and defaults. Any subset can be supplied;
        unspecified keys fall back to the dataclass defaults.
    =========================================================================================================
    """
    _check_python_version()

    config = cp.ConfigParser()
    config.read(config_file_name)

    #[SIMULATION SETTINGS]
    try:
        start_time_dt = datetime.strptime(config["SIMULATION SETTINGS"]["startTime"], "%d.%m.%Y %H:%M:%S")  # Start time: Year, Month, Day, Hour, Minute, Second
        start_time_dt = start_time_dt.replace(tzinfo=timezone.utc)  # Add timezone information
    except KeyError:
        raise KeyError("The 'startTime' key is missing in the 'SIMULATION SETTINGS' section of the configuration file.")  # noqa: B904

    try:
        step_size = int(config["SIMULATION SETTINGS"]["stepSize"])  # Seconds
    except KeyError:
        raise KeyError("The 'stepSize' key is missing in the 'SIMULATION SETTINGS' section of the configuration file.")  # noqa: B904

    sim_duration = int(config["SIMULATION SETTINGS"]["simulationDuration"]) if "simulationDuration" in config["SIMULATION SETTINGS"] else None  # Hours

    #[MOVING TARGET]
    try:
        moving_tgt_speed = float(config["MOVING TARGET"]["speed"]) if "MOVING TARGET" in config else None  # Speed of the moving target in km/h
        start_coord_ll = (
            ast.literal_eval(config["MOVING TARGET"]["start_pos"]) if "MOVING TARGET" in config else None
        )  # Start coordinates of the moving target in (longitude, latitude) format
        end_coord_ll = (
            ast.literal_eval(config["MOVING TARGET"]["end_pos"]) if "MOVING TARGET" in config else None
        )  # End coordinates of the moving target in (longitude, latitude) format
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

    pattern_section = config["ANTENNA PATTERN"] if "ANTENNA PATTERN" in config else None
    antenna_pattern_folder = _get_antenna_pattern_folder(config, pattern_section)

    #[ANTENNA PATTERN]
    # Parameters for AntennaPatternGenerator. Parsed up-front so the generator
    # can run (and rebuild stale JSONs) before [ANTENNA] is consumed below.
    antenna_pattern_settings = None
    if "ANTENNA PATTERN" in config:
        section = config["ANTENNA PATTERN"]
        required_keys = ("elements_along_x", "elements_along_y", "d_x", "d_y", "freqs", "elevation_directivity", "azimuth_directivity")
        missing = [k for k in required_keys if k not in section]
        if missing:
            raise KeyError(f"Missing required key(s) in [ANTENNA PATTERN] section: {missing}")
        if antenna_pattern_folder is None:
            raise KeyError("Missing antenna pattern folder. Add [ANTENNA] pattern_folder = <path>.")
        try:
            antenna_pattern_settings = {
                "elements_along_x": int(section["elements_along_x"]),
                "elements_along_y": int(section["elements_along_y"]),
                "d_x": float(section["d_x"]),
                "d_y": float(section["d_y"]),
                "freqs": [float(f) for f in ast.literal_eval(section["freqs"])],
                "elevation_directivity": ast.literal_eval(section["elevation_directivity"]),
                "azimuth_directivity": ast.literal_eval(section["azimuth_directivity"]),
                "elevation_resolution": int(section.get("elevation_resolution", "180")),
                "azimuth_resolution": int(section.get("azimuth_resolution", "360")),
                "output_folder": antenna_pattern_folder,
            }
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid value in [ANTENNA PATTERN] section: {e}") from None

        # Generate (or reuse cached) antenna-pattern JSONs before the [ANTENNA]
        # block reads them. AntennaPatternGenerator decides whether to skip
        # generation based on a metadata signature stored in the pattern folder.
        AntennaPatternGenerator(**antenna_pattern_settings).generate()

    #[ANTENNA]
    try:
        antenna_patterns = {}  # Initialize the dictionary here
        max_steering_angle_deg = (
            ast.literal_eval(config["ANTENNA"]["max_steering_angle_deg"])
            if ("ANTENNA" in config and "max_steering_angle_deg" in config["ANTENNA"])
            else None
        )
        bands = config["ANTENNA"]["bands"] if ("ANTENNA" in config and "bands" in config["ANTENNA"]) else None  # Antenna bands
        if bands is None:
            bands = []
            antenna_bands = []
        else:
            # Remove square brackets, quotes and split
            antenna_bands = [band.strip().strip('"') for band in bands[1:-1].split(",")]

        # If antenna is initialzied in the ini-file:
        if "ANTENNA" in config:
            if antenna_pattern_folder is None:
                raise KeyError("Missing [ANTENNA] pattern_folder.")
            if antenna_pattern_folder is not None and not os.path.isdir(antenna_pattern_folder):
                raise NotADirectoryError(f"The specified antenna pattern folder '{antenna_pattern_folder}' does not exist or is not a directory. Please check the folder path in the configuration file.")
            filepaths_antenna_pattern = []
            for file in sorted(os.listdir(antenna_pattern_folder)):
                # Pick up only the pattern JSONs; the AntennaPatternGenerator metadata file
                # (antenna_pattern_metadata.json) is a sibling and must be ignored here.
                if file.startswith("array_factor_") and file.endswith(".json"):
                    filepaths_antenna_pattern.append(os.path.join(antenna_pattern_folder, file))

            if not filepaths_antenna_pattern:
                raise FileNotFoundError(
                    f"No antenna pattern files found in '{antenna_pattern_folder}'. "
                    "Expected files named 'array_factor_*.json'."
                )

            for file in filepaths_antenna_pattern:
                if not os.path.isfile(file):
                    # Additional robustness checks
                    raise FileNotFoundError(f"The specified antenna pattern file '{file}' does not exist. Please check the file path in the configuration file.")

                # Get the azimuth and elevation of that file (through the filename)
                filename = os.path.basename(file)
                azimuth, elevation, freq = parse_antenna_angles(filename)

                # Create a dictionary, linking azimuth and elevation angle to the file
                antenna_patterns[(azimuth, elevation, freq)] = file
            # Sort the antenna_patterns files
            antenna_patterns = dict(sorted(antenna_patterns.items()))
            antenna_polarizations = ast.literal_eval(config["ANTENNA"]["polarizations"]) if "ANTENNA" in config else None  # Antenna polarizations (e.g., ["LHCP", "RHCP"])
        else:
            antenna_polarizations = None
            antenna_bands = None

        footprint_dB = config["ANTENNA"].get("footprint_dB", "-3.0") if "ANTENNA" in config else None
        try:
            footprint_dB = float(footprint_dB) if footprint_dB is not None else None
        except ValueError:
            raise ValueError(f"Invalid value for 'footprint_dB': {footprint_dB}. It must be a float.") from None
        if footprint_dB is not None and footprint_dB >= 0.0:
            raise ValueError("'footprint_dB' must be below 0 dB because antenna patterns are normalized to a 0 dB peak.")
    except KeyError:
        raise KeyError("The 'antenna_filepath_0deg', 'max_steering_angle_deg', 'polarizations', 'bands' or 'antenna_gain' key is missing in the 'ANTENNA' section of the configuration file.") # noqa: B904

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
        const_string = const_string.replace("[", "").replace("]", "")
        const_string = const_string.split(",")
        const_string = [s.strip() for s in const_string]  # Remove leading/trailing whitespace
        const_string_updated = []
        gnss_names = []
        for const_str in const_string:
            # Extract GNSS constellation name ["GPS", "GLONASS", "GALILEO", "BEIDOU"]
            # Find if the name contains one of the following GPS GLONASS GALILEO BEIDOU using regular expression
            match = re.search(r"(GPS|GLONASS|GALILEO|BEIDOU)", const_str, re.IGNORECASE)
            if match:
                const_name = match.group(0)
                gnss_names.append(const_name)

                # Add TLE folder path
                const_str = os.path.join(Constants.tle_folderpath, const_str)
                const_string_updated.append(const_str)
            else:
                warnings.warn(f"Skipping unknown constellation file: {const_str}")

        gnss_constellation = {}
        for idx, (const_tle, const_name) in enumerate(zip(const_string_updated, gnss_names, strict=False)):
            gnss_constellation[const_name] = {"tle_file": const_tle, "index": idx}
    except KeyError:
        raise KeyError("The 'gnss_tle_files' key is missing in the 'GNSS CONSTELLATIONS' section of the configuration file.") # noqa: B904



    # --- [REFLECTION] (NEW SECTION) ---
    # We read the reflection type here to pass it to MovingTargetDetection later
    reflection_type = "specular"  # Default fallback
    if "REFLECTION" in config:
        reflection_type = config["REFLECTION"].get("reflection_type", "specular").strip('"').strip("'")

    # --- [LINK BUDGET] (OPTIONAL SECTION) ---
    # Defaults derived from BistaticLinkBudgetSettings so the two stay in sync.
    link_budget = {f.name: f.default for f in fields(BistaticLinkBudgetSettings)}
    if "LINK BUDGET" in config:
        section = config["LINK BUDGET"]
        for key in link_budget:
            if key in section:
                try:
                    link_budget[key] = float(section[key])
                except ValueError:
                    raise ValueError(f"Invalid value for '{key}' in [LINK BUDGET]: {section[key]}") from None


    simulation_settings = {
        #[SIMULATION SETTINGS]
        "start_time": start_time_dt,
        "step_size": step_size,
        "sim_duration": sim_duration,
        "reflection_type": reflection_type,
        #[ORBITS]
        "orbits": orbits_input,
        "No_satellites": len(orbits_input),
        #[GNSS CONSTELLATIONS]
        "GNSS_constellations": gnss_constellation,
        #[COVERAGE]
        "shapefile_path": coverage_file_path,
        "grid_size_m": grid_size_m,
        #[MOVING TARGET]
        "moving_target_speed": moving_tgt_speed,
        "start_coord_ll": start_coord_ll,
        "end_coord_ll": end_coord_ll,
        #[ANTENNA PATTERN]
        "antenna_pattern_settings": antenna_pattern_settings,
        #[ANTENNA]
        "antenna_pattern_folder": antenna_pattern_folder,
        "antenna_pattern_files": antenna_patterns,
        "max_steering_angle_deg": max_steering_angle_deg,
        "polarizations": antenna_polarizations,
        "bands": antenna_bands,
        "footprint_dB": float(footprint_dB) if footprint_dB is not None else None,
        #[LINK BUDGET]
        "link_budget": link_budget,
    }
    return config, simulation_settings

# Export the initialize function
__all__ = ["initialize"]
# __all__ = ['initialize', 'optical_calc', 'georeferencing']
