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

def initialize(config_file_name="Configuration.ini"):
    """
    =========================================================================================================
    This function reads the configuration file (default: 'Configuration.ini') and initializes the simulation settings.
    The configuration file acts as as the main input for the simulation.

    INPUT:
    - config_file_name: str, optional
        The name of the configuration file to read. Default is 'Configuration.ini'.

    OUTPUT:
    - config: ConfigParser object
        The configuration settings read from the file.
    - simulation_settings: dict
        A dictionary containing the simulation settings, including:

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

    #[ANTENNA]
    try:
        antenna_patterns = {}  # Initialize the dictionary here
        antenna_pattern_folder = config["ANTENNA"]["antenna_filepaths"] if ("ANTENNA" in config and "antenna_filepaths" in config["ANTENNA"]) else None  # Antenna beamwidth -3dB in degrees
        bands = config["ANTENNA"]["bands"] if ("ANTENNA" in config and "bands" in config["ANTENNA"]) else None  # Antenna bands
        if bands is None:
            bands = []
        else:
            # Remove square brackets, quotes and split
            antenna_bands = [band.strip().strip('"') for band in bands[1:-1].split(",")]

        # If antenna is initialzied in the ini-file:
        if "ANTENNA" in config:
            # Make sure the folder has a slash on the end
            if not antenna_pattern_folder.endswith("/"):
                antenna_pattern_folder += "/"
            if antenna_pattern_folder is not None and not os.path.isdir(antenna_pattern_folder):
                raise NotADirectoryError(f"The specified antenna pattern folder '{antenna_pattern_folder}' does not exist or is not a directory. Please check the folder path in the configuration file.")
            filepaths_antenna_pattern = []
            for file in os.listdir(antenna_pattern_folder):
                # Write all files ending with *.json in that folder into a list
                if file.endswith(".json"):
                    filepaths_antenna_pattern.append(os.path.join(antenna_pattern_folder, file))

            if filepaths_antenna_pattern not in [None, ""]:
                for file in filepaths_antenna_pattern:
                    if not os.path.isfile(file):
                        # Additional robustness checks
                        raise FileNotFoundError(f"The specified antenna pattern file '{file}' does not exist. Please check the file path in the configuration file.")
                    else:
                        # Get the azimuth and elevation of that file (through the filename)
                        filename = os.path.basename(file)
                        azimuth, elevation, freq = parse_antenna_angles(filename)

                        # Create a dictionary, linking azimuth and elevation angle to the file
                        antenna_patterns[(azimuth, elevation, freq)] = file
                # Sort the antenna_patterns files
                antenna_patterns = dict(sorted(antenna_patterns.items()))
            else:
                pass
            antenna_polarizations = ast.literal_eval(config["ANTENNA"]["polarizations"]) if "ANTENNA" in config else None  # Antenna polarizations (e.g., ["LHCP", "RHCP"])
        else:
            antenna_polarizations = None
            antenna_bands = None

        footprint_dB = config["ANTENNA"]["footprint_dB"] if ("ANTENNA" in config and "footprint_dB" in config["ANTENNA"]) else None
        try:
            footprint_dB = float(footprint_dB) if footprint_dB is not None else None
        except ValueError:
            raise ValueError(f"Invalid value for 'footprint_dB': {footprint_dB}. It must be a float.") from None
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
                print(f"Warning: Skipping unknown constellation file: {const_str}")

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
        #[ANTENNA]
        "antenna_pattern_files": antenna_patterns,
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
