import ast
import configparser as cp
from datetime import UTC, datetime, timezone
import json as json
import os

# import numpy              as np
# import xmltodict          as xd
from object_definitions.Constants_def import Constants

## UPDATE THE SIMULATIOR VERSION HERE ##
__version__ = "0.1.0"


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
        simulationDuration = simulation_duration_in_hours (only needed for area-coverage analysis)

        [MOVING TARGET]
        start_pos = (longitude, latitude)                (only needed for moving target analysis)
        end_pos   = (longitude, latitude)                (only needed for moving target analysis)
        speed     = Speed_in_knots                       (only needed for moving target analysis)

        [ANTENNA]
        antenna_gain_dB

        [ORBITS]
        orbit1 = (inclination, eccentricity, raan, arg_perigee, mean_anomaly, apogee)
        orbit2 = ...

        [GNSS CONSTELLATIONS]
        gnss_tle_files   = [file1.tle, file2.tle, ...]    (TLE files for GNSS constellations)
        gnss_const_names = [const1, const2, ...]          (Names of the GNSS constellations, e.g., GPS, GLONASS, GALILEO, BEIDOU)
    =========================================================================================================
    """
    config = cp.ConfigParser()
    config.read(config_file_name)

    start_time_dt = datetime.strptime(config["SIMULATION SETTINGS"]["startTime"], "%d.%m.%Y %H:%M:%S")  # Start time: Year, Month, Day, Hour, Minute, Second
    start_time_dt = start_time_dt.replace(tzinfo=UTC)  # Add timezone information
    step_size = int(config["SIMULATION SETTINGS"]["stepSize"])  # Seconds
    sim_duration = int(config["SIMULATION SETTINGS"]["simulationDuration"]) if "simulationDuration" in config["SIMULATION SETTINGS"] else None  # Hours
    grid_size_m = (
        int(config["COVERAGE"]["grid_size_meters"]) if ("COVERAGE" in config and "grid_size_meters" in config["COVERAGE"]) else None
    )  # Default grid size in meters
    moving_tgt_speed = float(config["MOVING TARGET"]["speed"]) if "MOVING TARGET" in config else None  # Speed of the moving target in km/h
    start_coord_ll = (
        ast.literal_eval(config["MOVING TARGET"]["start_pos"]) if "MOVING TARGET" in config else None
    )  # Start coordinates of the moving target in (longitude, latitude) format
    end_coord_ll = (
        ast.literal_eval(config["MOVING TARGET"]["end_pos"]) if "MOVING TARGET" in config else None
    )  # End coordinates of the moving target in (longitude, latitude) format
    antenna_gain_dB = ast.literal_eval(config["ANTENNA"]["antenna_gain_dB"]) if "ANTENNA" in config else None  # Antenna gain in dB
    coverage_file_path = (
        config["COVERAGE"]["shapefile_path"] if ("COVERAGE" in config and "shapefile_path" in config["COVERAGE"]) else None
    )  # Path to the coverage shapefile

    # Check how many orbits are specified in the configuration file
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

    const_string = config["GNSS CONSTELATIONS"]["gnss_tle_files"]
    const_string = const_string.replace("[", "").replace("]", "")
    const_string = const_string.split(",")
    const_string = [s.strip() for s in const_string]  # Remove leading/trailing whitespace
    const_string_updated = []
    for const_str in const_string:
        # Add TLE folder path
        const_str = os.path.join(Constants.tle_folderpath, const_str)
        const_string_updated.append(const_str)

    gnss_name_str = config["GNSS CONSTELATIONS"]["gnss_const_names"]
    gnss_names = gnss_name_str.replace("[", "").replace("]", "").split(",")
    gnss_names = [s.strip() for s in gnss_names]  # Remove leading/trailing whitespace

    gnss_constellation = {}
    for idx, (const_tle, const_name) in enumerate(zip(const_string_updated, gnss_names, strict=False)):
        gnss_constellation[const_name] = {"tle_file": const_tle, "index": idx}

    # Define bounding box and grid
    bounding_box = [
        float(config["BOUNDING BOX"]["min_longitude"]),
        float(config["BOUNDING BOX"]["max_longitude"]),
        float(config["BOUNDING BOX"]["min_latitude"]),
        float(config["BOUNDING BOX"]["max_latitude"]),
    ]

    simulation_settings = {
        "start_time": start_time_dt,
        "step_size": step_size,
        "sim_duration": sim_duration,
        "orbits": orbits_input,
        "bounding_box": bounding_box,
        "No_satellites": len(orbits_input),
        "GNSS_constellations": gnss_constellation,
        "shapefile_path": coverage_file_path,
        "grid_size_m": grid_size_m,
        "moving_target_speed": moving_tgt_speed,
        "start_coord_ll": start_coord_ll,
        "end_coord_ll": end_coord_ll,
        "antenna_gain_dB": antenna_gain_dB,
    }

    return config, simulation_settings


# Export the initialize function
__all__ = ["initialize"]
# __all__ = ['initialize', 'optical_calc', 'georeferencing']
