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
    =========================================================================================================
    """
    config = cp.ConfigParser()
    config.read(config_file_name)

    #[SIMULATION SETTINGS]
    try:
        start_time_dt = datetime.strptime(config["SIMULATION SETTINGS"]["startTime"], "%d.%m.%Y %H:%M:%S")  # Start time: Year, Month, Day, Hour, Minute, Second
        start_time_dt = start_time_dt.replace(tzinfo=UTC)  # Add timezone information
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
        antenna_beamwidth = ast.literal_eval(config["ANTENNA"]["antenna_beamwidth_3dB"]) if "ANTENNA" in config else None  # Antenna beamwidth -3dB in degrees
        antenna_steering_angle = ast.literal_eval(config["ANTENNA"]["max_steering_angle_deg"]) if "ANTENNA" in config else None  # Steering angle in degrees
        antenna_polarizations = ast.literal_eval(config["ANTENNA"]["polarizations"]) if "ANTENNA" in config else None  # Antenna polarizations (e.g., ["LHCP", "RHCP"])
        antenna_bands = ast.literal_eval(config["ANTENNA"]["bands"]) if "ANTENNA" in config else None  # Antenna bands (e.g., ["L1", "L2", "L5"])
        antenna_gain = ast.literal_eval(config["ANTENNA"]["antenna_gain"]) if "ANTENNA" in config else None  # Antenna gain in dB
    except KeyError:
        raise KeyError("The 'antenna_beamwidth_3dB', 'max_steering_angle_deg', 'polarizations', 'bands' or 'antenna_gain' key is missing in the 'ANTENNA' section of the configuration file.") # noqa: B904

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
        for const_str in const_string:
            # Add TLE folder path
            const_str = os.path.join(Constants.tle_folderpath, const_str)
            const_string_updated.append(const_str)
    except KeyError:
        raise KeyError("The 'gnss_tle_files' key is missing in the 'GNSS CONSTELLATIONS' section of the configuration file.") # noqa: B904

    try:
        gnss_name_str = config["GNSS CONSTELLATIONS"]["gnss_const_names"]
        gnss_names = gnss_name_str.replace("[", "").replace("]", "").split(",")
        gnss_names = [s.strip() for s in gnss_names]  # Remove leading/trailing whitespace

        gnss_constellation = {}
        for idx, (const_tle, const_name) in enumerate(zip(const_string_updated, gnss_names, strict=False)):
            gnss_constellation[const_name] = {"tle_file": const_tle, "index": idx}
    except KeyError:
        raise KeyError("The 'gnss_const_names' key is missing in the 'GNSS CONSTELLATIONS' section of the configuration file.") # noqa: B904

    simulation_settings = {
        #[SIMULATION SETTINGS]
        "start_time": start_time_dt,
        "step_size": step_size,
        "sim_duration": sim_duration,
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
        "antenna_beamwidth": antenna_beamwidth,
        "steering_angle_deg": antenna_steering_angle,
        "polarizations": antenna_polarizations,
        "bands": antenna_bands,
        "antenna_gain": antenna_gain,
    }

    return config, simulation_settings

# Export the initialize function
__all__ = ["initialize"]
# __all__ = ['initialize', 'optical_calc', 'georeferencing']
