import ast, os
import json               as json
import configparser       as cp
import numpy              as np
import xmltodict          as xd
from object_definitions.Constants_def  import Constants
from datetime                          import timezone, datetime

## UPDATE THE SIMULATIOR VERSION HERE ##
__version__ = "0.1.0"

def initialize():
    config = cp.ConfigParser()
    config.read('Configuration.ini')

    start_time_dt          = datetime.strptime(config['SIMULATION SETTINGS']['startTime'], '%d.%m.%Y %H:%M:%S') # Start time: Year, Month, Day, Hour, Minute, Second
    start_time_dt          = start_time_dt.replace(tzinfo=timezone.utc)                                         # Add timezone information
    step_size              = int(config['SIMULATION SETTINGS']['stepSize'])                                     # Seconds
    sim_duration           = int(config['SIMULATION SETTINGS']['simulationDuration'])                           # Hours

    # Check how many orbits are specified in the configuration file
    orbits_input = []
    orbit_no = 1
    orbit_str = 'orbit'+ str(orbit_no)
    while orbit_str in config['ORBITS']:
        orb_str        = config['ORBITS'][orbit_str]
        orbit_values   = ast.literal_eval(orb_str)
        orbits_input.append([orbit_values[0], orbit_values[1], orbit_values[2], orbit_values[3], orbit_values[4], orbit_values[5], start_time_dt])
        orbit_no += 1
        orbit_str = 'orbit'+ str(orbit_no)

    const_string  = config['GNSS CONSTELATIONS']['gnss_tle_files']
    const_string  = const_string.replace('[', '').replace(']', '')
    const_string  = const_string.split(',')
    const_string  = [s.strip() for s in const_string]  # Remove leading/trailing whitespace
    const_string_updated = []
    for const_str in const_string:
        # Add TLE folder path
        const_str = os.path.join(Constants.tle_folderpath, const_str)
        const_string_updated.append(const_str)

    gnss_name_str = config['GNSS CONSTELATIONS']['gnss_const_names']
    gnss_names = gnss_name_str.replace('[', '').replace(']', '').split(',')
    gnss_names = [s.strip() for s in gnss_names]  # Remove leading/trailing whitespace

    gnss_constellation = {}
    for idx, (const_tle, const_name) in enumerate(zip(const_string_updated, gnss_names)):
        gnss_constellation[const_name] = {
            'tle_file': const_tle,
            'index':    idx
        }

    # Define bounding box and grid
    bounding_box = [float(config['BOUNDING BOX']['min_longitude']), 
                    float(config['BOUNDING BOX']['max_longitude']), 
                    float(config['BOUNDING BOX']['min_latitude']), 
                    float(config['BOUNDING BOX']['max_latitude'])]

    simulation_settings = {'start_time':           start_time_dt,
                           'step_size':            step_size,
                           'sim_duration':         sim_duration,
                           'orbits':               orbits_input,
                           'bounding_box':         bounding_box,
                           'No_satellites':        len(orbits_input),
                           'GNSS_constellations':  gnss_constellation,
                           'shapefile_path':       config['COVERAGE']['shapefile_path']}

    return config, simulation_settings

# Export the initialize function
__all__ = ['initialize']
#__all__ = ['initialize', 'optical_calc', 'georeferencing']