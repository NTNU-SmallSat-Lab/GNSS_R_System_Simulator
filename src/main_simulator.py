import os
import numpy               as np
import functions.functions as fn

import matplotlib
matplotlib.use('TkAgg', force=True)
from __init__               import initialize
from skyfield.api           import load

from object_definitions.Constants_def         import Constants
from object_definitions.Satellite_def         import Satellite
from object_definitions.SatConstellation_def  import SatelliteConstellation
from object_definitions.Coverage_def          import Coverage
from object_definitions.Static_classes_def    import Plotting

#############################################################################################################################
##################################################### - MAIN FUNCTION - #####################################################
#############################################################################################################################
def main_calculate_coverage_norwegian_coast(sim_settings):
    '''
    Run satellite simulation to calculate the coverage of GNSS-R satellites over the Norwegian coast.
    This function initializes the simulation, propagates orbits, calculates reflection points,
    and generates coverage statistics.
    The simulation is based on the provided simulation settings which are loaded from a configuration file.
    (Configuration.ini)

    INPUT:
        sim_settings: Dictionary containing simulation parameters
    '''
    print("Simulation start Time:", sim_settings['start_time'])

    ####################################
    # 1. Initialize GNSS constellations
    ####################################
    gnss_constl_dict = fn.load_gnss_constellation_dict(sim_settings['GNSS_constellations'])

    ####################################
    # 2. Create satellite constellation
    ####################################
    sat_array = [Satellite(i, orbit_n=sim_settings['orbits'][i]) 
                for i in range(sim_settings['No_satellites'])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_TLE()

    ####################################
    # 3. Propagate orbits
    ####################################
    fn.propagate_all_orbits(sat_const, gnss_constl_dict, sim_settings['start_time'], 
                            sim_settings['sim_duration'], sim_settings['step_size'])

    ####################################
    # 4. Setup time parameters (skyfield)
    #################################### # TODO move this to a function (e.g. in propagate orbits)
    duration    = int(sim_settings['sim_duration'] * Constants.Sec_per_hour)                            # Duration in seconds
    num_steps   = (duration) // sim_settings['step_size']                                               # derive no. of steps from duration and step size
    # Generate the time series (skyfield timescale object) for the simulation
    # The time series is generated from the start time to the end time with a step size of sim_settings['step_size']
    times_skyfield     = load.timescale().utc(                                                          # This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
        sim_settings['start_time'].year,
        sim_settings['start_time'].month,
        sim_settings['start_time'].day,
        sim_settings['start_time'].hour,
        sim_settings['start_time'].minute,
        range(0, num_steps * sim_settings['step_size'],sim_settings['step_size'])
    )
    # Convert the skyfield timescale to a list of UTC datetime objects (one datetime object for each time step)
    times_datetime = [time.utc_datetime() for time in times_skyfield]

    ####################################
    # 5. Calculate sunlight time
    #################################### # TODO improve this also
    satellite_idx = 0
    sunlit = sat_const.satellites[satellite_idx].calculate_sunlight_time(times_skyfield)
    print(f"Not_sunlit count for satellite {satellite_idx}: ", np.sum(sunlit == False))
    print(f"Sunlit count for satellite {satellite_idx}: ",     np.sum(sunlit == True))

    ####################################
    # 6. Calculate reflection points and coverage
    ####################################
    coverage = Coverage(sat_const, gnss_constl_dict, sim_settings['shapefile_path'], sim_settings['bounding_box'], grid_size_deg=1)
    coverage.calculate_reflection_points_and_coverage(times_datetime, verbose=True)

    ####################################
    # 7. Plot reflection points
    ####################################
    coverage.plot_reflection_points()
    coverage.plot_coverage_shapefile(gnss_constellation="GPS", file_name="Coverage.png")
    coverage.plot_coverage_shapefile(file_name="Coverage_both.png")
    coverage.plot_coverage_gridded_shapefile(gnss_constellation="GPS", file_name="Coverage_grid_GPS.png")
    coverage.plot_coverage_gridded_shapefile(file_name="Coverage_grid_GPS_GALIEO.png")

    ####################################
    # 9. Calculate statistics
    ####################################
    coverage.calculate_statistics()

    ####################################
    # 10. Plot orbits
    ####################################
    Plotting.plot_orbits(sat_const)

if  __name__ == "__main__":
    config, settings = initialize()
    main_calculate_coverage_norwegian_coast(settings)
