import os
import numpy           as np
import cartopy.crs     as ccrs
import cartopy.feature as cfeature
import geopandas       as gpd

import matplotlib
matplotlib.use('TkAgg', force=True)
from __init__               import initialize
from matplotlib             import pyplot as plt
from astropy                import units  as u
from skyfield.api           import load
from datetime               import datetime
from tqdm                   import tqdm
from shapely.geometry       import Polygon

from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Satellite_def         import Satellite
from object_definitions.SatConstellation_def  import SatelliteConstellation
from object_definitions.Coverage_def          import Coverage
from object_definitions.Static_classes_def    import Plotting

###########################################################################################################################
###########################################################################################################################
def load_gnss_constellations(gnss_costl_configs):
    '''
    Load GNSS constellation configurations.
    '''
    gnss_constl_dict = {}
    for name, tle_file_dict in gnss_costl_configs.items():
        print(f"GNSS constellation: \"{name}\" loaded from: {tle_file_dict['tle_file']}.")
        gnss_constl_dict[name] = SatelliteConstellation(
            tle_file=tle_file_dict['tle_file'], 
            satConfigName=name
        )
    return gnss_constl_dict

def propagate_all_orbits(sat_const, gnss_constl_dict, start_time, duration, step_size): # TODO move to functions.py
    '''
    Propagate orbits for satellite constellation and GNSS constellations.
    '''
    # Propagate the satellite constellation (GNSS-R / LEO)
    print(f"Propagating satellite constellation: \"{sat_const.satConfigName}\"")
    sat_const.propagate_constellation(start_time, duration, step_size)

    # Propagate all GNSS constellations
    for name, gnss_const in gnss_constl_dict.items():
        print(f"Propagating GNSS constellation: \"{name}\"")
        gnss_const.propagate_constellation(start_time, duration, step_size)

def calculate_reflection_points(sat_const, gnss_constl_dict, times, earth_model='spherical', angle=45, verbose=False): # TODO move to functions.py or Reflection_def.py or Coverage.py??
    '''
    Calculate reflection points between satellites.
    '''
    reflection_dict = {}

    # Calculate the total number of calculations to be performed
    total_computations = 0
    for gnss_name, gnss_const in gnss_constl_dict.items():
        total_computations += len(sat_const.satellites) * len(gnss_const.satellites)

    with tqdm(total=total_computations, desc="Reflection calculations", unit="reflection") as pbar:
        for gnss_name, gnss_const in gnss_constl_dict.items():
            if verbose:
                print(f'\nCalculating reflections for GNSS constellation \"{gnss_name}\"')
            reflections_vec = []

            for sat_no, sat in enumerate(sat_const.satellites):
                # Update status but don't advance the progress bar yet
                pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {gnss_name}")

                for i, gnss_sat in enumerate(gnss_const.satellites):
                    if i % 5 == 0:
                        pbar.set_description(
                            f"GNSS-R Sat-{sat_no + 1} with {gnss_name} [{i}/{len(gnss_const.satellites)}]"
                        )
                    # Perform the reflection calculation
                    refl_gnss = Reflection(
                        gnss_sat.satID, sat.satID, 
                        earth_model=earth_model, angle=angle, 
                        gnss_const=gnss_name
                    )
                    refl_gnss.compute_reflection_coordinates(gnss_sat, sat, times)
                    reflections_vec.append(refl_gnss)

                    # Update progress bar after each reflection calculation
                    pbar.update(1)

            reflection_dict[gnss_const.satConfigName] = reflections_vec

    return reflection_dict

def plot_reflection_points(sat_const, gnss_constl_dict, reflection_dict): # TODO move to Static Functons class ??
    '''
    Plot reflection points for all GNSS constellations.
    '''
    for gnss_costl_name, refl_vec in reflection_dict.items():
        plot_name          = f"Reflection_points_{gnss_costl_name}.png"
        gnss_constellation = gnss_constl_dict[gnss_costl_name]
        Plotting.plot_reflection_traces_gnss_const(
            sat_const, gnss_constellation, refl_vec, file_name=plot_name
        )

    const_names_str = '_'.join(gnss_constl_dict.keys())
    Plotting.plot_reflection_traces_total(
        sat_const, gnss_constl_dict, reflection_dict, 
        file_name=f"Reflection_traces_{const_names_str}.png"
    )

def calculate_statistics(coverage, grid_gdf):
    '''
    Calculate statistics from coverage data.
    '''
    # Average signal power
    all_powers    = np.concatenate([item.power for item in coverage.coverage_vector])
    average_power = np.mean(all_powers)
    print(f"Average Power: {average_power}")

    # Coverage percentage
    intersection_count  = {idx: len(coverage.coverage_vector[i].times) 
                          for i, idx in enumerate(grid_gdf.index)}
    non_zero_count      = sum(1 for value in intersection_count.values() if value != 0)
    coverage_percentage = non_zero_count / len(intersection_count) * 100
    print(f"Coverage percentage: {coverage_percentage}")

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
    gnss_constl_dict = load_gnss_constellations(sim_settings['GNSS_constellations'])

    ####################################
    # 2. Create satellite constellation
    ####################################
    sat_array = [Satellite(i, orbit_n=sim_settings['orbits'][i]) 
                for i in range(sim_settings['No_satellites'])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_TLE()

    ####################################
    # 3. Setup time parameters (skyfield)
    ####################################
    ts          = load.timescale()                                                                      # This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
    duration    = int(sim_settings['sim_duration'] * Constants.Sec_per_hour)                            # Duration in seconds
    num_steps   = (duration) // sim_settings['step_size']                                               # derive no. of steps from duration and step size
    # Generate the time series (skyfield timescale object) for the simulation
    # The time series is generated from the start time to the end time with a step size of sim_settings['step_size']
    times_skyfield     = ts.utc(
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
    # 4. Propagate orbits
    ####################################
    propagate_all_orbits(sat_const, gnss_constl_dict, sim_settings['start_time'], 
                         duration, sim_settings['step_size'])

    ####################################
    # 5. Calculate sunlight time
    ####################################
    sunlit = sat_const.satellites[0].calculate_sunlight_time(times_skyfield)
    print("Not_sunlit count for satellite 0: ", np.sum(sunlit == False))
    print("Sunlit count for satellite 0: ",     np.sum(sunlit == True))

    ####################################
    # 6. Calculate reflection points
    ####################################
    reflection_dict = calculate_reflection_points(sat_const, gnss_constl_dict, times_datetime, verbose=True) # TODO start cleaning from here

    ####################################
    # 7. Plot reflection points
    ####################################
    plot_reflection_points(sat_const, gnss_constl_dict, reflection_dict)

    ####################################
    # 8. Calculate and plot coverage
    ####################################
    coverage  = Coverage(sim_settings['shapefile_path'], sim_settings['bounding_box'], grid_size_deg=1)
    coverage.calculate_coverage(reflection_dict['GPS'], times_datetime)

    Plotting.plot_coverage_shapefile(coverage, reflection_dict['GPS'], sim_settings['bounding_box'], file_name="Coverage.png")
    Plotting.plot_coverage_gridded_shapefile(coverage, reflection_dict['GPS'], file_name="Coverage_gridded.png")

    ####################################
    # 9. Calculate statistics
    ####################################
    calculate_statistics(coverage, grid_gdf=gpd.read_file(os.path.join(os.getcwd(), 'data', 'iho', 'fishnet.shp')))

    ####################################
    # 10. Plot orbits
    ####################################
#    satellites_LEO_sk = [sat_const.satellites[0].skf, sat_const.satellites[1].skf]
#    plot_orbits(satellites_LEO_sk, sat_const.satellites[0].time_prop_vec)

    ####################################################################################################################################################################################
    ##########################################
    ## Plot orbits                          ##
    ##########################################
    # Create a 3D plot
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    # Plot the Earth as a sphere
    u  = np.linspace(0, 2 * np.pi, 100)
    v  = np.linspace(0, np.pi, 100)
    x  = Constants.R_earth*1e-3 * np.outer(np.cos(u), np.sin(v))
    y  = Constants.R_earth*1e-3 * np.outer(np.sin(u), np.sin(v))
    z  = Constants.R_earth*1e-3 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the satellite trajectory (line connecting the points)
    leo_positions_ECI_1 = np.array([satellites_LEO_sk[0].at(t).position.m for t in sat_const.satellites[0].time_prop_vec])
    leo_positions_ECI_1 = np.transpose(leo_positions_ECI_1/1e3)
    leo_positions_ECI_2 = np.array([satellites_LEO_sk[1].at(t).position.m for t in sat_const.satellites[0].time_prop_vec])
    leo_positions_ECI_2 = np.transpose(leo_positions_ECI_2/1e3)

    ax.plot(leo_positions_ECI_1[0], leo_positions_ECI_1[1], leo_positions_ECI_1[2], label="Satellite 1 Path", color='r')

    # Plot the satellite trajectory for the second satellite
    ax.plot(leo_positions_ECI_2[0], leo_positions_ECI_2[1], leo_positions_ECI_2[2], label="Satellite 2 Path", color='b')
    plt.savefig('Orbits.png')
    #plt.show()




if  __name__ == "__main__":
    config, settings = initialize()
    main_calculate_coverage_norwegian_coast(settings)
