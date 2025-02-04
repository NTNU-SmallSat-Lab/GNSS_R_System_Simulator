import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

import matplotlib
#matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt
#from astropy import units as u
from skyfield.api import load
from datetime import timezone, datetime
from tqdm import tqdm
#from shapely.geometry import Polygon, LineString
from scipy.spatial.distance import cdist

# Import your custom object definitions
from object_definitions.Reflection_def import Reflection
from object_definitions.Constants_def import Constants
from object_definitions.Satellite_def import Satellite
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Coverage_def import Coverage #, Coverage_OLD
from object_definitions.Static_classes_def import Plotting

def main(start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box):
    print("Start Time:", start_time_dt)

    # Define bounding box and grid
    min_lon, max_lon = -17, 30
    min_lat, max_lat =  56, 80
    bounding_box = [min_lon, max_lon, min_lat, max_lat]
    #############################################################################################################################
    #################################################### - INITIALIZATIONS - ####################################################
    #############################################################################################################################
    Nsat        = len(orbits_input)                                                     # Number of satellites
    ts          = load.timescale()                                                      # This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
    duration    = int(sim_duration * Constants.Sep_per_hour)                            # Duration in seconds
    num_steps   = (duration) // step_size                                                 # derive no. of steps from duration and step size
    # Generate the time series starting at 3:00 PM on October 23, 2024
    times_1     = ts.utc(start_time_dt.year, start_time_dt.month, start_time_dt.day, start_time_dt.hour, start_time_dt.minute, range(0, num_steps * step_size, step_size))

    # Need to create another datime obect, times_2, to run the latitude, longitude conversion
    times_2 = [time.utc_datetime() for time in times_1]  # Call the method on each Time object

    # Load the GNSS constellation
    GNSS_const = SatelliteConstellation(tle_file=gnss_tle_file_path)
    print(f"Loaded {GNSS_const.no_satellites} GNSS satellites from {gnss_tle_file_path}.")
    latest_epoch = None
    # Iterate through satellites and save latest time for integration
    for sat in GNSS_const.satellites:
        current_epoch = datetime.strptime(sat.skf.epoch.utc_jpl(), "A.D. %Y-%b-%d %H:%M:%S.%f UTC")

        # Update the latest epoch if this one is newer
        if latest_epoch is None or current_epoch > latest_epoch:
            latest_epoch = current_epoch

    # Print the latest epoch
    if latest_epoch is not None:
        print(f"Latest Epoch: {latest_epoch.strftime('%Y-%m-%d %H:%M:%S.%f')}")

    # Create TLE file for the constellation
    sat_array = []
    for i in range(0, Nsat):
        sat_array.append(Satellite(i, orbit_n=orbits_input[i]))
    sat_const = SatelliteConstellation(satConfigName="Constellation", satArray=sat_array)
    sat_const.generate_constellation_TLE()

    #############################################################################################################################
    ################################## - PROPAGATE ORBITS OF "GNSS" AND "GNSS-R" SATELLITES - ###################################
    #############################################################################################################################
    sat_const.propagate_constellation(start_time_dt, duration, step_size)
    GNSS_const.propagate_constellation(start_time_dt, duration, step_size)


    #############################################################################################################################
    ######################################## - CALCULATE SUNLITE TIME - #########################################################
    #############################################################################################################################
    #TODO improve the code below
    planets = load('de421.bsp')
    earth   = planets['earth']
    sun     = planets['sun']

  
    #############################################################################################################################
    ######################################### - CALCULATE "POINT" REFLECTION POINTS - ###########################################
    #############################################################################################################################
    sat_no         = 0
    reflection_vec = []
    for sat in sat_const.satellites:
        print("Calculating refl. for GNSS-R Sat-", sat_no + 1)
        for gnss_sat in tqdm(GNSS_const.satellites):
            refl_gps = Reflection(gnss_sat.satID, sat.satID, earth_model='spherical', angle=45, gnss_const = 'GPS')
            refl_gps.compute_reflection_coordinates(gnss_sat, sat, times_2)                                                             #sat.time_prop_vec.utc_datetime()
            reflection_vec.append(refl_gps)
            # Add the reflection points to the plot
        sat_no += 1

    #satellites_LEO_sk = [sat_const.satellites[0].skf, sat_const.satellites[1].skf]

    #############################################################################################################################
    ######################################### - PLOTTING REFLECTION POINTS - ####################################################
    #############################################################################################################################
    Plotting.plot_reflection_traces(sat_const, GNSS_const, reflection_vec, "Reflection_traces.png")




    #############################################################################################################################
    ########################################## - CALCULATE COVERAGE - ###########################################################
    #############################################################################################################################
    output_shapefile = os.path.join(os.getcwd(), 'iho', 'fishnet.shp')
    coverage         = Coverage(output_shapefile)
    coverage.calculate_coverage(bounding_box, reflection_vec, times_2)
    #Plotting.plot_coverage(coverage, reflection_vec, bounding_box, "Coverage.png")
    grid_gdf         = gpd.read_file(output_shapefile)

    ################################################################
    ######################## PLOTTING ##############################
    ################################################################
    intersection_count = {idx: 0 for idx in grid_gdf.index}


    for i in range(len(grid_gdf)):
        intersection_count[i] += len(coverage.coverage_vector[i].times)

    '''
    # Iterate through coverage_vec to evaluate the times attribute and count intersections
    for coverage in coverage_vec:
        # If there are any times recorded, update the intersection count for the corresponding grid cell
        # Assuming coverage_vec corresponds to the same index as grid_gdf
        grid_idx = coverage_vec.index(coverage)  # Get the index of the current coverage in coverage_vec
        intersection_count[grid_idx] += len(coverage.times)  # Increase count based on the number of times recorded
    '''
    # Define color based on the count
    def get_color(count):
        if count == 1:
            return 'red', 1  # Color for 1-time intersection
        elif count > 1:
            return 'blue', 0.5  # Color for more than 1-time intersection
        else:
            return 'none', 0  # If the cell isn't intersected at all



    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    #for i in range(len(reflection_vec)):
    #    lons, lats = fix_longitude_wraparound(reflection_vec[i].reflection_points['reflLL'][:,1], reflection_vec[i].reflection_points['reflLL'][:,0])
    #    ax.plot(lons, lats,'red')
    ax.set_global()
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])  # Add padding to the extent
    grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")

    # Loop through each grid cell to plot with the correct color
    for idx in grid_gdf.index:
        # Get the intersection count for the grid cell
        count = intersection_count[idx]
        
        # Get the color for the current grid cell
        color, alpha_vec = get_color(count)
        
        coverage.coverage_vector[idx].shape.plot(ax=ax, edgecolor='red', facecolor=color, alpha = 0.3)


    count_total = 0
    # Plot the blue grid cells (those intersected more than once) on top of the red ones
    for idx in grid_gdf.index:
        count = intersection_count[idx]
        color, alpha_vec = get_color(count)
        count_total = count_total + count
        
        if color == 'blue':  # Only plot blue cells (more than 1-time intersection)
            coverage.coverage_vector[idx].shape.plot(ax=ax, edgecolor='blue', facecolor=color, alpha = 0.3)

    for i in range(len(reflection_vec)):
        ax.plot(reflection_vec[i].refl_LL_vec[:, 1], reflection_vec[i].refl_LL_vec[:, 0], color='red', transform=ccrs.PlateCarree())
    plt.savefig("Coverage2.png", dpi = 300)


    ####################################################
    ########### Average  signal"Power" ###########
    all_powers = np.concatenate([item.power for item in coverage.coverage_vector])  # Flatten all power values
    average_power = np.mean(all_powers)  # Calculate the average
    print("Average Power:", average_power)

    ########### Coverage ###########
    non_zero_count = sum(1 for value in intersection_count.values() if value != 0)
    coverage_percentage = non_zero_count/len(intersection_count)*100
    print("Coverage percentage:", coverage_percentage)

    ########### Sunlight ###########
    # Compute sunlight or not
    sunlight_percentage = np.zeros(Nsat)
    for i in range(0, Nsat):
        sunlit      = sat_const.satellites[i].skf.at(times_1).is_sunlit(planets)
        false_count = np.sum(sunlit == False)
        true_count  = np.sum(sunlit == True)
        sunlight_percentage[i] = true_count/(true_count + false_count)*100
        print("Time in sunlight in percentage", sunlight_percentage[i])
    sunlight_percentage_avg = np.mean(sunlight_percentage)
    print("Time in sunlight in percentage:", sunlight_percentage_avg)

    ########### Avg passes per cell ###########
    count_avg = count_total/len(grid_gdf)
    print("Avg passes:", count_avg)

    print("Simulation complete.")


    return average_power, coverage_percentage, sunlight_percentage_avg, count_avg, coverage

def save_results_to_file(f, start_time_dt, step_size, sim_duration, bounding_box, orbits_input, gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage, count_avg):
    #with open(filename, 'a') as f:
        # Writing input parameters
    orbits_input_c = [[float(val) for val in orbit] for orbit in orbits_input]

    f.write("Input Parameters:")
    f.write(f"{start_time_dt},")
    f.write(f"{step_size},")
    f.write(f"{sim_duration},")
    #f.write(f"{bounding_box}")
    f.write(f"{gnss_tle_file_path},")
    # Remove the datetime object from each orbit (assuming datetime is the last element in each list)
    # Write the cleaned orbits input to the file (now excluding the datetime)
    f.write(f"{orbits_input_c},")


    # Writing results
    #f.write("Results:")
    f.write(f"{average_power},")
    f.write(f"{coverage_percentage},")
    f.write(f"{sunlight_percentage},")
    f.write(f"{count_avg}")
    f.write("-----\n")


def run_simulation(start_time_dt, step_size, sim_duration, bounding_box, gnss_tle_file_path, orbits_input):
    """Run simulation using the given parameters and save results."""
    # Convert start_time to datetime
    #start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    
    # Run the main simulation
    '''
    average_power, coverage_percentage, sunlight_percentage, count_avg = main(
        start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box
    )
    '''
    # Save results to file
    #output_file_path = "simulation_results.txt"
    #save_results_to_file(output_file_path, start_time_dt, step_size, sim_duration, bounding_box, orbits_input, gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage, count_avg)

    #print(f"Results and input saved to {output_file_path}")
    return 
'''
if __name__ == "__main__":
    # Define default inputs

    start_time_dt = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)
    step_size = 30  # Seconds
    sim_duration = 4  # Hours

    bounding_box = [-17, 30, 56, 80]  # [min_lon, max_lon, min_lat, max_lat]
    gnss_tle_file_path = os.path.join("TLE_EPHEMIRIS_files", "GPS_operational.tle")


    orbits_input = [
        [99.1, 0.00001, 0, 0, 0, 7300e3, start_time_dt],
        [99.1, 0.00001, 90.0, 0, 0, 7300e3, start_time_dt],
    ]




    # Run main with the default parameters
    average_power, coverage_percentage, sunlight_percentage, count_avg = main(start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box)

    # File to save the results
    output_file_path = "simulation_results.txt"

    # Save the input parameters and results to a text file
    save_results_to_file(output_file_path, start_time_dt, step_size, sim_duration, bounding_box, orbits_input, gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage)

    print(f"Results and input saved to {output_file_path}")
'''