
import os
import numpy           as np
import cartopy.crs     as ccrs
import cartopy.feature as cfeature
import geopandas       as gpd

import matplotlib
matplotlib.use('TkAgg', force=True)
from matplotlib             import pyplot as plt
from astropy                import units  as u
from skyfield.api           import load
from datetime               import timezone, datetime
from tqdm                   import tqdm
from shapely.geometry       import Polygon, LineString
from scipy.spatial.distance import cdist

##
from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Satellite_def         import Satellite
from object_definitions.SatConstellation_def  import SatelliteConstellation
from object_definitions.Coverage_def          import Coverage, Coverage_OLD
from object_definitions.Static_classes_def    import Plotting

#############################################################################################################################
######################################################## - INPUTS - #########################################################
#############################################################################################################################
start_time_dt          = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)              # Start time: Year, Month, Day, Hour, Minute, Second
step_size              = 30                                                                   # Seconds
sim_duration           = 4                                                                    # Hours

orbits_input           = [[99.1, 0.00001, 0,   0, 0, 7300e3, start_time_dt],                     # [inclination [°], eccentricity [-], RAAN [°], Argument of perigee [°], Mean anomaly [°], Semi-major axis [m], start_time_dt [datetime] (optional)]
                [99.1, 0.00001, 90.0, 0, 0, 7300e3, start_time_dt]]

gnss_tle_file_path     = os.path.join("TLE_EPHEMIRIS_files", "GPS_operational.tle") # Replace with your file's name 

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

# Compute sunlight or not
sunlit      = sat_const.satellites[0].skf.at(times_1).is_sunlit(planets)
false_count = np.sum(sunlit == False)
true_count  = np.sum(sunlit == True)
print(false_count)
print(true_count)

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

satellites_LEO_sk = [sat_const.satellites[0].skf, sat_const.satellites[1].skf]

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

# Plot the blue grid cells (those intersected more than once) on top of the red ones
for idx in grid_gdf.index:
    count = intersection_count[idx]
    color, alpha_vec = get_color(count)
    
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
print("Coverage percentage:", non_zero_count/len(intersection_count)*100)





'''
# Initialize coverage vector for storing coverage over each cell in the fishnet
coverage_vec = []
for idx in range(len(grid_gdf)):
    coverage_old = Coverage_OLD()
    # Compute the centroid of the current grid cell
    centroid = grid_gdf.loc[idx].geometry.centroid
    # Set the position attribute of the Coverage instance
    coverage_old.position = (centroid.x, centroid.y)
    # Append the instance to the coverage_vec list
    coverage_vec.append(coverage_old)
    coverage_old.shape = grid_gdf.loc[[idx]].boundary

# Initialize figure and axis with projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# Define bounding box and grid 
ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent

# Plot the grid as the background (light gray color)
grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")

# Loop through the reflection vectors #TODO
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()
ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent
grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")
for i in range(len(reflection_vec)):#range(11):
    refl_temp     = reflection_vec[i].refl_LL_vec
    points_in_box = []
    check         = False

    points_in_box = []
    # Filter points within the bounding box and ignore points outside it
    for lat, lon in refl_temp:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            points_in_box.append((lat, lon))
            check = True
        else:
            points_in_box.append((np.nan, np.nan))

    points_in_box_1 = np.array(points_in_box)
    valid           = ~np.isnan(points_in_box_1[:, 0])  # Check latitude for NaN
    clusters        = np.split(np.arange(len(points_in_box_1)), np.where(~valid)[0])  # Split indices by NaN gaps
    for cluster in clusters:
        intersecting_rectangles = []
        # Skip clusters with no valid data
        if len(cluster) <= 1: #not valid[cluster].any() &
            continue
        print(cluster)
        cluster = cluster[1::]
        if len(cluster) <= 1: #not valid[cluster].any() &
            continue        
        # Extract valid times and positions for this cluster
        valid_cluster  = cluster[valid[cluster]]
        valid_position = points_in_box_1[cluster]
        #print(valid_position)
        line = LineString(zip(valid_position[:, 1], valid_position[:, 0]))
        for idx, polygon in grid_gdf.iterrows():
            # Check if the polygon intersects with the line
            #print(polygon)
            intersection = polygon['geometry'].intersection(line)
            if not intersection.is_empty:
                intersecting_rectangles.append(idx)
        print(intersecting_rectangles) 
        ax.plot(valid_position[:, 1], valid_position[:, 0], color='red', transform=ccrs.PlateCarree()) # TODO add to plotting function
        if len(intersecting_rectangles) > 0:
            intersecting_gdf = grid_gdf.loc[intersecting_rectangles]
            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")  # TODO add to plotting function
        
            # Save results
            grid_gdf.loc[intersecting_rectangles]
            valid_position_clean     = valid_position[~np.isnan(valid_position).any(axis=1)]
            valid_positions_flipped  = valid_position_clean[:, [1, 0]]
            centroids_array          = np.array([(centroid.x, centroid.y) for centroid in grid_gdf.loc[intersecting_rectangles].centroid])
            distances                = cdist(centroids_array, valid_positions_flipped)
            nearest_indices          = np.argmin(distances, axis=1)
            nearest_points           = valid_positions_flipped[nearest_indices]
            times_at_nearest_points  = [times_2[i] for i in cluster[nearest_indices]]
            angle_at_nearest_points  = reflection_vec[i].refl_ang_vec[cluster[nearest_indices]]
            power_at_nearest_points  = reflection_vec[i].power[cluster[nearest_indices]]

            # Save the nearest points
            for rect_idx, time, angle, power in zip(intersecting_rectangles, times_at_nearest_points, angle_at_nearest_points, power_at_nearest_points):
                #print(f"Rectangle Index: {rect_idx}")
                #print(f"Nearest Time: {time}")
                # Assign time to coverage_vec
                coverage_vec[rect_idx].times.append(time)
                coverage_vec[rect_idx].tx_satID.append(reflection_vec[i].tx_satID)
                coverage_vec[rect_idx].rx_satID.append(reflection_vec[i].rx_satID)
                coverage_vec[rect_idx].angle.append(angle)
                coverage_vec[rect_idx].power.append(power) 

#plt.savefig("Coverage1.png", dpi = 300) #TODO add to plotting function


'''






###########################################################################################################################
# Function to create fishnet grid
def create_coverage(grid_size):
    # Define the path to your shapefile (iho.shp)
    cwd = os.getcwd()
    shapefile = os.path.join(cwd, 'iho', 'iho.shp')

    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile)

    # Get the bounds of the shapefile (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define the grid size (e.g., 0.1 degrees for latitude/longitude)
    grid_size = 1

    # Generate the fishnet grid (list of polygons)
    grid_polygons = []
    for x0 in np.arange(minx, maxx, grid_size):
        for y0 in np.arange(miny, maxy, grid_size):
            x1 = x0 + grid_size
            y1 = y0 + grid_size
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # Create a GeoDataFrame for the fishnet
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs=gdf.crs)
    # Filter out grid cells that do not intersect with the IHO shapefile
    grid_gdf = grid_gdf[grid_gdf.intersects(gdf.unary_union)]

    # Save the grid as a shapefile
    output_shapefile = os.path.join(cwd, 'iho', 'fishnet.shp')
    grid_gdf.to_file(output_shapefile)

    print(f"Fishnet shapefile saved as: {output_shapefile}")


    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original shapefile
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5, label="Original Shapefile")

    # Plot the fishnet grid
    grid_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Fishnet Grid")

    # Add title and legend
    ax.set_title('Fishnet Grid Overlayed on Original Shapefile')
    plt.savefig("rectangles_from_grid_points.png")

####################################################################################################################################################################################
##########################################
## Plot orbits                          ##
##########################################
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
