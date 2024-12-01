from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time, TimeDelta
#from poliastro.bodies import Earth
#from poliastro.twobody import Orbit as Orbit_poliastro
#from poliastro.plotting.static import StaticOrbitPlotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from functions import compute_reflection
from astropy import coordinates as coord
import pymap3d 
from skyfield.api import load, wgs84
from skyfield.sgp4lib import EarthSatellite
from datetime import timedelta, timezone, datetime
from skyfield.framelib import itrs
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D


##
from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Orbit_def             import Orbit
from object_definitions.Satellite_def         import Satellite
from object_definitions.GnssConstellation_def import GNSSConstellation
from object_definitions.ReadWriteSatData      import ReadWriteSatData
from object_definitions.SatConstellation_def  import SatelliteConfiguration

import os
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load




###########################################
######### Create Coverage class #########
class Coverage:
    def __init__(self, earth_model=None):

            self.position       = []      
            self.tx_satID          = []
            self.rx_satID          = []
            self.times = []
            self.angle = []
            self.shape = []
            self.power = []
#########################################




#######################################################################################################
###################################### NEW ADDITION!! 26.11 PUSH ######################################

def fix_longitude_wraparound(lons, lats):
    # Find where the difference between consecutive longitudes exceeds 180 degrees -> avoid plotting lines from -179 to 179 deg longitude
    wraparound = np.abs(np.diff(lons)) > 180
    # Replace the longitude and latitude at the wraparound point with NaN
    lons[1:][wraparound] = np.nan
    lats[1:][wraparound] = np.nan
    return lons, lats

#######################################################################################################





#########################################

################## INPUTS!!! ##################

# Receiver satellite
#orbit_input = [[99.1, 0.00001, 0, 0, 0,  7300e3],
#               [99.1, 0.00001, 90, 0, 0,  7300e3]]
#Nsat = 2

# Receiver satellite
orbit_input = [[99.1, 0.00001, 0, 0, 0,  7000e3],
               [99.1, 0.00001, 90, 0, 0,  7000e3]]
Nsat = 2

#Nsat = 1
#orbit_input = [70, 0.00001, 90, 0, 0,  7300e3]

# GNSS satellites
gnss_constellation = GNSSConstellation('GPS')
tle_file = "TLE_EPHEMIRIS_files/GPS_operational.tle"  # Replace with your file's name
#tle_file = "TLE_EPHEMIRIS_files/GALILEO.tle"  # Replace with your file's name

# Define the step size and duration
step_size = 30  # Seconds
hours = 8
duration = int(hours * 3600)  # Total duration in seconds (6 hours)
#duration = int(2*np.pi*np.sqrt((orbit_input[1][5]/1e3)**3/(398600.4)))



######################################

satellites_sk = load.tle_file(tle_file)

# Print loaded satellites_sk
print(f"Loaded {len(satellites_sk)} satellites_sk from {tle_file}.")
latest_epoch = None
# Iterate through satellites and save latest time for integration
for sat in satellites_sk:
    #print(f"Satellite: {sat.name}, Epoch: {sat.epoch.utc_jpl()}")
    current_epoch = sat.epoch.utc_jpl()
    # Parse the epoch string into a datetime object
    current_epoch_dt = datetime.strptime(current_epoch, "A.D. %Y-%b-%d %H:%M:%S.%f UTC")

    # Update the latest epoch if this one is newer
    if latest_epoch is None or current_epoch_dt > latest_epoch:
        latest_epoch = current_epoch_dt

# Print the latest epoch
if latest_epoch:
    print(f"Latest Epoch: {latest_epoch.strftime('%Y-%m-%d %H:%M:%S.%f')}")



# Load Timescale
ts = load.timescale()

# Set the latest epoch in ts.utc format if found
if latest_epoch:
    start_time = ts.utc(
        latest_epoch.year, 
        latest_epoch.month, 
        latest_epoch.day, 
        latest_epoch.hour, 
        latest_epoch.minute, 
        latest_epoch.second #+ latest_epoch.microsecond / 1e6
    )


# This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
start_time = ts.utc(2024, 10, 23, 12, 00, 00)
# Convert to naive datetime
print("Start Time:", start_time.utc_iso())

# Defibe the time in datetime format to create TLE files for the LEO satellites
year, month, day = start_time.utc.year, start_time.utc.month, start_time.utc.day
hour, minute, second = start_time.utc.hour, start_time.utc.minute, start_time.utc.second
start_time_datetime = datetime(year, month, day, hour, minute, int(second), tzinfo=timezone.utc)


###################################
# Create TLE file for the constellation

output_path = '/home/chiatante/GNSS_R_System_Simulator/TLE_EPHEMIRIS_files/autogenerated/Constellation.tle'
with open(output_path, 'w') as outfile:
    pass  # Do nothing, just clear the file

for i in range(0, Nsat):
    sat = Satellite(i)
    if Nsat == 1:
        sat.set_orbit(orbit_input)
    else:
        sat.set_orbit(orbit_input[i])
    sat.set_epoch(start_time_datetime)
    sat.write_tle_file()
    print(sat.tle_file_path)
    if Nsat > 1: # If more than a satellite, merge the TLE for using the syfield package
        output_path = '/home/chiatante/GNSS_R_System_Simulator/TLE_EPHEMIRIS_files/autogenerated/Constellation.tle'
        # Open the output file in append mode ('a')
        with open(output_path, 'a') as outfile:
            # Read the TLE file of the current satellite
            with open(sat.tle_file_path, 'r') as file1:
                tle_content = file1.readlines()  # Read the TLE lines as a list
            # Write the TLE data to the output file
            outfile.write("".join(tle_content))  # Append the current TLE to the output file
            # Ensure there's a blank line between entries
            outfile.write("\n")
###################################        
if Nsat == 1:
    satellites_LEO_sk = load.tle_file(sat.tle_file_path)
else: #NSAT > 1, then use the merged file
    satellites_LEO_sk = load.tle_file(output_path)
    

#######################################################################################################
###################################### NEW ADDITION!! 26.11 PUSH ######################################


# Calculate the total number of steps
num_steps = (duration ) // step_size
# Generate the time series starting at 3:00 PM on October 23, 2024
times_1 = ts.utc(2024, 10, 23, 15, 0, range(0, num_steps * step_size, step_size))


# Need to create another datime obect, times_2, to run the latitude, longitude conversion
times_2 = [time.utc_datetime() for time in times_1]  # Call the method on each Time object

######################################
# Load the required data
planets = load('de421.bsp')
earth = planets['earth']
sun = planets['sun']

# Compute sunlight or not
sunlit = satellites_LEO_sk[0].at(times_1).is_sunlit(planets)
false_count = np.sum(sunlit == False)
true_count = np.sum(sunlit == True)
print(false_count)
print(true_count)

#######################################################################################################

#################
# Start process

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()

reflection_vec = []

for j in range(Nsat):
    # Position and velocity of the j LEO satellite
    leo_positions_ECI = np.array([satellites_LEO_sk[j].at(t).position.m for t in times_1])
    leo_positions_ECI = np.transpose(leo_positions_ECI)
    leo_velocity_ECI = np.array([satellites_LEO_sk[j].at(t).velocity.m_per_s for t in times_1])
    leo_velocity_ECI = np.transpose(leo_velocity_ECI)


    for i in range(len(satellites_sk)): #len(satellites_sk)):
        # Position and velocity of the i GNSS satellite
        GNSS_positions_ECI = np.array([satellites_sk[i].at(t).position.m for t in times_1])
        GNSS_positions_ECI = np.transpose(GNSS_positions_ECI)
        GNSS_velocity_ECI = np.array([satellites_sk[i].at(t).velocity.m_per_s for t in times_1])
        GNSS_velocity_ECI = np.transpose(GNSS_velocity_ECI)
        print(i)
        specular_point_vec, theta_vec, latlon_ref = compute_reflection(GNSS_positions_ECI, leo_positions_ECI, 0, 45, times_2)
        #ax.scatter(latlon_ref[:, 1], latlon_ref[:, 0], color='red', s=1, transform=ccrs.PlateCarree())
        lons, lats = fix_longitude_wraparound(latlon_ref[:, 1], latlon_ref[:, 0])
        ax.plot(lons, lats, 'red', linewidth=0.5, transform=ccrs.PlateCarree())

        #######################################################################################################
        ###################################### NEW ADDITION!! 26.11 PUSH ######################################
        distance_R = np.linalg.norm((specular_point_vec - np.transpose(leo_positions_ECI))/1e3, axis=1)
        distance_T =np.linalg.norm((specular_point_vec - np.transpose(GNSS_positions_ECI))/1e3, axis=1)

        Power = 1/(distance_R**2*distance_T**2)*1e15
        #ax.plot(latlon_ref[:, 1], latlon_ref[:, 0], color='red', transform=ccrs.PlateCarree())

        reflection_vec.append(Reflection('spherical'))
        reflection_vec[i + j*len(satellites_sk)].reflection_points['time'] = times_1
        reflection_vec[i + j*len(satellites_sk)].reflection_points['reflECEF'] = specular_point_vec
        reflection_vec[i + j*len(satellites_sk)].reflection_points['reflLL']= latlon_ref
        reflection_vec[i + j*len(satellites_sk)].reflection_points['gnssPos'] = GNSS_positions_ECI
        reflection_vec[i + j*len(satellites_sk)].reflection_points['gnssVel'] = GNSS_velocity_ECI
        reflection_vec[i + j*len(satellites_sk)].reflection_points['satPos'] = leo_positions_ECI
        reflection_vec[i + j*len(satellites_sk)].reflection_points['satVel'] = latlon_ref
        reflection_vec[i + j*len(satellites_sk)].reflection_points['refl_ang'] = theta_vec
        reflection_vec[i + j*len(satellites_sk)].reflection_points['Power_con'] = Power
        reflection_vec[i + j*len(satellites_sk)].tx_satID = satellites_sk[i].name
        reflection_vec[i + j*len(satellites_sk)].rx_satID = satellites_LEO_sk[j].name
        #######################################################################################################

# Define bounding box and grid
min_lon, max_lon = -17, 30
min_lat, max_lat = 56, 80
#ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent

# Plot reflection points
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
gridlines.top_labels = False  # Disable top labels
gridlines.right_labels = False  # Disable right-side labels

plt.title('Specular Reflection Points for GPS and LEO Satellite Over Time')
plt.savefig('Reflection_traces' + tle_file.split('/')[-1] + ".png" , dpi = 300)




############################################################
######### Coverage calculation part ###############
cwd = os.getcwd()
output_shapefile = os.path.join(cwd, 'iho', 'fishnet.shp')
grid_gdf = gpd.read_file(output_shapefile)


# Initialize figure and axis with projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# Define bounding box and grid
min_lon, max_lon = -17, 30
min_lat, max_lat = 56, 80
ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent

# Load the grid shapefile using the correct variable for the path
grid_gdf = gpd.read_file(output_shapefile)

# Plot the grid as the background (light gray color)
grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")


# Initialize coverage vector for storing coverage over each cell in the fishnet
coverage_vec = []
for idx in range(len(grid_gdf)):
    coverage = Coverage()  
    # Compute the centroid of the current grid cell
    centroid = grid_gdf.loc[idx].geometry.centroid
    # Set the position attribute of the Coverage instance
    coverage.position = (centroid.x, centroid.y)
    # Append the instance to the coverage_vec list
    coverage_vec.append(coverage)
    coverage.shape = grid_gdf.loc[[idx]].boundary


# Loop through the reflection vectors
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()
ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent
grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")
for i in range(len(reflection_vec)):#range(11):
    refl_temp = reflection_vec[i].reflection_points['reflLL']
    points_in_box = []
    check = False

    points_in_box = []
    # Filter points within the bounding box and ignore points outside it
    for lat, lon in refl_temp:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            points_in_box.append((lat, lon))
            check = True
        else:
            points_in_box.append((np.nan, np.nan))

    points_in_box_1 = np.array(points_in_box)
    valid = ~np.isnan(points_in_box_1[:, 0])  # Check latitude for NaN
    clusters = np.split(np.arange(len(points_in_box_1)), np.where(~valid)[0])  # Split indices by NaN gaps
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
        valid_cluster = cluster[valid[cluster]]
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
        ax.plot(valid_position[:, 1], valid_position[:, 0], color='red', transform=ccrs.PlateCarree())
        if len(intersecting_rectangles) > 0:
            intersecting_gdf = grid_gdf.loc[intersecting_rectangles]
            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")
        
            # Save results
            grid_gdf.loc[intersecting_rectangles]
            valid_position_clean = valid_position[~np.isnan(valid_position).any(axis=1)]
            valid_positions_flipped = valid_position_clean[:, [1, 0]]
            centroids_array = np.array([(centroid.x, centroid.y) for centroid in grid_gdf.loc[intersecting_rectangles].centroid])
            distances = cdist(centroids_array, valid_positions_flipped)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_points = valid_positions_flipped[nearest_indices]
            times_at_nearest_points = [times_2[i] for i in cluster[nearest_indices]]
            angle_at_nearest_points = reflection_vec[i].reflection_points['refl_ang'][cluster[nearest_indices]]
            power_at_nearest_points = reflection_vec[i].reflection_points['Power_con'][cluster[nearest_indices]]

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

plt.savefig("Coverage1.png", dpi = 300)



################################################################
######################## PLOTTING ##############################

intersection_count = {idx: 0 for idx in grid_gdf.index}

# Iterate through coverage_vec to evaluate the times attribute and count intersections
for coverage in coverage_vec:
    # If there are any times recorded, update the intersection count for the corresponding grid cell
    # Assuming coverage_vec corresponds to the same index as grid_gdf
    grid_idx = coverage_vec.index(coverage)  # Get the index of the current coverage in coverage_vec
    intersection_count[grid_idx] += len(coverage.times)  # Increase count based on the number of times recorded

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
for i in range(len(reflection_vec)):
    lons, lats = fix_longitude_wraparound(reflection_vec[i].reflection_points['reflLL'][:,1], reflection_vec[i].reflection_points['reflLL'][:,0])
    ax.plot(lons, lats,'red')
ax.set_global()
ax.set_extent([min_lon, max_lon, min_lat, max_lat])  # Add padding to the extent
grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")

# Loop through each grid cell to plot with the correct color
for idx in grid_gdf.index:
    # Get the intersection count for the grid cell
    count = intersection_count[idx]
    
    # Get the color for the current grid cell
    color, alpha_vec = get_color(count)
    
    coverage_vec[idx].shape.plot(ax=ax, edgecolor='red', facecolor=color, alpha = 0.3)

# Plot the blue grid cells (those intersected more than once) on top of the red ones
for idx in grid_gdf.index:
    count = intersection_count[idx]
    color, alpha_vec = get_color(count)
    
    if color == 'blue':  # Only plot blue cells (more than 1-time intersection)
        coverage_vec[idx].shape.plot(ax=ax, edgecolor='blue', facecolor=color, alpha = 0.3)


plt.savefig("Coverage2.png", dpi = 300)


####################################################
########### Average  signal"Power" ###########
all_powers = np.concatenate([item.power for item in coverage_vec])  # Flatten all power values
average_power = np.mean(all_powers)  # Calculate the average
print("Average Power:", average_power)

########### Coverage ###########
non_zero_count = sum(1 for value in intersection_count.values() if value != 0)
print("Coverage percentage:", non_zero_count/len(intersection_count)*100)




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





'''
##############
## Plot orbits

earth_radius = 6371  # Earth's average radius in kilometers
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Earth as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax = fig.add_subplot(111, projection='3d')
# Plot the satellite trajectory (line connecting the points)
leo_positions_ECI_1 = np.array([satellites_LEO_sk[0].at(t).position.m for t in times_1])
leo_positions_ECI_1 = np.transpose(leo_positions_ECI_1/1e3)
leo_positions_ECI_2 = np.array([satellites_LEO_sk[1].at(t).position.m for t in times_1])
leo_positions_ECI_2 = np.transpose(leo_positions_ECI_2/1e3)

ax.plot(leo_positions_ECI_1[0], leo_positions_ECI_1[1], leo_positions_ECI_1[2], label="Satellite 1 Path", color='r')

# Plot the satellite trajectory for the second satellite
ax.plot(leo_positions_ECI_2[0], leo_positions_ECI_2[1], leo_positions_ECI_2[2], label="Satellite 2 Path", color='b')
plt.savefig('Orbits.png')
'''









'''
print(leo_positions_ECI)

leo_positions_ECI = np.transpose(leo_positions_ECI)
pos_km = satellites_LEO_sk[0].at(times_1[0]).position.km

print(pos_km)


leo_orbit = Orbit_poliastro.from_classical(Earth, leo_semi_major_axis, leo_eccentricity, leo_inclination,
                                 leo_raan, leo_arg_perigee, leo_true_anomaly, Time(current_time))
leo_positions_ECI_2 = np.array([leo_orbit.propagate(t).r.to(u.m).value for t in times_2])
leo_positions_ECI_2 = np.transpose(leo_positions_ECI_2)

difference = (leo_positions_ECI - leo_positions_ECI_2)/1e3

plt.figure(figsize=(12, 6))
plt.plot( range(len(times_1)), difference[0, :])
plt.plot(range(len(times_1)), difference[2, :])
plt.plot(range(len(times_1)), difference[1, :])
plt.plot(range(len(times_1)), np.linalg.norm(difference, axis=0))

plt.savefig('Difference_methods.png', dpi = 300)
'''





'''
if method == 1:
    for i in len(reflection_vec):#range(11):
        refl_temp = reflection_vec[i].reflection_points['reflLL']
        points_in_box = []
        check = False
        # Filter points within the bounding box and ignore points outside it
        for lat, lon in refl_temp:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                points_in_box.append((lat, lon))
                check = True
            else:
                points_in_box.append((np.nan, np.nan))
        cleaned_points_in_box = []
        for j in range(len(points_in_box)):
            if j > 0 and j < len(points_in_box) - 1:  # Check for non-edge points
                prev_point = points_in_box[j - 1]
                next_point = points_in_box[j + 1]
                current_point = points_in_box[j]

                # Replace current point with NaN if both previous and next points are NaN
                if np.isnan(prev_point[0]) and np.isnan(next_point[0]):
                    cleaned_points_in_box.append((np.nan, np.nan))
                else:
                    cleaned_points_in_box.append(current_point)
            else:
                # Keep edge points as is
                cleaned_points_in_box.append(points_in_box[j])

        # Replace `points_in_box` with the cleaned version
        points_in_box = cleaned_points_in_box
        if check and len(points_in_box) > 1:
            lats, lons = zip(*points_in_box)
            ax.plot(lons, lats, color='blue', label="Line", linewidth=2)
            # Create a line from the valid points within the bounding box
            line = LineString([(lon, lat) for lat, lon in points_in_box])
            # Find intersecting grid cells
            intersecting_rectangles = []
            print(intersecting_rectangles)
            for idx, polygon in grid_gdf.iterrows():
                # Check if the polygon intersects with the line
                #print(polygon)
                intersection = polygon['geometry'].intersection(line)
                if not intersection.is_empty:
                    intersecting_rectangles.append(idx)
            # Plot intersecting rectangles if any
            if intersecting_rectangles:
                intersecting_gdf = grid_gdf.loc[intersecting_rectangles]
                intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")
   
   
   
    # Finalize the plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
    gridlines.top_labels = False  # Disable top labels
    gridlines.right_labels = False  # Disable right-side labels
    # Add a legend only once
    #ax.legend(loc='upper right')
    # Save and show the plot
    plt.savefig("Intersection_rectangles_from_grid_points_with_background.png", dpi=300)

'''