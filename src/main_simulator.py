from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit as Orbit_poliastro
from poliastro.plotting.static import StaticOrbitPlotter
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


##
from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Orbit_def             import Orbit_native
from object_definitions.Satellite_def         import Satellite
from object_definitions.GnssConstellation_def import GNSSConstellation
from object_definitions.ReadWriteSatData      import ReadWriteSatData
from object_definitions.SatConstellation_def  import SatelliteConstellation


#############################################################################################################################

import os
#tle_file = r"\\functions_python\\gps-ops.txt"
#print(os.path.abspath(tle_file))  # Prints the absolute path being searched
#print(os.path.exists(tle_file))  # Checks if the file exists

#tle_url = "https://celestrak.org/NORAD/elements/gp.php?NAME=HYPSO-1&FORMAT=TLE"  # ISS (ZARYA) TLE
tle_file = "functions_python/gps-ops.txt"  # Replace with your file's name
satellites_sk = load.tle_file(tle_file)

# Print loaded satellites_sk
print(f"Loaded {len(satellites_sk)} satellites_sk from {tle_file}.")

# Iterate through satellites and print their names and epochs
#for sat in satellites:
#    print(f"Satellite: {sat.name}, Epoch: {sat.epoch.utc_jpl()}")


###################

# Define LEO Satellite Orbital Parameters
leo_semi_major_axis = 7300 * u.km # (Earth.R + leo_altitude).to(u.km)  # LEO altitude + Earth radius
leo_eccentricity = 0.00001 * u.one  # Almost circular orbit
leo_inclination = 98 * u.deg  # Sun-synchronous orbit inclination
leo_raan = 150 * u.deg
leo_arg_perigee = 0 * u.deg
leo_true_anomaly = 0 * u.deg

###################





# Load Timescale
ts = load.timescale()

# Define the start time
start_time = ts.utc(2024, 11, 10, 00, 00, 00)
# Convert to naive datetime
naive_datetime = start_time.utc_datetime()

# Add timezone information
aware_datetime = naive_datetime.replace(tzinfo=timezone.utc)

print(start_time)
#start_time = ts.utc(2024, 11, 7, 8, 53, 57, tzinfo = timezone.utc)

print("Start Time:", start_time.utc_iso())

# Define the step size and duration
step_size = 10  # Seconds
duration = 8 * 3600  # Total duration in seconds (6 hours)



###################################
sat = Satellite(1)
sat.set_orbit([98, 0.00001, 150, 0, 0,  7300])
sat.set_epoch(datetime(2024, 11, 10, 00, 00, 00, tzinfo=timezone.utc))
#sat.set_orbit([0.00001 * u.one, 98 * u.deg, 150 * u.deg, 0 * u.deg, 0 * u.deg,  7300 * u.km])


sat.write_tle_file()



#############################################################################################################################
# Generate a sequence of times
times_1 = []
current_time = start_time.utc_datetime()
for t in range(0, duration, step_size):
    new_time = current_time + timedelta(seconds=t)
    times_1.append(ts.utc(new_time))
    	

end_time = Time(current_time) + TimeDelta(duration * u.s)  # 3 hours later
sample_time = step_size * u.s  # Sample every 20 seconds
times_2 = Time([Time(current_time) + i * sample_time for i in range(int((end_time - Time(current_time)).sec / sample_time.value))])
satellite_LEO_sk = load.tle_file(sat.tle_file_path)
LEO_positions_ECI = np.array([satellite_LEO_sk[0].at(t).position.m for t in times_1])
LEO_positions_ECI = np.transpose(LEO_positions_ECI)


'''
leo_orbit = Orbit_poliastro.from_classical(Earth, leo_semi_major_axis, leo_eccentricity, leo_inclination,
                                 leo_raan, leo_arg_perigee, leo_true_anomaly, Time(current_time))
leo_positions_ECI = np.array([leo_orbit.propagate(t).r.to(u.m).value for t in times_2])
leo_positions_ECI = np.transpose(leo_positions_ECI)
'''

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()
print(range(len(satellites_sk)))

for i in range(0,1):# range(len(satellites_sk)):
    gps_positions_ECI = np.array([satellites_sk[i].at(t).position.m for t in times_1])
    gps_positions_ECI = np.transpose(gps_positions_ECI)

    print(i)
    specular_point_vec, theta_vec, latlon_ref = compute_reflection(gps_positions_ECI, leo_positions_ECI, 0, 45, times_2)
    ax.scatter(latlon_ref[:, 1], latlon_ref[:, 0], color='red', s=1, transform=ccrs.PlateCarree())

# Plot reflection points
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
gridlines.top_labels = False  # Disable top labels
gridlines.right_labels = False  # Disable right-side labels

plt.title('Specular Reflection Points for GPS and LEO Satellite Over Time')
plt.savefig('Reflection_traces.png', dpi = 300)







'''
# Define GPS Satellite Orbital Parameters (Approximate for MEO GPS satellites)
gps_semi_major_axis = 26561 * u.km  # GPS typical altitude + Earth radius
gps_eccentricity = 0.0085 * u.one  # Almost circular orbit
gps_inclination = 55.7032 * u.deg  # Typical inclination for GPS
gps_raan = 122.9752 * u.deg  # Right Ascension of Ascending Node
gps_arg_perigee = 54.3592 * u.deg  # Argument of Perigee
gps_true_anomaly = 306.5248 * u.deg  # Starting point in orbit

'''
'''
# Define LEO Satellite Orbital Parameters
leo_semi_major_axis = 7300 * u.km # (Earth.R + leo_altitude).to(u.km)  # LEO altitude + Earth radius
leo_eccentricity = 0.00001 * u.one  # Almost circular orbit
leo_inclination = 98 * u.deg  # Sun-synchronous orbit inclination
leo_raan = 150 * u.deg
leo_arg_perigee = 0 * u.deg
leo_true_anomaly = 0 * u.deg

# Initialize orbits
start_time_str = "2024-10-01 11:24:52"
epoch = Time(start_time_str)
#gps_orbit = Orbit.from_classical(Earth, gps_semi_major_axis, gps_eccentricity, gps_inclination,
#                                 gps_raan, gps_arg_perigee, gps_true_anomaly, epoch)
leo_orbit = Orbit.from_classical(Earth, leo_semi_major_axis, leo_eccentricity, leo_inclination,
                                 leo_raan, leo_arg_perigee, leo_true_anomaly, epoch)

# Propagate orbits over a 3-hour period with 20-second intervals


# Store positions and velocities
#gps_positions_ECI = np.array([gps_orbit.propagate(t).r.to(u.m).value for t in times])
leo_positions_ECI = np.array([leo_orbit.propagate(t).r.to(u.m).value for t in times])
#gps_positions_ECI = np.transpose(gps_positions_ECI)
leo_positions_ECI = np.transpose(leo_positions_ECI)




#lat, lon, alt = pymap3d.eci2geodetic(leo_positions_ECI[:, 0], leo_positions_ECI[:, 1], leo_positions_ECI[:, 2], times)
#lat = np.transpose(lat)
#lon = np.transpose(lon)



specular_point_vec, theta_vec, latlon_ref = compute_reflection(gps_positions_ECI, leo_positions_ECI, 0, 45, times)

print([np.transpose(leo_positions_ECI)/1e3, np.transpose(gps_positions_ECI)/1e3])



plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()

# Plot reflection points
ax.scatter(latlon_ref[:, 1], latlon_ref[:, 0], color='red', s=1, transform=ccrs.PlateCarree())
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
gridlines.top_labels = False  # Disable top labels
gridlines.right_labels = False  # Disable right-side labels

plt.title('Specular Reflection Points for GPS and LEO Satellite Over Time')
plt.savefig('temp.png', dpi = 300)

'''
