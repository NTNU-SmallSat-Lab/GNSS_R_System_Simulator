import os
import numpy             as np
import matplotlib.pyplot as plt

import cartopy.crs       as ccrs
import cartopy.feature   as cfeature

from datetime          import datetime, timezone, timedelta

from poliastro.bodies          import Earth
from poliastro.twobody         import Orbit as Orbit_poliastro
#from poliastro.plotting.static import StaticOrbitPlotter

from skyfield.api      import load, wgs84
from skyfield.sgp4lib  import EarthSatellite
from skyfield.framelib import itrs

from astropy           import units           as u
from astropy.time      import Time, TimeDelta

from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Orbit_def             import Orbit_native
from object_definitions.Satellite_def         import Satellite              as Satellite
from object_definitions.GnssConstellation_def import GNSSConstellation      as GNSSConst
from object_definitions.ReadWriteSatData      import ReadWriteSatData       as rw
from object_definitions.SatConstellation_def  import SatelliteConstellation as SatConfig

from functions import compute_reflection


tle_file = "TLE_EPHEMIRIS_files\\ISS.tle"             # Replace with your file's name

satellites_sk = load.tle_file(tle_file)

# Print loaded satellites_sk
print(f"Loaded {len(satellites_sk)} satellites_sk from {tle_file}.")

###################

# Define LEO Satellite Orbital Parameters
leo_semi_major_axis = 7300 * u.km      # (Earth.R + leo_altitude).to(u.km)  # LEO altitude + Earth radius
leo_eccentricity    = 0.00001 * u.one  # Almost circular orbit
leo_inclination     = 98 * u.deg       # Sun-synchronous orbit inclination
leo_raan            = 150 * u.deg
leo_arg_perigee     = 0 * u.deg
leo_true_anomaly    = 0 * u.deg

###################

# Define the start time
start_time_dt  = datetime(2024, 11, 10, 00, 00, 00, tzinfo=timezone.utc)
ts             = load.timescale()                        # initialize skyfield timescale object
start_time     = ts.from_datetime(start_time_dt)         # Convert to "Time" object from Skyfield
naive_datetime = start_time.utc_datetime()               # Convert to "datetime" object


print(start_time)
print("Start Time:", start_time.utc_iso())

# Define the step size and duration
step_size = 10  # Seconds
duration = 8 * 3600  # Total duration in seconds (6 hours)

###################################
sat = Satellite(1)
sat.set_orbit([98, 0.00001, 150, 0, 0,  7300])
sat.set_epoch(start_time_dt)
#sat.set_orbit([0.00001 * u.one, 98 * u.deg, 150 * u.deg, 0 * u.deg, 0 * u.deg,  7300 * u.km])


sat.write_tle_file()

#############################################################################################################################
# Generate a sequence of times
times_1      = []
current_time = start_time.utc_datetime()
for t in range(0, duration, step_size):
    new_time = current_time + timedelta(seconds=t)
    times_1.append(ts.utc(new_time))
    	

end_time          = Time(current_time) + TimeDelta(duration * u.s)  # 3 hours fligiht time
sample_time       = step_size * u.s                                 # Sample every 20 seconds
times_2           = Time([Time(current_time) + i * sample_time for i in range(int((end_time - Time(current_time)).sec / sample_time.value))])
satellite_LEO_sk  = load.tle_file(sat.tle_file_path)
LEO_positions_ECI = np.array([satellite_LEO_sk[0].at(t).position.m for t in times_1])
LEO_positions_ECI = np.transpose(LEO_positions_ECI)

"""
leo_orbit         = Orbit_poliastro.from_classical(Earth, leo_semi_major_axis, leo_eccentricity, leo_inclination,
                                                   leo_raan, leo_arg_perigee, leo_true_anomaly, Time(current_time))

# Calculate time deltas from the start time
time_deltas = [(t - Time(current_time)).sec * u.s for t in times_2]
# Propagate the orbit using time deltas and extract positions
#leo_positions_ECI = np.array([leo_orbit.propagate(delta).r.to(u.m).value for delta in time_deltas])
leo_positions_ECI = np.array([leo_orbit.propagate(t).r.to(u.m).value for t in times_2])
leo_positions_ECI = np.transpose(leo_positions_ECI)
"""

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
    specular_point_vec, theta_vec, latlon_ref = compute_reflection(gps_positions_ECI, LEO_positions_ECI, 0, 45, times_2)
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
plt.show()
a = 1
