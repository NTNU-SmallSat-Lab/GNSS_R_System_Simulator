import os

from astropy             import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time        import Time, TimeDelta
from poliastro.bodies    import Earth
from poliastro.twobody   import Orbit as Orbit_poliastro

import numpy           as np
import pandas          as pd
import cartopy.crs     as ccrs
import cartopy.feature as cfeature

from functions         import compute_reflection
from astropy           import coordinates as coord

import pymap3d 

from skyfield.api      import load, wgs84
#from skyfield.sgp4lib  import EarthSatellite
from datetime          import timedelta, timezone, datetime
#from skyfield.framelib import itrs

##
from object_definitions.Reflection_def        import Reflection
from object_definitions.Constants_def         import Constants
from object_definitions.Satellite_def         import Satellite
from object_definitions.ReadWriteSatData      import ReadWriteSatData
from object_definitions.SatConstellation_def  import SatelliteConstellation

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg', force=True)
print("Switched to:",matplotlib.get_backend())

#############################################################################################################################
tle_file_path       = "TLE_EPHEMIRIS_files/GPS_operational.tle"                            # Replace with your file's name 
satellites_sk       = load.tle_file(tle_file_path)
GNSS_const          = SatelliteConstellation(tle_file=tle_file_path)

print(f"Loaded {GNSS_const.no_satellites} GNSS satellites from {tle_file_path}.")                           # Print loaded satellites
latest_epoch = None
# Iterate through satellites and save latest time for integration
for sat in GNSS_const.satellites:
    #print(f"Satellite: {sat.name}, Epoch: {sat.epoch.utc_jpl()}")
    current_epoch = sat.skf.epoch.utc_jpl()
    # Parse the epoch string into a datetime object
    current_epoch_dt = datetime.strptime(current_epoch, "A.D. %Y-%b-%d %H:%M:%S.%f UTC")

    # Update the latest epoch if this one is newer
    if latest_epoch is None or current_epoch_dt > latest_epoch:
        latest_epoch = current_epoch_dt

# Print the latest epoch
if latest_epoch is not None:
    print(f"Latest Epoch: {latest_epoch.strftime('%Y-%m-%d %H:%M:%S.%f')}")

# Load Timescale
start_time_dt = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)

# This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
ts         = load.timescale()
start_time = ts.utc(start_time_dt)
# Convert to naive datetime
print("Start Time:", start_time.utc_iso())

# Define the step size and duration
step_size = 20             # Seconds
duration  = int(4 * 3600)  # Total duration in seconds (4 hours)

#################################################
orbits_input = [[70, 0.00001, 0,   0, 0, 7300e3, start_time_dt],
                [70, 0.00001, 180, 0, 0, 7300e3, start_time_dt]]
Nsat        = 2

#############################################################################################################################
# Create TLE file for the constellation
sat_array = []
for i in range(0, Nsat):
    sat_array.append(Satellite(i, orbit_n=orbits_input[i]))
sat_const = SatelliteConstellation(satConfigName="Constellation", satArray=sat_array)
sat_const.generate_constellation_TLE()

#############################################################################################################################
sat_const.propagate_constellation(start_time_dt, duration, step_size)
GNSS_const.propagate_constellation(start_time_dt, duration, step_size)

times_2_ = [time.utc_datetime() for time in sat_const.satellites[0].time_prop_vec]  # Call the method on each Time object

# Start process
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()

reflection_vec = []

sat_no = 0
for sat in sat_const.satellites:
    gnss_sat_no = 0
    for gnss_sat in GNSS_const.satellites:
        print(gnss_sat_no)
        specular_point_vec, theta_vec, latlon_ref = compute_reflection(gnss_sat.positions_eci, sat.positions_eci, 0, 45, times_2_)
        reflection_vec.append(Reflection())
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['time']     = sat.time_prop_vec
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['reflECEF'] = specular_point_vec
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['reflLL']   = latlon_ref
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['gnssPos']  = gnss_sat.positions_eci
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['gnssVel']  = gnss_sat.velocities_eci
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['satPos']   = specular_point_vec
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['satVel']   = latlon_ref
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].reflection_points['refl_ang'] = theta_vec
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].tx_satID                      = gnss_sat.satID    # gnss_sat.satName  # gnss_sat_no
        reflection_vec[gnss_sat_no + sat_no * GNSS_const.no_satellites].rx_satID                      = sat.satID         # sat.satName       # sat_no
        gnss_sat_no += 1
    sat_no += 1

satellites_LEO_sk = [sat_const.satellites[0].skf, sat_const.satellites[1].skf]

# Plot reflection points
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
gridlines              = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
gridlines.top_labels   = False  # Disable top labels
gridlines.right_labels = False  # Disable right-side labels

plt.title('Specular Reflection Points for GPS and LEO Satellite Over Time')
plt.savefig('Reflection_traces' + tle_file_path.split('/')[-1] + ".png" , dpi = 300)

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
plt.show()
a = 1

#print(leo_positions_ECI)
#leo_positions_ECI = np.transpose(leo_positions_ECI)
#pos_km = satellites_LEO_sk[0].at(times_1[0]).position.km
#print(pos_km)
#leo_orbit = Orbit_poliastro.from_classical(Earth, leo_semi_major_axis, leo_eccentricity, leo_inclination,
#                                           leo_raan, leo_arg_perigee, leo_true_anomaly, Time(start_time_dt))
#leo_positions_ECI_2 = np.array([leo_orbit.propagate(t).r.to(u.m).value for t in times_2])
#leo_positions_ECI_2 = np.transpose(leo_positions_ECI_2)

#difference = (leo_positions_ECI - leo_positions_ECI_2)/1e3

#plt.figure(figsize=(12, 6))
#plt.plot( range(len(times_1)), difference[0, :])
#plt.plot(range(len(times_1)), difference[2, :])
#plt.plot(range(len(times_1)), difference[1, :])
#plt.plot(range(len(times_1)), np.linalg.norm(difference, axis=0))

#plt.savefig('Difference_methods.png', dpi = 300)
