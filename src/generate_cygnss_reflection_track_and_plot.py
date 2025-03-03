import os
import json
import argparse 
from matplotlib import animation
import numpy as np
import datetime as dt
from dotenv import load_dotenv

from functions.functions import compute_reflection
import satellite_parameter_utils as utils

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pymap3d
from mpl_toolkits.basemap import Basemap

import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from skyfield.api import load, EarthSatellite

def animate_tracks(tracks, labels,start_time, filename="track_animation.mp4", frame_interval=100):
    if not tracks:
        print("No tracks available for animation.")
        return
    
    # Determine the bounding box dynamically
    all_lons = np.concatenate([track[:, 0] for track in tracks])
    all_lats = np.concatenate([track[:, 1] for track in tracks])
    
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    
    # Add some padding
    lon_padding = 5
    lat_padding = 5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    basemap = Basemap(projection='cyl',
                       llcrnrlon=min_lon - lon_padding, urcrnrlon=max_lon + lon_padding,
                       llcrnrlat=min_lat - lat_padding, urcrnrlat=max_lat + lat_padding,
                       resolution='l', ax=ax)
    basemap.drawcoastlines()
    basemap.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10)
    basemap.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], fontsize=10)
    
    # Initialize tracking dots
    dots = []
    for track, label in zip(tracks, labels):
        lons, lats = track[:, 0], track[:, 1]
        basemap.plot(lons, lats, linewidth=2, label=label)
        dot, = basemap.plot([], [], 'o', markersize=8, label=f"{label} Position")
        dots.append((dot, lons, lats))
    
    # Add legend at the bottom
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels), fontsize=10)

    # Add timestamp text
    timestamp_text = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=12, verticalalignment='bottom')

    def update(frame_idx):
        # Update the dots' positions
        for dot, lons, lats in dots:
            dot.set_data((lons[min(frame_idx, len(lons) - 1)],), (lats[min(frame_idx, len(lats) - 1)],))        
        # Compute elapsed time in seconds
        timestamp = frame_idx * (frame_interval / 1000.0)
        timestamp_text.set_text(f"Time: {timestamp:.2f} s\nAbsolute Time: {(start_time + dt.timedelta(seconds=timestamp)).strftime('%Y-%m-%d %H:%M:%S')}")
        return [dot for dot, _, _ in dots] + [timestamp_text]
    
    max_frames = max(len(track) for track in tracks)
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=frame_interval, blit=True)
    ani.save(filename, writer='ffmpeg')
    plt.close(fig)
    print(f"Animation saved as {filename}")

def does_track_cross_from_land_to_sea(basemap, track: np.array) -> bool:
    previous_point = None
    for evaluation_point in track:
        if not previous_point: # Handle the first point
            if Basemap.is_land(basemap, evaluation_point[0], evaluation_point[1]):
                previous_point = "land"
            else:
                previous_point = "sea"
            continue
        if Basemap.is_land(basemap, evaluation_point[0], evaluation_point[1]):
            if previous_point == "sea":
                return True
            previous_point = "land"
        else:
            if previous_point == "land":
                return True
            previous_point = "sea"
    return False

def generate_prn_norad_tuples(relevant_PNRs:List[int], date_start:dt.datetime, date_end:dt.datetime)->List[Tuple[int, int]]:
    prn_norad_tuples = []
    for prn in relevant_PNRs:
        # PRN needs to be on the format GXX where XX is the PRN number, the number needs to be 2 digits.
        prn = f"G{prn:02d}"
        prn_mapping_data = utils.load_satellite_parameter_mapping(utils.JSON_MAPPING_FILENAME)
        norad_id = utils.prn_to_norad_id(prn_mapping_data, prn, (date_start, date_end))

        if len(norad_id) > 1:
            raise ValueError(f"Multiple NORAD IDs found for PRN {prn} in the given date range. \
                            This is unexpected behaviour.")
        if len(norad_id) == 0:
            raise ValueError(f"No NORAD ID found for PRN {prn} in the given date range. \
                            This is unexpected behaviour.")
        prn_norad_tuples.append((int(prn[1:]), norad_id[0]))
    return prn_norad_tuples

def get_all_data_entries_that_crossed_from_sea_to_land(filename: str) -> Dict[str, List[Dict[str, List[List[float]]]]]:
    ''' 
    Returns a dict with all data entries that crossed from sea to land or vice versa.
    Data entries in this case mean a file from the data folder. (Having a name
    and start and stop times).
    The dict can be found in the already_processed_satellites_and_dates.json file. 
    The format is:
    {
    "cygfm01": [
        {
            "date_start": "2017-09-09 06:20:00",
            "date_end": "2017-09-09 06:21:00",
            "track": [[lon1, lat1], [lon2, lat2], ...]
        },
        {
            "date_start": "2018-08-27 00:34:43",
            "date_end": "2018-08-27 00:35:43",
            "track": []
        }
    ],
    "cygfm02": [
        {
            "date_start": "2017-09-21 15:41:15",
            "date_end": "2017-09-21 15:42:15",
            "track": [[lon1, lat1], [lon2, lat2], ...]
        }
    ]
    }
    '''
    with open(filename, 'r') as file:
        data = json.load(file)
    # Filter out all entries where track is null
    # If the track crosses from sea to land, the track was saved, if not, it didnt.
    filtered_data = {}
    for satellite_name, entries in data.items():
        filtered_data[satellite_name] = [entry for entry in entries if entry["track"]]
                
    return filtered_data

def plot_track_on_basemap(track: np.array, color: str, label: str):
    '''Just a helper for debugging'''
    # Find an appropriate bounding box for the plot

    min_lat = np.min(track[:, 1])
    max_lat = np.max(track[:, 1])
    min_lon = np.min(track[:, 0])
    max_lon = np.max(track[:, 0])

    # Create a padding around the bounding box. It has to be relative to the range of the data
    padding = 0.9
    lat_range = max_lat - min_lat

    basemap_map = Basemap(projection='cyl', lon_0=0, resolution='l')
    parallels = np.arange(-90, 91, 30)   # Latitude lines every 30 degrees
    meridians = np.arange(-180, 181, 30)  # Longitude lines every 30 degrees
    
    basemap_map.drawcoastlines()
    basemap_map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)  # Latitude labels on the left
    basemap_map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)  # Longitude labels on the bottom
    basemap_map.plot(track[:, 0], track[:, 1], linewidth=2, color=color, label=label)
    plt.legend()
    plt.show()

    

def generate_cygnss_reflection_track_and_plot(cygnss_satellite_name:str="cygfm01", relevant_PNRs:List[int]=[20, 10, 32, 14], 
                                            date_start:dt.datetime=dt.datetime(2018, 8, 22, 14, 26, 59), date_end:dt.datetime=dt.datetime(2018, 8, 22, 14, 27, 59)):

    dotenv_path = os.path.join(os.path.dirname(os.getcwd()), '.env')
    load_dotenv(dotenv_path, override=True) # If you dont override, it might not re-load the env variables if they change.

    # Use the Space-Track API to get the TLE data
    spacetrack_username = os.getenv("SPACETRACK_ID")
    spacetrack_password = os.getenv("SPACETRACK_PASS")
    st = SpaceTrackClient(identity=spacetrack_username, password=spacetrack_password)
    print(f"Space-Track API connected with user: {spacetrack_username}")


    # We need the NORAD ID for the GPS satellites to get the TLE data
    prn_norad_tuples = generate_prn_norad_tuples(relevant_PNRs, date_start, date_end)
        
    # One day search for the TLEs on spacetrack
    # this was done as TLE date typically is uploaded a few times a day
    # the exact time interval would likely not give a match
    date_search_start = date_start.date()
    date_search_end = date_start.date() + dt.timedelta(days=1)
    date_search_range = op.inclusive_range(date_search_start, date_search_end)

    ##################################################################
    #################### Create CYGNSS TLE data ######################
    ##################################################################
    cygnss_satellite_tle_data_json = st.satcat(satname=cygnss_satellite_name, format='json')
    cygnss_tle_data = []

    if cygnss_satellite_tle_data_json:
        cygnss_satellite_data = json.loads(cygnss_satellite_tle_data_json)
        norad_cat_id_cy = cygnss_satellite_data[0].get("NORAD_CAT_ID")
        lines = list(st.tle(iter_lines=True, norad_cat_id=norad_cat_id_cy, epoch=date_search_range, orderby="EPOCH", format="tle"))
        print(f"CYGNSS TLE data found: {lines}")
        
        if lines:
            cygnss_tle_data = lines[-2:]
    else:
        print("Satellite not found.")

    ##################################################################
    #################### Create GPS TLE data #########################
    ##################################################################
    gps_tle_data = {}
    
    for PNR in relevant_PNRs:
        matching_prn_norad_tuple = next((prn_norad_tuple for prn_norad_tuple in prn_norad_tuples if int(prn_norad_tuple[0]) == PNR), None)

        if matching_prn_norad_tuple:
            norad_cat_id = matching_prn_norad_tuple[1]
            lines = list(st.tle(iter_lines=True, norad_cat_id=norad_cat_id, epoch=date_search_range, orderby="EPOCH", format="tle"))
            
            if lines:
                gps_tle_data[norad_cat_id] = lines[-2:]
            else:
                print(f"No data found for PRN {PNR}.")

    # Create satellite objects directly from TLE data
    satellites_LEO_sk = [EarthSatellite(cygnss_tle_data[0], cygnss_tle_data[1])] if cygnss_tle_data else []
    satellites_GPS_sk = [EarthSatellite(tle[0], tle[1]) for tle in gps_tle_data.values()]





    ##################################################################
    ######################## Plot the tracks #########################
    ##################################################################

    plt.figure(figsize=(10, 10))
    earth_visualization_map = Basemap(projection='cyl', lon_0=0, resolution='l')


    earth_visualization_map.drawcoastlines()
    # Define latitude and longitude intervals suitable for zoomed-in view -> only for this case. Make it general
    parallels = np.arange(-90, 91, 30)   # Latitude lines every 30 degrees
    meridians = np.arange(-180, 181, 30)  # Longitude lines every 30 degrees
    

    # Draw latitude and longitude grid lines with labels
    earth_visualization_map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)  # Latitude labels on the left
    earth_visualization_map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)  # Longitude labels on the bottom

    # Generate Seconds time vector to add it to the time object. 
    seconds_increment = 1 # Seconds between each point
    seconds_array = np.arange(0, (date_end - date_start).total_seconds(), seconds_increment)
    # Load timescale and generate Skyfield time series
    start_time_dt = date_start 
    ts = load.timescale()
    utc_time_series = ts.utc(
        start_time_dt.year,
        start_time_dt.month,
        start_time_dt.day,
        start_time_dt.hour,
        start_time_dt.minute,
        start_time_dt.second + seconds_array
    )

    # Need to create another datime obect, times_2, to run the latitude, longitude conversion
    gps_time_datetimes = [time.utc_datetime() for time in utc_time_series]  # Call the method on each Time object


    satellite_leo = satellites_LEO_sk[0]
    cygnss_satellite_position = np.array([satellite_leo.at(t).position.m       for t in utc_time_series])
    #velocity2 = np.array([satellite_leo.at(t).velocity.m_per_s for t in times_1])
    cygnss_satellite_position = np.transpose(cygnss_satellite_position)
    color_vector = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

    gps_sat_tracks = []
    gps_sat_tack_names = []
    # Iterate along different GPS satellites (and PRN codes)
    for satellite_gps, prn, color in zip(satellites_GPS_sk, relevant_PNRs, color_vector):
        # Extract position and velocity vectors
        gps_satellite_position = np.array([satellite_gps.at(t).position.m for t in utc_time_series])
        gps_satellite_position = np.transpose(gps_satellite_position)

        MIN_ANGLE = 0
        MAX_ANGLE = 80 # Corrado says this is a large max and no reflection should be constrained by this limit
        # Compute reflection
        specular_point_vec, theta_vec, latlon_ref = compute_reflection(gps_satellite_position, cygnss_satellite_position, MIN_ANGLE, MAX_ANGLE, gps_time_datetimes)
        gps_sat_tracks.append(np.column_stack((latlon_ref[:,1], latlon_ref[:,0])))
        # Plot reflection track
        plot_label=f"Reflection track {prn}"
        earth_visualization_map.plot(latlon_ref[:, 1], latlon_ref[:, 0], linewidth=2, color=color, label=plot_label,linestyle='dashed')
        gps_sat_tack_names.append(plot_label)

        # Check if the track crosses from land to sea or vice versa
        if does_track_cross_from_land_to_sea(earth_visualization_map, latlon_ref):
            print(f"Reflection track for PRN {prn} crosses from land to sea or vice versa.")

        # Convert ECI to geodetic coordinates
        lat, lon, alt = pymap3d.eci2geodetic(gps_satellite_position[0, :], gps_satellite_position[1, :], gps_satellite_position[2, :], gps_time_datetimes)
        
        # Plot GPS satellite track
        earth_visualization_map.plot(lon, lat, linewidth=4, color=color, label=f"GPS Satellite {prn}")

    lat, lon, alt = pymap3d.eci2geodetic(cygnss_satellite_position[0, :], cygnss_satellite_position[1, :], cygnss_satellite_position[2, :], gps_time_datetimes)

    earth_visualization_map.plot(lon, lat, linewidth=2, color='k', label = cygnss_satellite_name)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='lower left') # Place in the bottom left location
    plt.show()

    animate_tracks(gps_sat_tracks, gps_sat_tack_names, start_date,filename="track_animation.mp4", frame_interval=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and plot the reflection track of CYGNSS satellites')
    parser.add_argument('--cygnss_satellite_name', type=str, default="cygfm01", help='Name of the CYGNSS satellite to track')
    parser.add_argument('--relevant_PNRs', type=int, nargs='+', default=[20, 10, 32, 14], help='List of PRNs to track')
    parser.add_argument('--date_start', type=str, default="2018-08-22 14:26:59", help='Start date and time for the track')
    parser.add_argument('--date_end', type=str, default="2018-08-22 14:27:59", help='End date and time for the track')
    args = parser.parse_args()

    # date_start = dt.datetime.strptime(args.date_start, "%Y-%m-%d %H:%M:%S")
    # date_end = dt.datetime.strptime(args.date_end, "%Y-%m-%d %H:%M:%S")

    # generate_cygnss_reflection_track_and_plot(args.cygnss_satellite_name, args.relevant_PNRs, date_start, date_end)
    filtered_data = get_all_data_entries_that_crossed_from_sea_to_land("already_processed_satellites_and_dates.json")
    # Print the frist entry
    should_terminate = False
    for satellite_name, entries in filtered_data.items():
        if should_terminate:
            break
        for i, entry in enumerate(entries):
            plot_track_on_basemap(np.array(entry["track"]), "r", f"{satellite_name} track {i}\n {entry['date_start']} - {entry['date_end']}")
            # Ask user if they want to plot the reflection track
            print(f"Length of track: {len(entry['track'])}")
            user_input = input("Do you want to plot the reflection track? (y/[n])")
            if user_input.lower() == "y":
                start_date = dt.datetime.strptime(entry["date_start"], "%Y-%m-%d %H:%M:%S")
                end_date = dt.datetime.strptime(entry["date_end"], "%Y-%m-%d %H:%M:%S")
                generate_cygnss_reflection_track_and_plot(satellite_name, args.relevant_PNRs, start_date, end_date)
            else:
                print("Skipping reflection track plotting.")
            user_input = input("Do you want to continue? ([y]/n)")
            if user_input.lower() == "n":
                should_terminate = True
                break
    # example_entry = filtered_data["cygfm02"]
    # print(f"Example entry: {example_entry[4]}")
    # earth_visualization_map = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=60,
    #             llcrnrlon=10,urcrnrlon=130, resolution='i')

    # earth_visualization_map.drawcoastlines()
    # earth_visualization_map.drawparallels(range(-10, 61, 10), labels=[1, 0, 0, 0], fontsize=10)  # Latitude labels on the left
    # earth_visualization_map.drawmeridians(range(10, 131, 10), labels=[0, 0, 0, 1], fontsize=10)  # Longitude labels on the bottom
    # earth_visualization_map.plot(np.array(example_entry[4]["track"])[:, 0], np.array(example_entry[4]["track"])[:, 1], linewidth=2, color="r", label="Track")    
    # plt.legend()
    # plt.show()
