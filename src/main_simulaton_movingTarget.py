import os
import numpy               as np
import functions.functions as fn

import matplotlib
matplotlib.use('TkAgg', force=True)
from datetime                                     import timedelta
from __init__                                     import initialize
from object_definitions.Satellite_def             import Satellite
from object_definitions.SatConstellation_def      import SatelliteConstellation
from object_definitions.Coverage_def              import Coverage
from object_definitions.Static_classes_def        import Plotting
from object_definitions.ShipTrajectory_def        import ShipTrajectory
from object_definitions.MovingTargetDetection     import MovingTargetDetection

#############################################################################################################################
##################################################### - MAIN FUNCTION - #####################################################
#############################################################################################################################
def Skagerrak_to_BarentsSea(settings):
    '''
    This function calculates the probability of detecting a ship traveling from the
      Skagerrak sea (Denmark/Norway) to the Barents Sea (Norway/Russia).

      NOTE: 
        Shapefile data download from https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/
    '''
    Helgoland   = (54.199416, 7.892937) 
    Murmansk    = (69.345572, 33.646450)
    Oslo        = (59.9139,   10.7522)
    Stavanger   = (58.9700,   5.7331)
    Skagerrak   = (58.0,      10.0)        # Approximate center of Skagerrak sea
    Kaliningrad = (55.0,      20.0)        # Coast in front of Kaliningrad (Koenigsberg)

    ########################################################################
    # 1: Calculate ship trajectory from Skagerrak to Barents Sea
    ########################################################################
    shipTrajectory = ShipTrajectory(
        ship_id              = "DarkShip_001",         # Ship ID
        ship_speed_knot      = 15.0,                   # 15 knots ship speed
        start_position       = Helgoland,              # Start position (latitude, longitude)
        start_time           = settings['start_time'], # Start time (UTC)
        end_position         = Murmansk,               # End position (latitude, longitude)
        interval_minutes     = 10,                     # Time interval in minutes between each position
        grid_resolution_km   = 150,                    # Grid resolution in degrees latitude/longitude
        fast_path_gen_method = False                   # Use fast path generation method (True) or slow method which is more accurate (False)
    )
    shipTrajectory.plot_route()

    ####################################
    # 2: Load simulation settings
    ####################################
    gnss_constl_dict = fn.load_gnss_constellation_dict(settings['GNSS_constellations'])

    ####################################
    # 3. Create satellite constellation
    ####################################
    sat_array = [Satellite(i, orbit_n=settings['orbits'][i]) 
                           for i in range(settings['No_satellites'])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_TLE()

    ########################################################################
    # 4: Set simulation duration for the GNSS constellations
    ########################################################################
    travel_duration_h = shipTrajectory.get_travel_duration_hours() # Add 1 hour for the simulation duration
    for name, gnss_const in gnss_constl_dict.items():
        gnss_const.set_simulation_duration(settings['start_time'], travel_duration_h, settings['step_size'])
        print(f"Simulation duration for GNSS constellation: \"{name}\" set to {gnss_const.duration_sec} seconds.")
    sat_const.set_simulation_duration(settings['start_time'], travel_duration_h, settings['step_size'])

    ####################################
    # 5. Propagate orbits
    ####################################
    fn.propagate_all_orbits(sat_const, gnss_constl_dict, settings['start_time'], 
                            travel_duration_h, settings['step_size'])

    ####################################
    # 6. Calculate sunlight time
    #################################### # TODO improve this also
    for sat_idx in range(sat_const.no_satellites):
        sat_const.satellites[sat_idx].calculate_sunlight_time()
        print(f"Not sunlit count for satellite {sat_idx}: ",     np.sum(sat_const.satellites[sat_idx].sunlit_count == False))
        print(f"Sunlit count for satellite {sat_idx}: ",         np.sum(sat_const.satellites[sat_idx].sunlit_count == True))
        print(f"Sunlit percentage for satellite {sat_idx}: ",    sat_const.satellites[sat_idx].sunlit_fraction * 100)

    ##########################################
    # 7. Moving target detection probablility
    ##########################################
    moving_target_detection = MovingTargetDetection(sat_const, gnss_constl_dict, shipTrajectory, footprint_size_m=300_000, deltaT=settings['step_size'])
    moving_target_detection.calculate_moving_target_reflection_intersection()
    moving_target_detection.show_track_and_detection_points(file_name="Detection_points.png", show_figure=False)
    time_1h       = fn.ensure_utc_aware(shipTrajectory.start_time + timedelta(hours=1))
    moving_target_detection.show_reflections_and_target_position_at_time_i(time_1h, file_name="Reflections_1h_into_journey", label_reflections=True)
    moving_target_detection.create_reflection_animation(time_step_min=5, output_filename="Reflection_animation.gif")

#    ####################################
#    # 8. Calculate reflection points and coverage
#    ####################################
    coverage = Coverage(sat_const, gnss_constl_dict, settings['shapefile_path'], settings['bounding_box'], grid_size_deg=1)
    coverage.calculate_reflection_points_and_coverage(verbose=True)

#    ####################################
#    # 9. Plot reflection points
#    ####################################
    coverage.plot_reflection_points()
    coverage.plot_coverage_shapefile(gnss_constellation="GPS", file_name="Coverage.png")
    coverage.plot_coverage_shapefile(file_name="Coverage_both.png")
    coverage.plot_coverage_gridded_shapefile(gnss_constellation="GPS", file_name="Coverage_grid_GPS.png")
    coverage.plot_coverage_gridded_shapefile(file_name="Coverage_grid_GPS_GALIEO.png")

#    ####################################
#    # 10. Calculate statistics
#    ####################################
    coverage.calculate_statistics()

    ####################################
    # 11. Plot orbits
    ####################################
    Plotting.plot_orbits(sat_const)

if  __name__ == "__main__":
    config, settings = initialize('Configuration_ship.ini')
    Skagerrak_to_BarentsSea(settings)
