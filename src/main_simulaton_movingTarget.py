import os
import numpy               as np
import functions.functions as fn

import matplotlib
matplotlib.use('TkAgg', force=True)
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

    # 1: Calculate ship trajectory from Skagerrak to Barents Sea
    # Create ship trajectory
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

    # 2: Load simulation settings
    gnss_constl_dict = fn.load_gnss_constellation_dict(settings['GNSS_constellations'])

    # 3. Create satellite constellation
    sat_array = [Satellite(i, orbit_n=settings['orbits'][i]) 
                           for i in range(settings['No_satellites'])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_TLE()
    sat_const.set_simulation_duration(settings['start_time'], settings['sim_duration'], settings['step_size'])

    # 4. Propagate orbits
    fn.propagate_all_orbits(sat_const, gnss_constl_dict, settings['start_time'], 
                            shipTrajectory.get_travel_duration_hours(), settings['step_size'])

    ####################################
    # 5. Calculate sunlight time
    #################################### # TODO improve this also
    for sat_idx in range(sat_const.no_satellites):
        sat_const.satellites[sat_idx].calculate_sunlight_time()
        print(f"Not sunlit count for satellite {sat_idx}: ",     np.sum(sat_const.satellites[sat_idx].sunlit_count == False))
        print(f"Sunlit count for satellite {sat_idx}: ",         np.sum(sat_const.satellites[sat_idx].sunlit_count == True))
        print(f"Sunlit percentage for satellite {sat_idx}: ",    sat_const.satellites[sat_idx].sunlit_fraction * 100)

    ##########################################
    # 6. Moving target detection probablility
    ##########################################
    moving_target_detection = MovingTargetDetection(sat_const, gnss_constl_dict, shipTrajectory, footprint_size_m=300_000, create_shapefile_for_visualization=True)
    moving_target_detection.calculate_moving_target_reflection_intersection(verbose=True)
    moving_target_detection.show_reflections_and_target_position_at_time_i()
    moving_target_detection.plot_detection_points(file_name="Coverage.png")
    moving_target_detection.plot_time_until_detection(gnss_constellation="GALILEO", file_name="Coverage_time_until_detection.png")

#    ####################################
#    # 6. Calculate reflection points and coverage
#    ####################################
    coverage = Coverage(sat_const, gnss_constl_dict, settings['shapefile_path'], settings['bounding_box'], grid_size_deg=1)
    coverage.calculate_reflection_points_and_coverage(verbose=True)

#    ####################################
#    # 7. Plot reflection points
#    ####################################
#    coverage.plot_reflection_points()
#    coverage.plot_coverage_shapefile(gnss_constellation="GPS", file_name="Coverage.png")
#    coverage.plot_coverage_shapefile(file_name="Coverage_both.png")
#    coverage.plot_coverage_gridded_shapefile(gnss_constellation="GPS", file_name="Coverage_grid_GPS.png")
#    coverage.plot_coverage_gridded_shapefile(file_name="Coverage_grid_GPS_GALIEO.png")

#    ####################################
#    # 9. Calculate statistics
#    ####################################
#    coverage.calculate_statistics()

    ####################################
    # 10. Plot orbits
    ####################################
    Plotting.plot_orbits(sat_const)

    # 2: Download right TLE file from CelesTrak

    # 3: Propagate Satellite constellations

    # 4: Calculate the coverage of the satellite constellation over the ship trajectory

    # 5: Calculate the probability of detecting the ship using the satellite constellation

    # 6: Save the results

    # 7: Go to step 2 and repeat until the ship reaches the Barents Sea

    # 8: Plot the ship trajectory and coverage area

    # 9: Save the results to a file
    pass

if  __name__ == "__main__":
    config, settings = initialize('Configuration_ship.ini')
    Skagerrak_to_BarentsSea(settings)
