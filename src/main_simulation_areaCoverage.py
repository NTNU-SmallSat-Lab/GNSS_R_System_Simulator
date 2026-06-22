import logging

import matplotlib
import numpy as np

import functions.functions as fn

matplotlib.use("TkAgg", force=True)

# Application-level logging setup. Library modules use logging.getLogger(__name__) and do not
# configure handlers themselves; this is the single place where the user-visible default is set.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from __init__ import initialize
from object_definitions.Coverage_def import Coverage
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.Static_classes_def import Plotting


#############################################################################################################################
##################################################### - MAIN FUNCTION - #####################################################
#############################################################################################################################
def run_area_coverage_simulation(sim_settings):
    """
    =========================================================================================================
    Run satellite simulation to calculate the coverage of GNSS-R satellites over the Norwegian coast.
    This function initializes the simulation, propagates orbits, calculates reflection points,
    and generates coverage statistics. The simulation is based on the provided simulation settings
    which are loaded from Configuration_area.ini.

    INPUTS:
        sim_settings: [dict]   Dictionary containing simulation parameters
    =========================================================================================================
    """
    earth = "elliptical"  #'spherical'
    print("Simulation start Time:", sim_settings["start_time"])

    ####################################
    # 1. Initialize GNSS constellations
    ####################################
    gnss_constl_dict = fn.load_gnss_constellation_dict(sim_settings["GNSS_constellations"])

    ####################################
    # 2. Create satellite constellation
    ####################################
    gnss_r_antenna = RxAntenna(sim_settings) if sim_settings.get("antenna_pattern_folder") is not None else None
    sat_array = [Satellite(i, orbit_n=sim_settings["orbits"][i], rx_antenna=gnss_r_antenna) for i in range(sim_settings["No_satellites"])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.set_simulation_duration(
        sim_settings["start_time"],
        sim_settings["sim_duration"],
        sim_settings["step_size"],
    )

    ####################################
    # 3. Propagate orbits
    ####################################
    fn.propagate_all_orbits(sat_const, gnss_constl_dict)

    ####################################
    # 4. Calculate sunlight time
    ####################################
    for sat_idx in range(sat_const.no_satellites):
        sat_const.satellites[sat_idx].calculate_sunlight_time()
        print(
            f"Not sunlit count for satellite {sat_idx}: ",
            np.sum(~sat_const.satellites[sat_idx].sunlit_count),
        )
        print(
            f"Sunlit count for satellite {sat_idx}: ",
            np.sum(sat_const.satellites[sat_idx].sunlit_count),
        )
        print(
            f"Sunlit percentage for satellite {sat_idx}: ",
            sat_const.satellites[sat_idx].sunlit_fraction * 100,
        )

    ####################################
    # 5. Calculate reflection points and coverage
    ####################################
    #    coverage = Coverage(sat_const, gnss_constl_dict, sim_settings['shapefile_path'], sim_settings['bounding_box'], grid_size_deg=1)
    coverage = Coverage(
        sat_const,
        gnss_constl_dict,
        sim_settings["shapefile_path"],
        grid_size_m=sim_settings["grid_size_m"],
        earth_model=earth,
        link_budget_settings=sim_settings.get("link_budget"),
        max_central_angle_deg=sim_settings["max_central_angle_deg"],
        selected_signals=sim_settings["selected_signals"],
    )
    if not coverage.coverage_constellations:
        raise ValueError("No usable constellations for coverage. Check [ANTENNA].bands and the loaded GNSS constellations.")

    ####################################
    # 6/7. Per-constellation geometric coverage: compute, plot reflection/coverage maps, print stats.
    #      Coverage is band-independent; [ANTENNA].bands only selects which constellations participate.
    ####################################
    for sig in coverage.coverage_constellations:
        constellation = sig["constellation"]
        coverage.compute_coverage_for_constellation(
            constellation=constellation,
            freq_hz=sig["freq_hz"],
            eirp_dbw=sig["eirp_dbw"],
            verbose=True,
        )
        coverage.plot_coverage_shapefile(gnss_constellation=constellation, file_name=f"Coverage_{constellation}.png")
        coverage.plot_coverage_gridded_shapefile(gnss_constellation=constellation, file_name=f"Coverage_grid_{constellation}.png")
        coverage.calculate_statistics(gnss_constellation=constellation)

    ####################################
    # 7.5. Combined regional coverage across all computed constellations (overlaid on one map).
    ####################################
    present_consts = sorted({sig["constellation"] for sig in coverage.coverage_constellations})
    combined_filter = ", ".join(present_consts)
    coverage.plot_coverage_shapefile(gnss_constellation=combined_filter, file_name="Coverage_all_configurations.png")
    coverage.plot_coverage_gridded_shapefile(gnss_constellation=combined_filter, file_name="Coverage_grid_all_configurations.png")
    coverage.calculate_statistics(gnss_constellation=combined_filter)

    ####################################
    # 8. Whole-earth reflection-trace plots for every computed constellation (geometry is band-independent).
    ####################################
    coverage.plot_reflection_points()

    ####################################
    # 9. Plot orbits (band-independent)
    ####################################
    Plotting.plot_orbits(sat_const)


if __name__ == "__main__":
    config, settings = initialize("Configuration_area.ini")
    run_area_coverage_simulation(settings)
