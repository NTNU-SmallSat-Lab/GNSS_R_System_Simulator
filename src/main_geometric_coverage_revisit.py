#!/usr/bin/env python3
"""
GEOMETRIC COVERAGE WITH REVISIT TIME ANALYSIS

Computes coverage based on:
1. Tx beam geometry (elevation ≥ 30°, antenna pattern)
2. Rx beam geometry (steered toward horizon)
3. Bistatic angle constraint (β ≤ 10°)
4. Revisit time and pass count for each cell
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

import functions.functions as fn
from __init__ import initialize
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.Coverage_def import Coverage
from object_definitions.GeometricCoverageRevisit_def import GeometricCoverageRevisit


def run_geometric_coverage_revisit_simulation(sim_settings, output_tag=""):
    """
    Run geometric coverage simulation with revisit time analysis.

    INPUTS:
        sim_settings: [dict] Simulation settings from config file
        output_tag: [str] Tag for output filenames
    """
    print("\n" + "=" * 80)
    print("GEOMETRIC COVERAGE SIMULATION WITH REVISIT TIME ANALYSIS")
    print("=" * 80)
    print("Computing coverage based on antenna beam geometry and bistatic constraints")
    print("Tracking: pass count, revisit times, and illumination statistics")
    print("=" * 80 + "\n")

    earth = "elliptical"

    # 1. Load GNSS constellations
    print("Loading GNSS constellations...")
    gnss_constl_dict = fn.load_gnss_constellation_dict(sim_settings["GNSS_constellations"])

    # 2. Create Rx satellite constellation
    print("Creating Rx satellite constellation...")
    gnss_r_antenna = RxAntenna(sim_settings) if sim_settings.get("antenna_pattern_folder") is not None else None
    sat_array = [Satellite(i, orbit_n=sim_settings["orbits"][i], rx_antenna=gnss_r_antenna) for i in range(sim_settings["No_satellites"])]
    rx_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    rx_const.generate_constellation_tle()
    rx_const.set_simulation_duration(
        sim_settings["start_time"],
        sim_settings["sim_duration"],
        sim_settings["step_size"],
    )

    # 3. Propagate orbits
    print("Propagating orbits...")
    fn.propagate_all_orbits(rx_const, gnss_constl_dict)

    # 4. Calculate sunlight
    print("Calculating sunlight visibility...")
    for sat_idx in range(rx_const.no_satellites):
        rx_const.satellites[sat_idx].calculate_sunlight_time()
        print(
            f"  Rx satellite {sat_idx}: {np.sum(rx_const.satellites[sat_idx].sunlit_count)} sunlit samples"
        )

    # 5. Load grid
    print("\nLoading grid cells...")
    coverage = Coverage(
        rx_const,
        gnss_constl_dict,
        sim_settings["shapefile_path"],
        grid_size_m=30_000,
        earth_model=earth,
        link_budget_settings=sim_settings.get("link_budget"),
    )

    times = [time.utc_datetime() for time in rx_const.times_skyfield]
    print(f"Time steps: {len(times)} (24 hours @ 30 second intervals)")
    print(f"Grid cells: {len(coverage.grid_gdf)}")

    # 6. Compute geometric coverage with revisit analysis
    print("\n" + "=" * 80)
    print("COMPUTING GEOMETRIC COVERAGE WITH REVISIT TIME ANALYSIS")
    print("=" * 80)

    geom_cov = GeometricCoverageRevisit(
        sat_const=rx_const,
        grid_gdf=coverage.grid_gdf,
        min_tx_elevation_deg=sim_settings.get("min_tx_elevation_deg", 30.0),
        max_bistatic_angle_deg=sim_settings.get("max_bistatic_angle_deg", 10.0),
    )

    # Prepare Tx positions from GNSS constellations
    tx_pos_dict = {}
    for gnss_name, gnss_const in gnss_constl_dict.items():
        tx_positions = np.stack([sat.positions_eci for sat in gnss_const.satellites], axis=1)
        tx_pos_dict[gnss_name] = tx_positions

    # Calculate geometric coverage with revisit tracking
    geom_cov.calculate_geometric_coverage_with_revisit(times, tx_pos_dict, verbose=True)

    # 7. Print statistics
    geom_cov.print_statistics()

    print("\n" + "=" * 80)
    print("GEOMETRIC COVERAGE WITH REVISIT TIME SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Configuration_area.ini"
    tag = os.path.splitext(os.path.basename(config_file))[0].replace("Configuration_", "")

    config, settings = initialize(config_file)
    run_geometric_coverage_revisit_simulation(settings, output_tag=tag)
