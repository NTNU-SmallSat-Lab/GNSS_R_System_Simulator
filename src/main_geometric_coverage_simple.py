#!/usr/bin/env python3
"""
GEOMETRIC COVERAGE - Track reflection count per cell
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
from object_definitions.GeometricCoverage_def import GeometricCoverage

def run_geometric_coverage_simulation(sim_settings, output_tag=""):
    """
    Run geometric coverage simulation and track reflection count per cell.
    """
    print("\n" + "="*80)
    print("GEOMETRIC COVERAGE SIMULATOR - REFLECTION COUNT ANALYSIS")
    print("="*80)
    print("Tracking how many times each cell is illuminated")
    print("="*80 + "\n")

    earth = "elliptical"

    # Load GNSS constellations
    print("Loading GNSS constellations...")
    gnss_constl_dict = fn.load_gnss_constellation_dict(sim_settings["GNSS_constellations"])

    # Create Rx satellite constellation
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

    # Propagate orbits
    print("Propagating orbits...")
    fn.propagate_all_orbits(rx_const, gnss_constl_dict)

    # Calculate sunlight
    print("Calculating sunlight visibility...")
    for sat_idx in range(rx_const.no_satellites):
        rx_const.satellites[sat_idx].calculate_sunlight_time()
        print(
            f"  Rx satellite {sat_idx}: {np.sum(rx_const.satellites[sat_idx].sunlit_count)} sunlit samples"
        )

    # Load grid
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

    # Compute geometric coverage
    print("\n" + "="*80)
    print("COMPUTING GEOMETRIC COVERAGE")
    print("="*80)

    geom_cov = GeometricCoverage(
        sat_const=rx_const,
        grid_gdf=coverage.grid_gdf,
        min_tx_elevation_deg=sim_settings.get("min_tx_elevation_deg", 30.0),
        max_bistatic_angle_deg=sim_settings.get("max_bistatic_angle_deg", 10.0),
    )

    # Prepare Tx positions
    tx_pos_dict = {}
    for gnss_name, gnss_const in gnss_constl_dict.items():
        tx_positions = np.stack([sat.positions_eci for sat in gnss_const.satellites], axis=1)
        tx_pos_dict[gnss_name] = tx_positions

    # Calculate coverage
    geom_cov.calculate_geometric_coverage(times, tx_pos_dict, verbose=True)

    # Extract reflection counts
    reflection_counts = np.zeros(len(coverage.grid_gdf))
    for cell_idx, details in geom_cov.coverage_details.items():
        reflection_counts[cell_idx] = len(details)

    # Print statistics
    print("\n" + "="*80)
    print("REFLECTION COUNT STATISTICS")
    print("="*80)

    print(f"\nCoverage Statistics:")
    print(f"  Total cells: {len(coverage.grid_gdf)}")
    print(f"  Cells with ≥1 reflection: {np.sum(reflection_counts > 0)}")
    print(f"  Coverage percentage: {100 * np.sum(reflection_counts > 0) / len(coverage.grid_gdf):.2f}%")

    print(f"\nReflection Count Statistics:")
    valid_counts = reflection_counts[reflection_counts > 0]
    if len(valid_counts) > 0:
        print(f"  Min reflections per cell: {np.min(valid_counts):.0f}")
        print(f"  Max reflections per cell: {np.max(valid_counts):.0f}")
        print(f"  Mean reflections per cell: {np.mean(valid_counts):.2f}")
        print(f"  Median reflections per cell: {np.median(valid_counts):.0f}")
        print(f"  Total reflections: {np.sum(reflection_counts):.0f}")

        # Distribution
        unique_counts = sorted(set(valid_counts.astype(int)))
        print(f"\n  Reflection count distribution:")
        for count in unique_counts[:20]:  # Show first 20 levels
            num_cells = np.sum(reflection_counts == count)
            pct = 100 * num_cells / len(valid_counts)
            print(f"    {count:4.0f} reflections: {num_cells:4d} cells ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("GEOMETRIC COVERAGE SIMULATOR COMPLETE")
    print("="*80)

    return coverage.grid_gdf, reflection_counts


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Configuration_area.ini"
    tag = os.path.splitext(os.path.basename(config_file))[0].replace("Configuration_", "")

    config, settings = initialize(config_file)
    grid_gdf, reflection_counts = run_geometric_coverage_simulation(settings, output_tag=tag)

    # Save results for visualization
    import pickle
    with open(f"OUTPUT/reflection_counts_{tag}.pkl", "wb") as f:
        pickle.dump((grid_gdf, reflection_counts), f)
    print(f"\n✓ Results saved to: OUTPUT/reflection_counts_{tag}.pkl")
