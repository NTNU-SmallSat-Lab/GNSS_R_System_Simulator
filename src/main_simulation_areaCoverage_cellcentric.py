import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

import functions.functions as fn

matplotlib.use("Agg", force=True)
from __init__ import initialize
from object_definitions.Coverage_def import Coverage
from object_definitions.CoverageCellCentric_def import CoverageCellCentric
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.Static_classes_def import Plotting


def _map_panel(ax, grid_gdf, counts, cmap, norm, title):
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    for i, row in grid_gdf.iterrows():
        if counts[i] == 0:
            continue
        xs, ys = row["geometry"].exterior.xy
        ax.fill(xs, ys, color=cmap(norm(counts[i])),
                transform=ccrs.PlateCarree(), zorder=3)
    nonzero = counts[counts > 0]
    stats = f"min={nonzero.min() if len(nonzero) else 0}  " \
            f"median={int(np.median(nonzero)) if len(nonzero) else 0}  " \
            f"max={counts.max()}" if counts.max() > 0 else "no coverage"
    ax.set_title(f"{title}\n{stats}", fontsize=9)


def plot_revisit_count(coverage, cell_centric, tag, sim_settings):
    """Side-by-side: Tx-beam-only illumination (reference) vs Tx+Rx joint coverage."""
    gnss_name = list(coverage.gnss_constellation.keys())[0]
    n = len(coverage.grid_gdf)

    txrx = np.array([len(coverage.coverage_dict[gnss_name][i].times) for i in range(n)])
    tx   = cell_centric.tx_only_counts.get(gnss_name, np.zeros(n, dtype=int))

    vmax = max(tx.max(), txrx.max(), 1)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                             subplot_kw={"projection": ccrs.PlateCarree()})

    _map_panel(axes[0], coverage.grid_gdf, tx,   cmap, norm, "Tx beam only  (reference)")
    _map_panel(axes[1], coverage.grid_gdf, txrx, cmap, norm,
               f"Tx + steered Rx beam  (steer_fraction={sim_settings.get('steer_fraction', 0.2)})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cb.set_label("Illumination count (timesteps)")

    out_path = f"OUTPUT/Revisit_count{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Revisit count plot saved to {out_path}")


def run_area_coverage_simulation_cellcentric(sim_settings, output_tag=""):
    """
    Run area coverage simulation and then apply cell-centric analysis.
    First runs the standard trajectory-based calculation, then overlays
    cell-centric geometric constraint analysis.

    INPUTS:
        sim_settings: [dict]   Simulation settings from config file
        output_tag: [str]      Tag for output filenames
    """
    earth = "elliptical"
    print("Simulation start Time:", sim_settings["start_time"])

    # 1. Initialize GNSS constellations
    gnss_constl_dict = fn.load_gnss_constellation_dict(sim_settings["GNSS_constellations"])

    # 2. Create satellite constellation
    gnss_r_antenna = RxAntenna(sim_settings) if sim_settings.get("antenna_pattern_folder") is not None else None
    sat_array = [Satellite(i, orbit_n=sim_settings["orbits"][i], rx_antenna=gnss_r_antenna) for i in range(sim_settings["No_satellites"])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_tle()
    sat_const.set_simulation_duration(
        sim_settings["start_time"],
        sim_settings["sim_duration"],
        sim_settings["step_size"],
    )

    # 3. Propagate orbits
    fn.propagate_all_orbits(sat_const, gnss_constl_dict)

    # 4. Calculate sunlight time
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

    # 5. Calculate reflection points and coverage
    coverage = Coverage(
        sat_const,
        gnss_constl_dict,
        sim_settings["shapefile_path"],
        grid_size_m=30_000,
        earth_model=earth,
        link_budget_settings=sim_settings.get("link_budget"),
    )
    coverage.calculate_reflection_points_and_coverage(verbose=True)

    # 6. Apply cell-centric analysis
    print("\n" + "=" * 80)
    print("APPLYING CELL-CENTRIC COVERAGE ANALYSIS")
    print("=" * 80)

    # Reset coverage_dict so trajectory-based results don't contaminate cell-centric output
    for gnss_name in coverage.gnss_constellation:
        for cell in coverage.coverage_dict[gnss_name]:
            cell.times.clear()
            cell.tx_satID.clear()
            cell.rx_satID.clear()
            cell.angles.clear()
            cell.power.clear()

    steer_fraction = 0.2
    times = [time.utc_datetime() for time in sat_const.times_skyfield]
    cell_centric = CoverageCellCentric(
        coverage, antenna=gnss_r_antenna, steer_fraction=steer_fraction,
        link_budget_settings=sim_settings.get("link_budget"),
        nadir_pointing=sim_settings.get("nadir_pointing", False),
    )
    cell_centric.calculate_coverage_cell_centric(times, verbose=True)

    # 7. Plot results
    tag = f"_{output_tag}_cellcentric" if output_tag else "_cellcentric"
    coverage.plot_coverage_shapefile(file_name=f"Coverage{tag}.png")
    coverage.plot_coverage_gridded_shapefile(file_name=f"Coverage_grid{tag}.png")
    sim_settings["steer_fraction"] = steer_fraction
    plot_revisit_count(coverage, cell_centric, tag, sim_settings)
    Plotting.plot_orbits(sat_const)

    # Save count arrays for cross-scenario comparison plots
    gnss_name = list(coverage.gnss_constellation.keys())[0]
    n = len(coverage.grid_gdf)
    txrx_counts = np.array([len(coverage.coverage_dict[gnss_name][i].times) for i in range(n)])
    tx_counts   = cell_centric.tx_only_counts.get(gnss_name, np.zeros(n, dtype=int))
    np.save(f"OUTPUT/counts_txrx{tag}.npy", txrx_counts)
    np.save(f"OUTPUT/counts_tx{tag}.npy",   tx_counts)
    print(f"Count arrays saved: OUTPUT/counts_tx{tag}.npy  &  counts_txrx{tag}.npy")

    # Save SNR arrays if link budget was enabled
    if cell_centric.snr_max:
        snr_arr = cell_centric.snr_max.get(gnss_name, np.full(n, -np.inf))
        np.save(f"OUTPUT/snr_max{tag}.npy", snr_arr)
        finite = snr_arr[np.isfinite(snr_arr)]
        if len(finite):
            print(f"SNR saved: min={finite.min():.1f} dB  median={np.median(finite):.1f} dB  "
                  f"max={finite.max():.1f} dB  ({len(finite)} covered cells)")
        else:
            print("SNR saved: no cells with finite SNR")

    # 8. Calculate statistics
    coverage.calculate_statistics()

    print("\n" + "=" * 80)
    print("CELL-CENTRIC SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Configuration_area.ini"
    tag = os.path.splitext(os.path.basename(config_file))[0].replace("Configuration_", "")

    config, settings = initialize(config_file)
    run_area_coverage_simulation_cellcentric(settings, output_tag=tag)
