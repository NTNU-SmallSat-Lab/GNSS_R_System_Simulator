"""
Epoch-offset sweep with varying numbers of Rx satellites.
Tests detection performance with 1, 2, 4, and 8 satellites using CELESTE_IOD3.tle.

Run from project root:
    python src/main_epoch_sweep_sat_comparison.py
"""

import os
import sys
sys.path.insert(0, "src")

import copy
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from __init__ import initialize
from main_simulation_movingTarget import run_moving_target_detection, _format_footprint_db_label
from object_definitions.Constants_def import Constants

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
ORBITAL_PERIOD_MIN = 95.9      # minutes — 568 km LEO
N_RUNS             = 8         # one per plane; step = period / N_RUNS ≈ 12 min
STEP_MIN           = ORBITAL_PERIOD_MIN / N_RUNS   # ~12 min
SAT_COUNTS         = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Load base settings
# ---------------------------------------------------------------------------
_, base_settings = initialize("Configuration_ship.ini")

# Change TLE file to CELESTE_IOD3.tle
base_settings["GNSS_constellations"] = ["CELESTE_IOD3.tle"]

footprint_db_label = _format_footprint_db_label(base_settings.get("footprint_dB"))
mode = base_settings["reflection_type"]

print(f"Satellite count comparison epoch sweep")
print(f"TLE file: CELESTE_IOD3.tle")
print(f"Reflection mode: {mode}")
print(f"Epoch sweep: {N_RUNS} runs, step = {STEP_MIN:.1f} min")
print("=" * 70)

# Store results for each satellite count
all_results = {}

# ---------------------------------------------------------------------------
# Run sweep for each satellite count
# ---------------------------------------------------------------------------
for sat_count in SAT_COUNTS:
    print(f"\n{'='*70}")
    print(f"SATELLITE COUNT: {sat_count}")
    print(f"{'='*70}")

    # Create a modified settings dict with only the specified number of satellites
    settings = copy.deepcopy(base_settings)
    settings["No_satellites"] = sat_count

    # Rebuild orbits dict to include only the first sat_count orbits
    orbits_dict = {}
    for i in range(sat_count):
        orbit_key = f"orbit{i+1}"
        if orbit_key in settings["orbits"]:
            orbits_dict[i] = settings["orbits"][i]
    settings["orbits"] = orbits_dict

    all_detections = []
    run_detect_counts = []

    for i in range(N_RUNS):
        offset_min = i * STEP_MIN
        suffix = f"iod3_sat{sat_count}_offset{int(round(offset_min)):03d}min"
        print(f"  [Run {i+1}/{N_RUNS}]  offset = {offset_min:.1f} min  →  {suffix}")

        mtd = run_moving_target_detection(
            settings,
            epoch_offset_minutes=offset_min,
            output_suffix=suffix,
        )

        n = sum(len(v) for v in mtd.target_detect_dict.values())
        run_detect_counts.append(n)
        print(f"    Detections: {n}")

        for gnss_name, detections in mtd.target_detect_dict.items():
            all_detections.extend(detections)

    total = len(all_detections)
    avg_per_run = total / N_RUNS if N_RUNS > 0 else 0
    print(f"\n  Total detections (all runs): {total}")
    print(f"  Average per run: {avg_per_run:.1f}")
    print(f"  Per-run breakdown: {run_detect_counts}")

    all_results[sat_count] = {
        "detections": all_detections,
        "total": total,
        "per_run": run_detect_counts,
        "average": avg_per_run,
    }

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMMARY: Detection counts by satellite count")
print(f"{'='*70}")
for sat_count in SAT_COUNTS:
    result = all_results[sat_count]
    print(f"{sat_count:1d} sat: {result['total']:3d} total  ({result['average']:5.1f} avg/run)  "
          f"per-run: {result['per_run']}")

# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                         subplot_kw={"projection": ccrs.PlateCarree()})
axes = axes.flatten()

west, east, south, north = -10, 35, 35, 82

for ax_idx, sat_count in enumerate(SAT_COUNTS):
    ax = axes[ax_idx]
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

    ax.set_title(f"{sat_count} Rx Satellite{'s' if sat_count > 1 else ''} "
                 f"({all_results[sat_count]['total']} detections)")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Ship trajectory
    _, sim_settings = initialize("Configuration_ship.ini")
    from object_definitions.ShipTrajectory_def import ShipTrajectory
    ship = ShipTrajectory(
        ship_id="DarkShip_001",
        ship_speed_knot=sim_settings["moving_target_speed"],
        start_position=sim_settings["start_coord_ll"],
        start_time=sim_settings["start_time"],
        end_position=sim_settings["end_coord_ll"],
        interval_minutes=10, grid_resolution_km=150, fast_path_gen_method=False,
    )
    traj_x = [p[1] for p in ship.trajectory_points]
    traj_y = [p[0] for p in ship.trajectory_points]
    ax.plot(traj_x, traj_y, "b-", linewidth=1.5, alpha=0.8, label="Trajectory",
            transform=ccrs.PlateCarree())

    # Plot detections
    detections = all_results[sat_count]["detections"]
    lats = [d.position_ll[0] for d in detections if d.position_ll is not None and not np.any(np.isnan(d.position_ll))]
    lons = [d.position_ll[1] for d in detections if d.position_ll is not None and not np.any(np.isnan(d.position_ll))]

    if lats:
        ax.scatter(lons, lats, c="red", s=12, alpha=0.25, marker="o",
                   transform=ccrs.PlateCarree(), zorder=5)

    ax.legend(loc="upper right", fontsize=9)

plt.suptitle(f"CELESTE_IOD3.tle — Detection by Rx Satellite Count\n"
             f"({mode}, footprint {footprint_db_label}, {N_RUNS} epoch offsets)",
             fontsize=12, fontweight="bold")
plt.tight_layout()

output_file = f"Detection_comparison_iod3_sat_counts_{mode}_footprint_{footprint_db_label}.png"
out_path = os.path.join(Constants.output_folder_path, output_file)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nComparison plot saved as: {output_file}")

print("\n" + "=" * 70)
print("Epoch sweep with satellite count comparison completed.")
