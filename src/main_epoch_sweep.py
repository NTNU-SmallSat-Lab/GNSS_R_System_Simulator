"""
Epoch-offset sweep to remove timing bias from moving-target detection simulations.

Runs N_RUNS simulations with the start epoch shifted by evenly spaced fractions of
one orbital period (~96 min for 568 km LEO). Each run saves its own detection plot.
A final aggregated plot overlays all detections from every run.

Run from project root:
    python src/main_epoch_sweep.py
"""

import os
import sys
sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from __init__ import initialize
from main_simulation_movingTarget import run_moving_target_detection, _format_footprint_db_label
from object_definitions.Constants_def import Constants
import functions.functions as fn

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
ORBITAL_PERIOD_MIN = 95.9      # minutes — 568 km LEO, 15.017 rev/day
N_RUNS             = 8         # one per plane; step = period / N_RUNS ≈ 12 min
STEP_MIN           = ORBITAL_PERIOD_MIN / N_RUNS   # ~12 min

# ---------------------------------------------------------------------------
# Load settings once
# ---------------------------------------------------------------------------
_, settings = initialize("Configuration_ship.ini")
footprint_db_label = _format_footprint_db_label(settings.get("footprint_dB"))
mode = settings["reflection_type"]

print(f"Epoch sweep: {N_RUNS} runs, step = {STEP_MIN:.1f} min "
      f"(full period = {ORBITAL_PERIOD_MIN:.1f} min)")
print(f"Reflection mode : {mode}")
print(f"Base start time : {settings['start_time']}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Run sweep — collect all detection structs
# ---------------------------------------------------------------------------
all_detections   = []   # flat list of MovingTargetDetectionStruct
run_detect_counts = []

for i in range(N_RUNS):
    offset_min = i * STEP_MIN
    suffix     = f"offset{int(round(offset_min)):03d}min"
    print(f"\n[Run {i+1}/{N_RUNS}]  epoch offset = {offset_min:.1f} min  →  {suffix}")

    mtd = run_moving_target_detection(
        settings,
        epoch_offset_minutes=offset_min,
        output_suffix=suffix,
    )

    n = sum(len(v) for v in mtd.target_detect_dict.values())
    run_detect_counts.append(n)
    print(f"  Detections this run: {n}")

    for gnss_name, detections in mtd.target_detect_dict.items():
        all_detections.extend(detections)

print("\n" + "=" * 60)
print(f"Total detections across all runs: {len(all_detections)}")
for i, n in enumerate(run_detect_counts):
    print(f"  Run {i+1} (offset {i*STEP_MIN:.0f} min): {n} detections")

# ---------------------------------------------------------------------------
# Aggregated detection plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
west, east, south, north = -10, 35, 35, 82
ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

ax.set_title(
    f"Aggregated Detection Points — {N_RUNS} epoch offsets × {STEP_MIN:.0f} min\n"
    f"({mode}, footprint {footprint_db_label})"
)
ax.add_feature(cfeature.LAND,      facecolor="lightgray")
ax.add_feature(cfeature.OCEAN,     facecolor="lightblue")
ax.add_feature(cfeature.LAKES,     facecolor="lightblue")
ax.add_feature(cfeature.BORDERS,   linestyle=":")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
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

# Plot all detections (semi-transparent so density is visible)
lats = [d.position_ll[0] for d in all_detections if d.position_ll is not None and not np.any(np.isnan(d.position_ll))]
lons = [d.position_ll[1] for d in all_detections if d.position_ll is not None and not np.any(np.isnan(d.position_ll))]

if lats:
    ax.scatter(lons, lats, c="red", s=12, alpha=0.25, marker="o",
               label=f"Detections (n={len(lats)})", transform=ccrs.PlateCarree(), zorder=5)

ax.legend(loc="upper right")

output_agg = f"Detection_points_{mode}_footprint_{footprint_db_label}_aggregated_{N_RUNS}runs.png"
out_path = os.path.join(Constants.output_folder_path, output_agg)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nAggregated plot saved as: {output_agg}")
