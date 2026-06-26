"""
Sweep max_bistatic_angle_deg_cc over a range for each Rx constellation.
Propagates orbits ONCE per constellation, then re-runs CC analysis for each beta.

Usage:
    python src/run_beta_sweep.py
"""

import copy
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

matplotlib.use("Agg", force=True)

import functions.functions as fn
from __init__ import initialize
from object_definitions.MovingTargetCellCentric_def import MovingTargetCellCentric
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory

CONFIGS = [
    ("1sat", "Configuration_ship_1sat.ini"),
    ("2sat", "Configuration_ship_2sat.ini"),
    ("4sat", "Configuration_ship_4sat.ini"),
    ("8sat", "Configuration_ship_8sat.ini"),
]

BETA_VALUES = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60]  # None = no constraint


def setup_and_propagate(settings):
    """Run everything up to (but not including) the CC analysis. Returns mtd, gnss_r_ant."""
    ship_trajectory = ShipTrajectory(
        ship_id="DarkShip_001",
        ship_speed_knot=settings["moving_target_speed"],
        start_position=settings["start_coord_ll"],
        start_time=settings["start_time"],
        end_position=settings["end_coord_ll"],
        interval_minutes=10,
        grid_resolution_km=150,
        fast_path_gen_method=False,
    )

    gnss_constl_dict = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
    antenna = GNSS_R_antenna(settings)

    sat_array = [Satellite(i, orbit_n=settings["orbits"][i]) for i in range(settings["No_satellites"])]
    sat_const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sat_array)
    sat_const.generate_constellation_tle()

    travel_duration_h = ship_trajectory.get_travel_duration_hours()
    sat_const.set_simulation_duration(settings["start_time"], travel_duration_h, settings["step_size"])
    for name, gnss_const in gnss_constl_dict.items():
        gnss_const.set_simulation_duration(settings["start_time"], travel_duration_h, settings["step_size"])

    mtd = MovingTargetDetection(
        sat_const, gnss_constl_dict, ship_trajectory, antenna,
        deltaT=settings["step_size"],
        earth_model="elliptical",
        reflection_mode=settings["reflection_type"],
        link_budget_settings=settings.get("link_budget"),
        cutoff_angle=settings.get("cutoff_angle", 45),
        min_tx_elevation_deg=settings.get("min_tx_elevation_deg", 5.0),
        max_bistatic_angle_deg=settings.get("max_bistatic_angle_deg", 0.0),
    )

    rx_elev_min = 5 if settings["reflection_type"] == "backscatter" else 40
    ship_resampled, sat_visible_times = fn.compute_visible_times_and_resample_ship(
        mtd, sat_const, elev_min=rx_elev_min
    )
    fn.propagate_only_visible_satellites(sat_const, sat_visible_times,
                                         gnss_constl_dict=mtd.gnss_constellation)
    return mtd, antenna


def run_cc_for_beta(mtd, antenna, beta):
    cc = MovingTargetCellCentric(mtd, antenna, steer_fraction=0.2,
                                 max_bistatic_angle_deg=beta)
    cc.calculate(verbose=False)
    total = sum(len(dets) for dets in cc.detections.values())
    return total


def main():
    results = {}   # label -> {beta: count}

    for label, config_file in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Constellation: {label}  ({config_file})")
        print(f"{'='*60}")
        _, settings = initialize(config_file)

        print("  Propagating orbits (once)...")
        mtd, antenna = setup_and_propagate(settings)

        results[label] = {}
        for beta in BETA_VALUES:
            count = run_cc_for_beta(mtd, antenna, beta)
            label_str = f"{beta}°" if beta is not None else "None"
            print(f"  beta={label_str:>5}  →  {count} detections")
            results[label][beta] = count

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(CONFIGS)))
    x_vals = [b for b in BETA_VALUES if b is not None]

    for (label, _), col in zip(CONFIGS, colors):
        counts = [results[label][b] for b in x_vals]
        no_constraint = results[label][None]
        ax.plot(x_vals, counts, "o-", color=col, lw=2, ms=6, label=label)
        ax.axhline(no_constraint, color=col, lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Max bistatic angle β [°]", fontsize=12)
    ax.set_ylabel("CC detections", fontsize=12)
    ax.set_title("Cell-centric detections vs bistatic angle constraint\n"
                 "(dashed = no constraint)", fontsize=12)
    ax.legend(title="Rx constellation")
    ax.grid(True, alpha=0.4)
    ax.set_xticks(x_vals)

    out = "OUTPUT/Beta_sweep.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out}")

    # Print summary table
    print(f"\n{'Beta':>6}", end="")
    for label, _ in CONFIGS:
        print(f"  {label:>8}", end="")
    print()
    print("-" * (6 + 10 * len(CONFIGS)))
    for beta in BETA_VALUES:
        label_str = f"{beta}°" if beta is not None else "None"
        print(f"{label_str:>6}", end="")
        for label, _ in CONFIGS:
            print(f"  {results[label][beta]:>8}", end="")
        print()


if __name__ == "__main__":
    main()
