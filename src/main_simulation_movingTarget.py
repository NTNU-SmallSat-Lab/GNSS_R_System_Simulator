import matplotlib
import numpy as np
from datetime import timedelta

import functions.functions as fn
from __init__ import initialize
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.MovingTargetCellCentric_def import MovingTargetCellCentric
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory
from object_definitions.Static_classes_def import Plotting

# Force non-interactive backend for server/headless environments
matplotlib.use("Agg", force=True)


def run_moving_target_detection(settings: dict, epoch_offset_minutes: float = 0, output_suffix: str = ""):
    """
    =========================================================================================================
    Calculate the probability of detecting a ship traveling from point A to B using GNSS-R LEO satellites.

    INPUTS:
        settings:               [dict]    Simulation settings loaded from Configuration_ship.ini
        epoch_offset_minutes:   [float]   Shift the simulation start time by this many minutes (default: 0)
        output_suffix:          [str]     Appended to the output plot filename before the extension

    OUTPUT:
        cc: [MovingTargetCellCentric]  Completed cell-centric detection object (detections in cc.detections)
    =========================================================================================================
    """

    # --- 0. Apply epoch offset ---
    import copy
    settings = copy.copy(settings)
    if epoch_offset_minutes:
        settings["start_time"] = settings["start_time"] + timedelta(minutes=epoch_offset_minutes)

    # --- 1. Environment & Ship Setup ---
    earth_model = "elliptical"

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
    ship_trajectory.plot_route()

    # --- 2. GNSS & Satellite Constellation Setup ---
    gnss_constl_dict = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
    gnss_r_antenna = GNSS_R_antenna(settings)

    # Initialize LEO satellites from settings
    sat_array = [
        Satellite(i, orbit_n=settings["orbits"][i]) 
        for i in range(settings["No_satellites"])
    ]
    sat_const = SatelliteConstellation(
        satConfigName="LEO GNSS-R constellation", 
        satArray=sat_array
    )
    sat_const.generate_constellation_tle()

    # --- 3. Simulation Timing ---
    travel_duration_h = ship_trajectory.get_travel_duration_hours()
    step_size = settings["step_size"]
    start_time = settings["start_time"]

    # Apply timing to all constellations
    sat_const.set_simulation_duration(start_time, travel_duration_h, step_size)
    
    for name, gnss_const in gnss_constl_dict.items():
        gnss_const.set_simulation_duration(start_time, travel_duration_h, step_size)
        print(f"[{name}] Duration set to {gnss_const.duration_sec / 3600:.2f} hours.")

    # --- 4. Moving Target Detection Logic ---
    print(f"Using reflection mode: {settings['reflection_type']}")
    mtd = MovingTargetDetection(
        sat_const,
        gnss_constl_dict,
        ship_trajectory,
        gnss_r_antenna,
        deltaT=step_size,
        earth_model=earth_model,
        reflection_mode=settings["reflection_type"],
        link_budget_settings=settings.get("link_budget"),
        cutoff_angle=settings.get("cutoff_angle", 45),
        min_tx_elevation_deg=settings.get("min_tx_elevation_deg", 5.0),
        max_bistatic_angle_deg=settings.get("max_bistatic_angle_deg", 0.0),
    )

    # In backscatter mode the BRP is steered toward Tx, not the ship, so Rx does not
    # need to be overhead the ship — just above the horizon from it.
    rx_elev_min = 5 if settings["reflection_type"] == "backscatter" else 40

    # Optimize: only compute and propagate for visible windows
    ship_resampled, sat_visible_times = fn.compute_visible_times_and_resample_ship(
        mtd, sat_const, elev_min=rx_elev_min
    )

    _ = fn.propagate_only_visible_satellites(
        sat_const,
        sat_visible_times,
        gnss_constl_dict=mtd.gnss_constellation
    )

    # --- 5. Cell-centric detection ---
    suffix = f"_{output_suffix}" if output_suffix else ""
    cc = MovingTargetCellCentric(mtd, gnss_r_antenna, steer_fraction=0.2,
                                 max_bistatic_angle_deg=settings.get("max_bistatic_angle_deg_cc"))
    cc.calculate(verbose=True)

    cc.show_detections(file_name=f"Detection{suffix}.png")
    cc.show_timeline(file_name=f"Detection_timeline{suffix}.png")

    for const, dets in cc.detections.items():
        print(f"DETECTION_COUNT {const}: {len(dets)}")

    return cc


if __name__ == "__main__":
    import os
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Configuration_ship.ini"
    tag = os.path.splitext(os.path.basename(config_file))[0].replace("Configuration_", "")
    _, simulation_settings = initialize(config_file)
    run_moving_target_detection(simulation_settings, output_suffix=tag)
