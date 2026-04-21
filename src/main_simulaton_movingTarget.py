import matplotlib
import numpy as np
from datetime import timedelta

import functions.functions as fn
from __init__ import initialize
from object_definitions.MovingTargetDetection import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory
from object_definitions.Static_classes_def import Plotting

# Force non-interactive backend for server/headless environments
matplotlib.use("Agg", force=True)

def skagerrak_to_barents_sea(settings: dict):
    """
    Calculates the probability of detecting a ship traveling from point A to B
    using GNSS-R LEO satellites.
    """
    
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
    )

    # Optimize: only compute and propagate for visible windows
    ship_resampled, sat_visible_times = fn.compute_visible_times_and_resample_ship(
        mtd, sat_const, elev_min=40
    )

    _ = fn.propagate_only_visible_satellites(
        sat_const,
        sat_visible_times,
        gnss_constl_dict=mtd.gnss_constellation
    )

    # --- 5. Output & Visualization ---
    mtd.calculate_moving_target_reflection_intersection()
    
    # Save detection plot
    output_plot = f"Detection_points_{mtd.reflection_mode}.png"
    mtd.show_track_and_detection_points(file_name=output_plot, show_figure=False)
    print(f"Detection plot saved as: {output_plot}")

    # Optional: Animation (Set to True via settings if needed)
    if settings.get("create_animation", False):
        mtd.create_reflection_animation(
            time_step_min=5, 
            output_format="mp4", 
            output_filename="Reflection_animation.mp4", 
            fps=2
        )

if __name__ == "__main__":
    # Initialize returns both config (raw) and settings (processed)
    _, simulation_settings = initialize("Configuration_ship.ini")
    skagerrak_to_barents_sea(simulation_settings)