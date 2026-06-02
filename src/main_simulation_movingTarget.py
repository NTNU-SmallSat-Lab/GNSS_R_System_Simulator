import logging

import matplotlib

from __init__ import initialize
import functions.functions as fn
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory

# Force non-interactive backend for server/headless environments
matplotlib.use("Agg", force=True)

# Application-level logging setup. Library modules use logging.getLogger(__name__) and do not
# configure handlers themselves; this is the single place where the user-visible default is set.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _format_footprint_db_label(footprint_db: float | None) -> str:
    if footprint_db is None:
        return "unknown_dB"

    footprint_db = float(footprint_db)
    if footprint_db.is_integer():
        value = str(int(footprint_db))
    else:
        value = f"{footprint_db:g}"
    return f"{value}dB"


def run_moving_target_detection(settings: dict):
    """
    =========================================================================================================
    Calculate the probability of detecting a ship traveling from point A to B using GNSS-R LEO satellites.

    INPUTS:
        settings: [dict]   Simulation settings dictionary loaded from Configuration_ship.ini
    =========================================================================================================
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
        bbox_buffer_deg=settings["detection_bbox_buffer_deg"],
        max_central_angle_deg=settings["max_central_angle_deg"],
        max_incidence_angle_deg=settings["max_incidence_angle_deg"],
        ship_dimensions=settings["ship"],
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
    footprint_db_label = _format_footprint_db_label(settings.get("footprint_dB"))
    output_plot = f"Detection_points_{mtd.reflection_mode}_footprint_{footprint_db_label}.png"
    mtd.show_track_and_detection_points(file_name=output_plot, show_figure=False)
    print(f"Detection plot saved as: {output_plot}")

    # Link-budget breakdown: ocean clutter vs ship signal per detection event.
    contributions_plot = f"Detection_Pr_breakdown_{mtd.reflection_mode}.png"
    mtd.show_link_budget_contributions(file_name=contributions_plot, show_figure=False)
    print(f"Pr breakdown plot saved as: {contributions_plot}")

    if settings["create_animation"]:
        mtd.create_reflection_animation(**settings["animation_settings"])

if __name__ == "__main__":
    # Initialize returns both config (raw) and settings (processed)
    _, simulation_settings = initialize("Configuration_ship.ini")
    run_moving_target_detection(simulation_settings)
