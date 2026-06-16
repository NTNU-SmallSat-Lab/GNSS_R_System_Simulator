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
        selected_signals=settings["selected_signals"],
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

    # 5. Output & Visualization ---
    mtd.calculate_moving_target_reflection_intersection()

    # 6. Per-(constellation, band) footprint-overlap map + Pr breakdown (one set of files per signal).
    # NOTE: the maps show where the antenna footprint overlaps the target -- not a detectability gate
    # (no SCR/SNR threshold is applied), hence "footprint overlap" rather than "detection".
    footprint_db_label = _format_footprint_db_label(settings.get("footprint_dB"))
    for gnss_name, sigs in mtd.signals_per_constellation.items():
        for sig in sigs:
            band = sig["band"]
            # Footprint-overlap map, scoped to this constellation and band.
            output_plot = f"Footprint_overlap_{mtd.reflection_mode}_{gnss_name}_{band}_{footprint_db_label}.png"
            mtd.show_track_and_detection_points(
                gnss_constellation=gnss_name, file_name=output_plot, show_figure=False, band=band
            )
            print(f"Footprint-overlap plot saved as: {output_plot}")

            # Link-budget breakdown: ocean clutter, ship signal, noise floor, and SINR qualification.
            contributions_plot = f"Link_budget_breakdown_{mtd.reflection_mode}_{gnss_name}_{band}.png"
            mtd.show_link_budget_contributions(file_name=contributions_plot, show_figure=False, band=band)
            print(f"Link-budget breakdown plot saved as: {contributions_plot}")

    # 6b. Combined plots: all constellations and bands together, color-coded by constellation.
    constellations_with_signals = [name for name, sigs in mtd.signals_per_constellation.items() if sigs]
    combined_plot = f"Footprint_overlap_{mtd.reflection_mode}_combined_{footprint_db_label}.png"
    mtd.show_track_and_detection_points(
        gnss_constellation=constellations_with_signals, file_name=combined_plot, show_figure=False, band=None
    )
    print(f"Combined footprint-overlap plot saved as: {combined_plot}")

    combined_contributions_plot = f"Link_budget_breakdown_{mtd.reflection_mode}_combined.png"
    mtd.show_link_budget_contributions(file_name=combined_contributions_plot, show_figure=False, band=None)
    print(f"Combined link-budget breakdown plot saved as: {combined_contributions_plot}")

    # 7. Reflection animation (band-independent: reflection geometry is the same across bands).
    if settings["create_animation"]:
        mtd.create_reflection_animation(**settings["animation_settings"])

if __name__ == "__main__":
    # Initialize returns both config (raw) and settings (processed)
    _, simulation_settings = initialize("Configuration_ship.ini")
    run_moving_target_detection(simulation_settings)
