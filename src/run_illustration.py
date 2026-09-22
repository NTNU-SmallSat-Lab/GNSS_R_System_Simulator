"""Chapter-11 illustration runner: dark-fleet tanker in the open Skagerrak.

Runs the tested MovingTargetDetection pipeline in BOTH specular (forward) and backscatter
modes off Configuration_illustration.ini, then dumps the per-event ship-vs-ocean link budget
for the events nearest 16:00 local (14:00 UTC) on 1 Oct 2025. Run from the repo root:
    python SRC/run_illustration.py
"""
import json
import logging
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np

SHIP_LL = (58.30, 10.10)  # fixed tanker location


def _gc_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

from __init__ import initialize
import functions.functions as fn
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

ANCHOR = datetime(2025, 10, 1, 14, 0, 0, tzinfo=timezone.utc)  # 16:00 local (CEST)


def build_and_run(settings: dict, mode: str):
    settings = dict(settings)
    settings["reflection_type"] = mode
    ship = ShipTrajectory(
        ship_id="DarkShip_001",
        ship_speed_knot=settings["moving_target_speed"],
        start_position=settings["start_coord_ll"],
        start_time=settings["start_time"],
        end_position=settings["end_coord_ll"],
        interval_minutes=10,
        grid_resolution_km=2,
        fast_path_gen_method=False,
    )
    gnss = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
    ant = GNSS_R_antenna(settings)
    sats = [Satellite(i, orbit_n=settings["orbits"][i]) for i in range(settings["No_satellites"])]
    const = SatelliteConstellation(satConfigName="LEO GNSS-R", satArray=sats)

    dur_h = ship.get_travel_duration_hours()
    step = settings["step_size"]
    t0 = settings["start_time"]
    print(f"[{mode}] window = {dur_h:.2f} h from {t0.isoformat()}, step {step}s")
    const.set_simulation_duration(t0, dur_h, step)
    for gc in gnss.values():
        gc.set_simulation_duration(t0, dur_h, step)

    mtd = MovingTargetDetection(
        const, gnss, ship, ant, deltaT=step,
        earth_model=settings["earth_model"], reflection_mode=mode,
        link_budget_settings=settings.get("link_budget"),
        bbox_buffer_deg=settings["detection_bbox_buffer_deg"],
        max_central_angle_deg=settings["max_central_angle_deg"],
        max_incidence_angle_deg=settings["max_incidence_angle_deg"],
        ship_dimensions=settings["ship"],
        selected_signals=settings["selected_signals"],
    )
    _, vis = fn.compute_visible_times_and_resample_ship(mtd, const, elev_min=0.0)
    fn.propagate_only_visible_satellites(const, vis, gnss_constl_dict=mtd.gnss_constellation)
    mtd.calculate_moving_target_reflection_intersection()
    return mtd


def dump(mtd, mode: str):
    print(f"\n==================== MODE: {mode} ====================")
    recs = []
    for g, evs in mtd.target_detect_dict.items():
        for e in evs:
            rp = getattr(e, "reflection_point_ll", None)
            try:
                d_km = _gc_km(SHIP_LL[0], SHIP_LL[1], rp[0], rp[1])
            except Exception:
                d_km = float("nan")
            G = float(getattr(e, "processing_gain_linear", 1.0) or 1.0)
            po = getattr(e, "pr_ocean_w", None)
            N = getattr(e, "noise_power_w", None)
            ps = getattr(e, "pr_ship_w", None)
            fs = getattr(e, "forward_shadow_pr_w", None)
            sinr = float("nan")
            if mode == "backscatter" and ps and po and N and ps > 0 and po > 0 and N > 0:
                sinr = 10 * np.log10(ps * G / (po * G + N))
            recs.append({
                "gnss": g, "time": e.time.isoformat(), "dt_min": (e.time - ANCHOR).total_seconds() / 60.0,
                "refl_km_from_ship": d_km, "refl_ll": list(rp) if rp is not None else None,
                "tx": getattr(e, "tx_satID", None), "rx": getattr(e, "rx_satID", None),
                "pr_ocean_w": po, "pr_ship_w": ps, "noise_w": N, "G_proc": G,
                "scr_db": getattr(e, "scr_db", None), "sinr_db": sinr, "shadow_pr_w": fs,
            })
    print(f"total footprint-overlap events: {len(recs)}")
    with open(f"illustration_events_{mode}.json", "w") as fh:
        json.dump(recs, fh, indent=1, default=float)
    if not recs:
        return
    recs.sort(key=lambda r: (r["refl_km_from_ship"] if np.isfinite(r["refl_km_from_ship"]) else 9e9))
    print("Events with reflection point CLOSEST to the ship:")
    for r in recs[:10]:
        t = r["time"][11:19]
        line = (f"  {t}UTC ({r['dt_min']:+.0f}min) refl@{r['refl_km_from_ship']:.1f}km "
                f"{r['gnss']} tx={r['tx']} rx={r['rx']} Pr_ocean={r['pr_ocean_w']:.2e}W N={r['noise_w']:.2e}W")
        if mode == "backscatter":
            line += f" Pr_ship={r['pr_ship_w']:.2e}W SCR={r['scr_db']:.1f}dB SINR={r['sinr_db']:.1f}dB"
        else:
            po = r["pr_ocean_w"]; fs = r["shadow_pr_w"]
            if fs and po and po > 0:
                line += f" shadowPr={fs:.2e}W (delta/ocean={10*np.log10(abs(fs)/po):.1f}dB)"
        print(line)


if __name__ == "__main__":
    _, settings = initialize("Configuration_illustration.ini")
    for mode in ("backscatter", "specular"):
        mtd = build_and_run(settings, mode)
        dump(mtd, mode)
