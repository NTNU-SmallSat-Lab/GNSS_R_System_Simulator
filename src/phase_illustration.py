"""Chapter-11 illustration, part (i): phase a LEO receiver so the backscatter reflection
point (BRP) lands ON the tanker at 14:00 UTC 1 Oct 2025 with a favorable (oblique) incidence,
then run the tested pipeline for the co-located ocean-vs-ship link budget.

Run from repo root:  python SRC/phase_illustration.py
"""
import json
import logging
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np
import pymap3d
from shapely.geometry import Point
from skyfield.api import load as skyload

from __init__ import initialize
import functions.functions as fn
from object_definitions.Constants_def import Constants
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

MU = Constants.GM_EARTH_TT
A_LEO = 7100000.0          # LEO Rx semi-major axis [m] (~722 km alt)
SHIP_LL = (58.30, 10.10)   # tanker (lat, lon)
T0 = datetime(2025, 10, 1, 14, 0, 0, tzinfo=timezone.utc)  # 16:00 local


def circular_elements_at(r_eci):
    """Classical elements [i,e,RAAN,argp,nu,a] (deg,-,deg,deg,deg,m) for a circular orbit through r_eci,
    with a near-polar (max-inclination) velocity direction."""
    r = np.asarray(r_eci, float)
    R = np.linalg.norm(r)
    rhat = r / R
    zhat = np.array([0.0, 0.0, 1.0])
    vdir = zhat - np.dot(zhat, rhat) * rhat        # component of +z perpendicular to r -> northward
    vdir /= np.linalg.norm(vdir)
    v = np.sqrt(MU / R) * vdir                       # circular speed, perpendicular to r
    h = np.cross(r, v); H = np.linalg.norm(h)
    nvec = np.cross(zhat, h); N = np.linalg.norm(nvec)
    inc = np.degrees(np.arccos(np.clip(h[2] / H, -1, 1)))
    raan = np.degrees(np.arctan2(nvec[1], nvec[0])) % 360.0
    nhat = nvec / N; hhat = h / H
    ahead = np.cross(hhat, nhat)                     # 90 deg ahead of node, in-plane
    u = np.degrees(np.arctan2(np.dot(r, ahead), np.dot(r, nhat))) % 360.0
    return [float(inc), 1e-5, float(raan), 0.0, float(u), R]


def line_sphere_far_point(p_from, p_to, radius):
    """Point at |P|=radius on the line P(s)=p_from + s*(p_to-p_from), taking the intersection BETWEEN
    p_from (Tx) and p_to (ship) -- i.e. the receiver above the ship, so the ship is beyond it (=BRP).
    That is the smaller root, which lies in (0,1). Returns the ECI point or None."""
    d = p_to - p_from
    a = np.dot(d, d); b = 2 * np.dot(p_from, d); c = np.dot(p_from, p_from) - radius ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = (-b - np.sqrt(disc)) / (2 * a)               # smaller root: between Tx and ship
    return p_from + s * d


def main():
    _, settings = initialize("Configuration_illustration.ini")
    ts = skyload.timescale()
    t_sf = ts.utc(T0.year, T0.month, T0.day, T0.hour, T0.minute, T0.second)

    # Ship ECI at T0 (same pymap3d path the detector uses).
    sx, sy, sz = pymap3d.geodetic2ecef(SHIP_LL[0], SHIP_LL[1], 0.0)
    sxe, sye, sze = pymap3d.ecef2eci(sx, sy, sz, T0)
    ship_eci = np.array([float(sxe), float(sye), float(sze)])
    up = ship_eci / np.linalg.norm(ship_eci)

    # GPS positions at T0; pick a Tx near 45 deg elevation (=> ~45 deg incidence at the ship).
    # Target ~63 deg elevation -> incidence ~27 deg: above the 20 deg Bragg gate, and the receiver
    # look angle to the BRP stays within the +/-40 deg beam-steering limit.
    gnss = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
    gps = gnss["GPS"]
    best = None
    for sat in gps.satellites:
        tx = np.asarray(sat.skf.at(t_sf).position.m, float)
        los = tx - ship_eci
        elev = np.degrees(np.arcsin(np.dot(los, up) / np.linalg.norm(los)))
        if elev < 20:
            continue
        score = abs(elev - 63.0)
        if best is None or score < best[0]:
            best = (score, sat, tx, elev)
    _, tx_sat, tx_eci, elev = best
    print(f"Chosen GPS Tx satID={tx_sat.satID} elevation={elev:.1f} deg (=> incidence ~{90-elev:.1f} deg at ship)")

    # Receiver on the Tx->ship line at LEO altitude (ship is then the BRP beyond the Rx).
    r_rx = line_sphere_far_point(tx_eci, ship_eci, A_LEO)
    print(f"Ideal Rx altitude = {(np.linalg.norm(r_rx)-Constants.R_EARTH_MEAN)/1000:.1f} km")
    rx_nadir = -r_rx / np.linalg.norm(r_rx)
    rx2ship = (ship_eci - r_rx) / np.linalg.norm(ship_eci - r_rx)
    look = np.degrees(np.arccos(np.clip(np.dot(rx_nadir, rx2ship), -1, 1)))
    print(f"Rx look angle off nadir = {look:.1f} deg (must be < ~40 deg to steer onto the ship)")
    central = np.degrees(np.arccos(np.clip(np.dot(tx_eci, r_rx) / (np.linalg.norm(tx_eci) * np.linalg.norm(r_rx)), -1, 1)))
    print(f"Tx-Rx central (Earth-centre) angle = {central:.1f} deg")
    elements = circular_elements_at(r_rx)
    print(f"LEO elements [i,e,RAAN,argp,nu,a] = {[round(x,4) for x in elements]}")

    # Build the single-LEO Rx and the (near-static) ship over a short window starting at T0.
    leo = Satellite(0, orbit_n=elements)
    leo.set_epoch(T0)   # osc elements are defined at T0 -> generates the TLE / skf at that epoch
    ship = ShipTrajectory(ship_id="DarkShip_001", ship_speed_knot=0.1,
                          start_position=SHIP_LL, start_time=T0, end_position=(58.3003, 10.10),
                          interval_minutes=1, grid_resolution_km=2, fast_path_gen_method=False)
    ant = GNSS_R_antenna(settings)
    const = SatelliteConstellation(satConfigName="LEO", satArray=[leo])
    dur_h = ship.get_travel_duration_hours()
    step = settings["step_size"]
    const.set_simulation_duration(T0, dur_h, step)   # sets the orbit epoch -> builds leo.skf
    for gc in gnss.values():
        gc.set_simulation_duration(T0, dur_h, step)

    # Verify the osc->TLE->SGP4 round-trip reproduces the ideal Rx position at T0.
    pos_check = np.asarray(leo.skf.at(t_sf).position.m, float)
    print(f"Rx placement error at T0 = {np.linalg.norm(pos_check - r_rx)/1000:.2f} km")

    mtd = MovingTargetDetection(const, gnss, ship, ant, deltaT=step,
                                earth_model=settings["earth_model"], reflection_mode="backscatter",
                                link_budget_settings=settings.get("link_budget"),
                                bbox_buffer_deg=settings["detection_bbox_buffer_deg"],
                                max_central_angle_deg=179.0,
                                max_incidence_angle_deg=89.0,
                                ship_dimensions=settings["ship"], selected_signals=settings["selected_signals"])
    # Full propagation over the short window (the visibility optimization assumes the Rx passes over
    # the target, which is false in backscatter -- the Rx is ~700 km from the ship, the BRP is on it).
    for sat in const.satellites:
        sat.propagate_orbit_for_times(const.times_skyfield)
    for gc in gnss.values():
        for sat in gc.satellites:
            sat.propagate_orbit_for_times(gc.times_skyfield)
    # Ship ~static over the short window: fixed position at every satellite time (bypass visibility filter).
    sat_times_dt = list(const.times_skyfield.utc_datetime())
    mtd.timestamps_resampled = sat_times_dt
    mtd.ship_positions_resampled = [Point(SHIP_LL[1], SHIP_LL[0]) for _ in sat_times_dt]
    mtd.calculate_moving_target_reflection_intersection()

    print("footprint_stats:", mtd.footprint_stats)
    events = [e for evs in mtd.target_detect_dict.values() for e in evs]
    print(f"\nbackscatter events: {len(events)}")

    def dist_km(rp):
        r = 6371.0
        p1, p2 = np.radians(SHIP_LL[0]), np.radians(rp[0])
        dphi = np.radians(rp[0] - SHIP_LL[0]); dl = np.radians(rp[1] - SHIP_LL[1])
        a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return 2*r*np.arcsin(np.sqrt(a))

    events.sort(key=lambda e: dist_km(e.reflection_point_ll))
    recs = []
    for e in events:
        G = float(getattr(e, "processing_gain_linear", 1.0) or 1.0)
        po = e.pr_ocean_w; ps = getattr(e, "pr_ship_w", None); N = e.noise_power_w
        sinr = 10 * np.log10(ps * G / (po * G + N)) if (ps and po and N and ps > 0 and po > 0 and N > 0) else float("nan")
        recs.append({"time": e.time.isoformat(), "refl_km_from_ship": dist_km(e.reflection_point_ll),
                     "refl_ll": list(e.reflection_point_ll), "pr_ocean_w": po, "pr_ship_w": ps, "noise_w": N,
                     "G_proc": G, "rcs_ship_m2": getattr(e, "rcs_ship_m2", None),
                     "rcs_ocean_m2": getattr(e, "rcs_ocean_m2", None),
                     "scr_db": e.scr_db, "sinr_db": sinr})
    with open("illustration_events_colocated.json", "w") as fh:
        json.dump(recs, fh, indent=1, default=float)
    for e in events[:5]:
        G = float(getattr(e, "processing_gain_linear", 1.0) or 1.0)
        po = e.pr_ocean_w; ps = getattr(e, "pr_ship_w", None); N = e.noise_power_w
        sinr = 10*np.log10(ps*G/(po*G+N)) if (ps and po and N and ps>0 and po>0 and N>0) else float("nan")
        rcs_s = getattr(e, "rcs_ship_m2", None); rcs_o = getattr(e, "rcs_ocean_m2", None)
        inc = getattr(e, "incidence_angle_deg", None); bis = getattr(e, "bistatic_angle_deg", None)
        print(f"  {e.time.strftime('%H:%M:%S')}UTC BRP@{dist_km(e.reflection_point_ll):.1f}km "
              f"reflLL=({e.reflection_point_ll[0]:.3f},{e.reflection_point_ll[1]:.3f}) "
              f"inc={inc}deg bistatic={bis}deg rcs_ship={rcs_s:.2e}m2 rcs_ocean={rcs_o:.2e}m2 "
              f"Pr_ocean={po:.2e} Pr_ship={ps:.2e} N={N:.2e} SCR={e.scr_db:.1f}dB SINR={sinr:.1f}dB")


if __name__ == "__main__":
    main()
