"""Monte Carlo feasibility study for GNSS-R ship detection (backscatter or specular).

Holds the Chapter-11 platform fixed (read from a dedicated Configuration_mc_*.ini) and samples the
variables listed in that file's [MONTE CARLO] section: ship heading/position, epoch, sea state, and
the receiver knobs (coherent integration time, incoherent looks, system noise temperature). Each
trial runs the tested MovingTargetDetection pipeline and records, per trial, the best achievable
detectability metric and whether the target would be detected. Aggregates into a detection
probability and feasibility plots that answer "whether" and "when" each method is feasible.

Run from the repo root:
    python SRC/run_monte_carlo.py Configuration_mc_backscatter.ini
    python SRC/run_monte_carlo.py Configuration_mc_specular.ini [n_trials_override]
    python SRC/run_monte_carlo.py Configuration_mc_backscatter.ini aggregate [--output-dir OUTPUT/new_run]

New trials also retain paired powers for every footprint-overlap event. Event plots combine
selected bands and weight trials by their number of valid events, independently of detection.
"""
import argparse
import ast
import configparser
from datetime import timedelta
import json
import math
import multiprocessing as mp
from multiprocessing import cpu_count
import os
import queue
import time

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from __init__ import initialize
import functions.functions as fn
from functions.mc_event_plots import make_event_plots, serialize_power_events
from object_definitions.Constants_def import Constants
from object_definitions.MovingTargetDetection_def import MovingTargetDetection
from object_definitions.Rx_antenna_def import RxAntenna as GNSS_R_antenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite
from object_definitions.ShipTrajectory_def import ShipTrajectory

EARTH_RADIUS_KM = 6371.0
_SHARED: dict = {}  # per-worker cache (antenna is fixed across trials)

# A rare drawn position/epoch can hit a pathological geometry (e.g. a ship position in a narrow
# fjord that stalls the route-planning search) and never return. Normal trials finish in well
# under a minute, so this is a generous upper bound; a trial that exceeds it is terminated and
# logged as a timeout rather than blocking the rest of the study indefinitely.
TRIAL_TIMEOUT_S = 600


# --------------------------------------------------------------------------------------------------
# [MONTE CARLO] parsing and per-trial draws
# --------------------------------------------------------------------------------------------------
def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("true", "1", "yes", "on")


def _as_list(text: str) -> list:
    return list(ast.literal_eval(text))


def parse_mc_section(config) -> dict:
    """Parse the [MONTE CARLO] section (INI-strict: required keys, missing keys raise)."""
    if "MONTE CARLO" not in config:
        raise KeyError("Configuration is missing the required [MONTE CARLO] section.")
    s = config["MONTE CARLO"]
    spec = {
        "n_trials": int(s["n_trials"]),
        "seed": int(s["seed"]),
        "window_hours": float(s["window_hours"]),
        "vary_heading": _as_bool(s["vary_heading"]),
        "vary_position": _as_bool(s["vary_position"]),
        "roi_bbox": _as_list(s["roi_bbox"]),                       # [min_lon, max_lon, min_lat, max_lat]
        "vary_epoch": _as_bool(s["vary_epoch"]),
        "epoch_window_hours": float(s["epoch_window_hours"]),
        "vary_sea_state": _as_bool(s["vary_sea_state"]),
        "wind_speed_range_m_s": _as_list(s["wind_speed_range_m_s"]),
        "wave_height_range_m": _as_list(s["wave_height_range_m"]),
        "wind_direction_range_deg": _as_list(s["wind_direction_range_deg"]),
        "inverse_wave_age_range": _as_list(s["inverse_wave_age_range"]),
        "vary_coherent_integration": _as_bool(s["vary_coherent_integration"]),
        "coherent_integration_time_s_values": _as_list(s["coherent_integration_time_s_values"]),
        "vary_incoherent_looks": _as_bool(s["vary_incoherent_looks"]),
        "num_incoherent_looks_values": _as_list(s["num_incoherent_looks_values"]),
        "vary_system_noise_temperature": _as_bool(s["vary_system_noise_temperature"]),
        "system_noise_temperature_k_range": _as_list(s["system_noise_temperature_k_range"]),
    }
    return spec


def destination_point(lat_deg, lon_deg, bearing_deg, distance_km):
    """Great-circle destination from (lat, lon) along bearing for distance_km."""
    ang = distance_km / EARTH_RADIUS_KM
    brg = math.radians(bearing_deg)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(ang) * math.cos(lat1),
                             math.cos(ang) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def draw_trials(spec, base_settings, ocean_polygons):
    """Pre-draw all trials (reproducible from the seed). Returns a list of override dicts."""
    from shapely.geometry import Point

    rng = np.random.default_rng(spec["seed"])
    lb = base_settings["link_budget"]
    base_lat, base_lon = base_settings["start_coord_ll"]
    min_lon, max_lon, min_lat, max_lat = spec["roi_bbox"]
    draws = []
    for _ in range(spec["n_trials"]):
        # Position (on water).
        if spec["vary_position"]:
            lat, lon = base_lat, base_lon
            for _attempt in range(200):
                clat = rng.uniform(min_lat, max_lat)
                clon = rng.uniform(min_lon, max_lon)
                if ocean_polygons is None or ocean_polygons.contains(Point(clon, clat)):
                    lat, lon = clat, clon
                    break
        else:
            lat, lon = base_lat, base_lon
        heading = rng.uniform(0.0, 360.0) if spec["vary_heading"] else 90.0
        epoch_off_h = rng.uniform(-spec["epoch_window_hours"], spec["epoch_window_hours"]) if spec["vary_epoch"] else 0.0
        if spec["vary_sea_state"]:
            wind_speed = float(rng.uniform(*spec["wind_speed_range_m_s"]))
            wave_height = float(rng.uniform(*spec["wave_height_range_m"]))
            wind_dir = float(rng.uniform(*spec["wind_direction_range_deg"]))
            inv_wave_age = float(rng.uniform(*spec["inverse_wave_age_range"]))
        else:
            wind_speed, wave_height = lb["ocean_wind_speed_m_s"], lb["ocean_wave_height_m"]
            wind_dir, inv_wave_age = lb["ocean_wind_direction_deg"], lb["ocean_inverse_wave_age"]
        t_coh = float(rng.choice(spec["coherent_integration_time_s_values"])) if spec["vary_coherent_integration"] else lb["coherent_integration_time_s"]
        looks = int(rng.choice(spec["num_incoherent_looks_values"])) if spec["vary_incoherent_looks"] else lb["num_incoherent_looks"]
        t_sys = float(rng.uniform(*spec["system_noise_temperature_k_range"])) if spec["vary_system_noise_temperature"] else lb["system_noise_temperature_k"]
        draws.append({
            "lat": float(lat), "lon": float(lon), "heading_deg": float(heading), "epoch_off_h": float(epoch_off_h),
            "wind_speed_m_s": wind_speed, "wave_height_m": wave_height, "wind_direction_deg": wind_dir,
            "inverse_wave_age": inv_wave_age, "coherent_integration_time_s": t_coh,
            "num_incoherent_looks": looks, "system_noise_temperature_k": t_sys,
        })
    return draws


# --------------------------------------------------------------------------------------------------
# Per-trial execution (worker process)
# --------------------------------------------------------------------------------------------------
def _init_worker(base_settings):
    matplotlib.use("Agg", force=True)
    _SHARED["base"] = base_settings
    _SHARED["antenna"] = GNSS_R_antenna(base_settings)   # fixed across trials; load once per worker


def _blank_result(idx, draw, error=None):
    """Zero-opportunity result shape, shared by a caught exception inside a trial and by a
    trial that never returned at all (timeout / worker crash)."""
    return {"idx": idx, **draw, "n_opportunities": 0, "best_metric_db": float("nan"),
            "best_secondary_db": float("nan"), "detected": False, "error": error, "power_events": []}


def _trial_worker_entry(settings, task, out_queue):
    """Process entry point for the timeout-safe runner: init the worker's antenna cache, run one
    trial, and push the result. Runs in its own process so a hang can be killed from outside."""
    _init_worker(settings)
    idx, draw = task
    try:
        res = _run_trial(task)
    except Exception as exc:  # _run_trial already catches internally; defensive backstop only
        res = _blank_result(idx, draw, f"WorkerCrash: {type(exc).__name__}: {exc}")
    out_queue.put(res)


def run_trials_timeout_safe(settings, tasks, n_workers, timeout_s):
    """Run trials with bounded concurrency and a hard per-trial wall-clock timeout.

    Each task runs in its own process. A trial that exceeds timeout_s is terminated (its worker
    is killed and the slot freed for the next pending task) and yielded as a recorded timeout
    error, so one pathological trial can never block the rest of the study the way an unbounded
    Pool.imap would -- and unlike giving up on a Pool AsyncResult, the stuck worker's CPU/process
    is actually reclaimed.
    """
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    running: dict = {}  # idx -> {"process", "queue", "task", "start"}

    def _launch(task):
        idx, _draw = task
        q = ctx.Queue()
        p = ctx.Process(target=_trial_worker_entry, args=(settings, task, q), daemon=True)
        p.start()
        running[idx] = {"process": p, "queue": q, "task": task, "start": time.monotonic()}

    while pending or running:
        while len(running) < n_workers and pending:
            _launch(pending.pop(0))

        finished = []
        for idx, info in list(running.items()):
            p = info["process"]
            # Drain results while the worker is alive: event lists can exceed the pipe buffer,
            # and multiprocessing.Queue's feeder must finish flushing before the worker can exit.
            try:
                res = info["queue"].get_nowait()
            except queue.Empty:
                if not p.is_alive():
                    # Retry after observing exit, in case the final queue flush raced the first read.
                    try:
                        res = info["queue"].get_nowait()
                    except queue.Empty:
                        res = _blank_result(idx, info["task"][1], f"WorkerExit: process ended without a result (exitcode {p.exitcode})")
                    p.join()
                    finished.append((idx, res))
                elif time.monotonic() - info["start"] > timeout_s:
                    p.terminate()
                    p.join(5)
                    res = _blank_result(idx, info["task"][1], f"TimeoutError: exceeded {timeout_s:.0f}s")
                    finished.append((idx, res))
            else:
                p.join()
                finished.append((idx, res))

        if not finished:
            time.sleep(0.5)
            continue
        for idx, res in finished:
            running[idx]["queue"].close()
            del running[idx]
            yield res


def _run_trial(task):
    idx, draw = task
    base = _SHARED["base"]
    mode = base["reflection_type"]
    result = _blank_result(idx, draw)
    try:
        settings = dict(base)
        settings["start_time"] = base["start_time"] + timedelta(hours=draw["epoch_off_h"])
        settings["start_coord_ll"] = (draw["lat"], draw["lon"])
        speed_kmh = settings["moving_target_speed"] * 1.852           # knots -> km/h
        dist_km = max(speed_kmh * settings["window_hours_mc"], 0.2)   # travel so the window ~= window_hours
        settings["end_coord_ll"] = destination_point(draw["lat"], draw["lon"], draw["heading_deg"], dist_km)
        link_budget = dict(base["link_budget"])
        link_budget.update({
            "ocean_wind_speed_m_s": draw["wind_speed_m_s"], "ocean_wave_height_m": draw["wave_height_m"],
            "ocean_wind_direction_deg": draw["wind_direction_deg"], "ocean_inverse_wave_age": draw["inverse_wave_age"],
            "coherent_integration_time_s": draw["coherent_integration_time_s"],
            "num_incoherent_looks": draw["num_incoherent_looks"],
            "system_noise_temperature_k": draw["system_noise_temperature_k"],
        })
        settings["link_budget"] = link_budget

        ship = ShipTrajectory(
            ship_id=f"MC_{mode}_{idx}", ship_speed_knot=settings["moving_target_speed"],
            start_position=settings["start_coord_ll"], start_time=settings["start_time"],
            end_position=settings["end_coord_ll"], interval_minutes=10, grid_resolution_km=2,
            fast_path_gen_method=False,
        )
        gnss = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
        sats = [Satellite(i, orbit_n=settings["orbits"][i]) for i in range(settings["No_satellites"])]
        const = SatelliteConstellation(satConfigName="LEO MC", satArray=sats)
        dur_h = ship.get_travel_duration_hours()
        step = settings["step_size"]
        const.set_simulation_duration(settings["start_time"], dur_h, step)
        for gc in gnss.values():
            gc.set_simulation_duration(settings["start_time"], dur_h, step)

        mtd = MovingTargetDetection(
            const, gnss, ship, _SHARED["antenna"], deltaT=step,
            earth_model=settings["earth_model"], reflection_mode=mode,
            link_budget_settings=settings["link_budget"], bbox_buffer_deg=settings["detection_bbox_buffer_deg"],
            max_central_angle_deg=settings["max_central_angle_deg"],
            max_incidence_angle_deg=settings["max_incidence_angle_deg"],
            ship_dimensions=settings["ship"], selected_signals=settings["selected_signals"],
        )
        _, vis = fn.compute_visible_times_and_resample_ship(mtd, const, elev_min=0.0)
        if not any(vis.values()):
            return result                                 # no satellite pass over the ship this window -> 0 opportunities
        fn.propagate_only_visible_satellites(const, vis, gnss_constl_dict=mtd.gnss_constellation)
        mtd.calculate_moving_target_reflection_intersection()

        events = [e for evs in mtd.target_detect_dict.values() for e in evs]
        result["n_opportunities"] = len(events)
        result["power_events"] = serialize_power_events(mtd.target_detect_dict, mode)
        threshold = settings["link_budget"].get("detection_sinr_threshold_db", 0.0) or 0.0
        if mode == "backscatter":
            primary = [e.sinr_ship_db for e in events if e.sinr_ship_db is not None and np.isfinite(e.sinr_ship_db)]
            secondary = [e.scr_db for e in events if e.scr_db is not None and np.isfinite(e.scr_db)]
        else:
            primary = [e.forward_delta_pr_over_noise_db for e in events
                       if e.forward_delta_pr_over_noise_db is not None and np.isfinite(e.forward_delta_pr_over_noise_db)]
            secondary = [e.forward_delta_pr_fraction_db for e in events
                         if e.forward_delta_pr_fraction_db is not None and np.isfinite(e.forward_delta_pr_fraction_db)]
        if primary:
            result["best_metric_db"] = float(max(primary))
            result["detected"] = bool(result["best_metric_db"] >= threshold)
        if secondary:
            result["best_secondary_db"] = float(max(secondary))
    except Exception as exc:  # a failed trial counts as zero-opportunity, not a crash of the whole study
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# --------------------------------------------------------------------------------------------------
# Aggregation and plotting
# --------------------------------------------------------------------------------------------------
def summarize(results, mode, threshold_db):
    detected = np.array([r["detected"] for r in results], dtype=bool)
    opp = np.array([r["n_opportunities"] for r in results], dtype=int)
    n = len(results)
    with_opp = opp > 0
    pd_all = float(detected.mean()) if n else float("nan")
    pd_given_opp = float(detected[with_opp].mean()) if with_opp.any() else float("nan")
    summary = {
        "mode": mode, "n_trials": n, "detection_threshold_db": threshold_db,
        "pd_overall": pd_all,
        "p_detect_given_opportunity": pd_given_opp,
        "opportunity_rate": float(with_opp.mean()) if n else float("nan"),
        "mean_opportunities_per_trial": float(opp.mean()) if n else float("nan"),
        "n_errors": int(sum(1 for r in results if r["error"])),
    }
    return summary


def make_plots(results, summary, mode, out_dir):
    """Write each MC panel as its own standalone figure (no subplots). Returns the saved paths."""
    metric = np.array([r["best_metric_db"] for r in results], dtype=float)
    finite = np.isfinite(metric)
    metric_label = r"best ship $\mathrm{SINR}$ [dB]" if mode == "backscatter" else r"best $\Delta P_\mathrm{r}\,/\,N$ [dB]"
    stem = os.path.join(out_dir, f"mc_{mode}")
    suptxt = (rf"Monte Carlo feasibility -- {mode}  "
              rf"($P_\mathrm{{d}}$={summary['pd_overall']:.2%}, P(det|opp)={summary['p_detect_given_opportunity']:.2%}, "
              rf"opp rate={summary['opportunity_rate']:.2%}, n={summary['n_trials']})")
    saved = []

    def _finish(fig, suffix):
        fig.suptitle(suptxt, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = f"{stem}_{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # (1) Distribution of the best detectability metric, with the detection threshold.
    fig, ax = plt.subplots(figsize=(8, 6))
    if finite.any():
        ax.hist(metric[finite], bins=30, color="#4477aa", edgecolor="black", alpha=0.8)
    ax.axvline(summary["detection_threshold_db"], ls="--", c="k", label=f"threshold {summary['detection_threshold_db']:.0f} dB")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("trials")
    ax.set_title("Detectability distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, "distribution")

    # (2) Pd vs coherent integration time (receiver axis).
    fig, ax = plt.subplots(figsize=(8, 6))
    _pd_vs_discrete(ax, results, "coherent_integration_time_s", r"$T_\mathrm{coh}$ [s]")
    _finish(fig, "pd_vs_tcoh")

    # (3) Pd vs incoherent looks (receiver axis).
    fig, ax = plt.subplots(figsize=(8, 6))
    _pd_vs_discrete(ax, results, "num_incoherent_looks", r"incoherent looks $M$")
    _finish(fig, "pd_vs_looks")

    # (4) best metric vs system noise temperature (receiver axis) and wind speed (environment), colored.
    fig, ax = plt.subplots(figsize=(8, 6))
    tsys = np.array([r["system_noise_temperature_k"] for r in results], dtype=float)
    wind = np.array([r["wind_speed_m_s"] for r in results], dtype=float)
    if finite.any():
        sc = ax.scatter(tsys[finite], metric[finite], c=wind[finite], cmap="viridis", s=18, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="wind speed [m/s]")
    ax.axhline(summary["detection_threshold_db"], ls="--", c="k")
    ax.set_xlabel(r"system noise temperature $T_\mathrm{sys}$ [K]")
    ax.set_ylabel(metric_label)
    ax.set_title(r"Metric vs $T_\mathrm{sys}$ (color = wind)")
    ax.grid(alpha=0.3)
    _finish(fig, "metric_vs_tsys")

    return saved


def _pd_vs_discrete(ax, results, key, label):
    vals = sorted({r[key] for r in results})
    pds, counts = [], []
    for v in vals:
        sub = [r for r in results if r[key] == v]
        pds.append(np.mean([r["detected"] for r in sub]) if sub else 0.0)
        counts.append(len(sub))
    x = np.arange(len(vals))
    ax.bar(x, pds, color="#ee6677", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}\n(n={c})" for v, c in zip(vals, counts, strict=True)], fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel(r"detection probability $P_\mathrm{d}$")
    ax.set_ylim(0, 1)
    ax.set_title(rf"$P_\mathrm{{d}}$ vs {label}")
    ax.grid(alpha=0.3, axis="y")


# --------------------------------------------------------------------------------------------------
def _read_jsonl(path):
    results = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def aggregate_and_report(mode, threshold_db, out_jsonl, out_json, out_dir):
    """Build the summary + feasibility plots from whatever trials are in the JSONL (partial or full)."""
    results = _read_jsonl(out_jsonl)
    results.sort(key=lambda r: r["idx"])
    if not results:
        print(f"No trials found in {out_jsonl}; nothing to aggregate.")
        return
    summary = summarize(results, mode, threshold_db)
    with open(out_json, "w") as fh:
        json.dump({"summary": summary, "trials": results}, fh, indent=1, default=float)
    saved = make_plots(results, summary, mode, out_dir)
    saved.extend(make_event_plots(results, mode, out_dir))
    print(json.dumps(summary, indent=2))
    print(f"saved {out_json}")
    for path in saved:
        print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Configuration_mc_*.ini")
    parser.add_argument("action", nargs="?", help="trial count override, or aggregate")
    parser.add_argument("--output-dir", default=Constants.output_folder_path, help="MC result directory (default: OUTPUT)")
    parser.add_argument("--trial-timeout-s", type=float, default=TRIAL_TIMEOUT_S, help=f"hard wall-clock limit per trial (default: {TRIAL_TIMEOUT_S} s)")
    args = parser.parse_args()
    out_dir = args.output_dir

    # Aggregation reads only the reporting settings. In particular, it must not initialize
    # antenna patterns, draw trials, propagate satellites, or start simulation workers.
    if args.action == "aggregate":
        config = configparser.ConfigParser()
        with open(args.config) as fh:
            config.read_file(fh)
        mode = config.get("REFLECTION", "reflection_type", fallback="specular").strip().strip("\"'")
        if mode not in {"backscatter", "specular"}:
            raise ValueError("[REFLECTION].reflection_type must be either 'specular' or 'backscatter'.")
        threshold_db = config.getfloat("LINK BUDGET", "detection_sinr_threshold_db", fallback=0.0)
        if not math.isfinite(threshold_db):
            raise ValueError("[LINK BUDGET].detection_sinr_threshold_db must be finite.")
        aggregate_and_report(
            mode, threshold_db,
            os.path.join(out_dir, f"mc_{mode}_trials.jsonl"),
            os.path.join(out_dir, f"mc_{mode}_results.json"), out_dir,
        )
        return

    config, settings = initialize(args.config)
    spec = parse_mc_section(config)
    settings["window_hours_mc"] = spec["window_hours"]
    mode = settings["reflection_type"]
    threshold_db = settings["link_budget"].get("detection_sinr_threshold_db", 0.0) or 0.0
    out_jsonl = os.path.join(out_dir, f"mc_{mode}_trials.jsonl")
    out_json = os.path.join(out_dir, f"mc_{mode}_results.json")
    if args.action is not None:
        spec["n_trials"] = int(args.action)
    os.makedirs(out_dir, exist_ok=True)

    # Ocean polygons (once, main process) for on-water position draws.
    try:
        import geopandas as gpd
        from shapely import ops
        ocean_polygons = ops.unary_union(gpd.read_file(Constants.earth_coastline_path).geometry)
    except Exception as exc:
        print(f"WARNING: could not load ocean polygons ({exc}); position draws will not be water-checked.")
        ocean_polygons = None

    draws = draw_trials(spec, settings, ocean_polygons)
    # Resume: skip trials already present in the JSONL (so a killed run continues where it left off).
    done_idx = {r["idx"] for r in _read_jsonl(out_jsonl)}
    tasks = [(i, d) for i, d in enumerate(draws) if i not in done_idx]
    print(f"[MC {mode}] {spec['n_trials']} trials (resuming; {len(done_idx)} done, {len(tasks)} pending), "
          f"window {spec['window_hours']} h, seed {spec['seed']}")

    # Incremental JSONL so a kill never loses completed trials. Each trial runs in its own process
    # under a hard timeout (TRIAL_TIMEOUT_S), so one hung trial cannot block the rest of the study.
    n_workers = max(1, min(cpu_count() - 1, len(tasks))) if tasks else 1
    completed = len(done_idx)
    if tasks:
        with open(out_jsonl, "a") as fh:
            for res in run_trials_timeout_safe(settings, tasks, n_workers, args.trial_timeout_s):
                fh.write(json.dumps(res, default=float) + "\n")
                fh.flush()
                completed += 1
                if res.get("error"):
                    print(f"  trial {res['idx']}: {res['error']}", flush=True)
                if completed % 20 == 0:
                    print(f"  {completed}/{spec['n_trials']} trials done", flush=True)

    aggregate_and_report(mode, threshold_db, out_jsonl, out_json, out_dir)


if __name__ == "__main__":
    main()
