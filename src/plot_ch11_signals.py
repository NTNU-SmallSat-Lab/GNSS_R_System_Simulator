"""Chapter-11 signal-level figures from the pinned-receiver ensembles (no dependence on the noise floor).

Reads the per-event powers saved by run_monte_carlo.py (power_events in mc_<mode>_trials.jsonl) and writes
single-axes figures to the input directory:
  ch11_forward_shadow_vs_glint.png       forward: ship shadow |dP_r| vs ocean glint power, coloured by wind speed
  ch11_backscatter_ship_vs_clutter.png   backscatter: ship power vs ocean clutter power, coloured by wind speed
  ch11_scr_vs_distance.png               backscatter: SCR vs distance of the BRP from the ship
  ch11_signal_ratio.png                  ship signature relative to its background, both modes
  ch11_total_power.png                   absolute received power: total and ship signature, both modes
  ch11_signal_summary.json               numbers for the text
Run from the repo root after both ensembles:  python SRC/plot_ch11_signals.py [--input-dir OUTPUT/ch11]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

C_FWD, C_BKS = "#4477aa", "#cc6677"  # forward / backscatter
# Signature powers below this (1e-40 W) are underflow of the incoherent term (the ship patch lies far outside
# the glistening zone), not faint signals; the physical events sit above about -350 dBW. They are excluded from the
# figures but counted and reported, never floored or clipped.
FLOOR_DBW = -400.0
DIST_BINS_KM = [0.0, 10.0, 25.0, 50.0, 100.0, np.inf]  # BRP-to-ship distance bins for the summary


def read_events(path, field):
    """Return (arrays, n_trials, n_undefined, n_zero): valid events of one mode's JSONL as float arrays."""
    cols = {"ocean": [], "sig": [], "wind": [], "dist": []}
    n_trials = n_undefined = n_zero = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            n_trials += 1
            trial = json.loads(line)
            wind = trial.get("wind_speed_m_s")
            for ev in trial.get("power_events") or []:
                po, ps = ev.get("pr_ocean_w"), ev.get(field)
                if po is None or ps is None or po <= 0.0 or ps <= 0.0:
                    n_undefined += 1                      # power not available for this event
                elif 10.0 * np.log10(ps) < FLOOR_DBW:
                    n_zero += 1                           # numerically zero, not a faint signal
                else:
                    dist = ev.get("ship_dist_km")
                    cols["ocean"].append(po)
                    cols["sig"].append(ps)
                    cols["wind"].append(np.nan if wind is None else wind)
                    cols["dist"].append(np.nan if dist is None else dist)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}, n_trials, n_undefined, n_zero


def dbw(w):
    return 10.0 * np.log10(w)  # W -> dBW


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def hist_step(ax, data, bins, color, ls, label):
    """Event-fraction step histogram, so modes with different event counts are comparable."""
    ax.hist(data, bins=bins, weights=np.full(len(data), 1.0 / len(data)), histtype="step",
            linewidth=1.9, color=color, linestyle=ls, label=label)


def signature_scatter(ev, limits, xlabel, ylabel, path, point_size):
    """Ship signature power vs background power per event, coloured by wind speed, with the equal-power line."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    x, y, wind = dbw(ev["ocean"]), dbw(ev["sig"]), ev["wind"]
    order = np.argsort(np.nan_to_num(wind))  # draw high wind last
    sc = ax.scatter(x[order], y[order], c=wind[order], cmap="viridis", s=point_size, alpha=0.6, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label=r"Wind speed $U_{10}$ [m/s]")
    ax.plot(limits, limits, color="k", ls="--", lw=1, label="signature = background")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, path)


def scr_vs_distance(ev, path):
    """Backscatter SCR vs distance of the BRP from the ship; returns the per-bin summary."""
    ok = np.isfinite(ev["dist"])
    if not ok.any():
        print("No ship_dist_km in the backscatter events: rerun the backscatter ensemble to plot SCR vs distance.")
        return None
    scr, dist = dbw(ev["sig"])[ok] - dbw(ev["ocean"])[ok], ev["dist"][ok]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(dist, scr, s=6, alpha=0.4, color=C_FWD, linewidths=0, rasterized=True)
    ax.axhline(0.0, color="grey", ls="--", lw=1)
    ax.set_xlabel(r"Distance of the $\mathrm{BRP}$ from the ship [km]")
    ax.set_ylabel(r"$\mathrm{SCR}$ [dB]")
    ax.grid(alpha=0.3)
    save(fig, path)
    rows = []
    for lo, hi in zip(DIST_BINS_KM[:-1], DIST_BINS_KM[1:], strict=True):
        sel = (dist >= lo) & (dist < hi)
        if sel.any():
            rows.append({"dist_km": [lo, None if np.isinf(hi) else hi], "n_events": int(sel.sum()),
                         "median_scr_db": float(np.median(scr[sel])), "fraction_scr_above_0db": float(np.mean(scr[sel] > 0.0))})
    return {"n_events_with_distance": int(ok.sum()), "max_scr_db": float(scr.max()),
            "distance_at_max_scr_km": float(dist[np.argmax(scr)]), "by_distance": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=os.path.join("OUTPUT", "ch11"))
    ap.add_argument("--output-dir", default=None, help="default: same as --input-dir")
    args = ap.parse_args()
    out_dir = args.output_dir or args.input_dir

    fwd, nt_f, und_f, zero_f = read_events(os.path.join(args.input_dir, "mc_specular_trials.jsonl"), "forward_shadow_pr_w")
    bks, nt_b, und_b, zero_b = read_events(os.path.join(args.input_dir, "mc_backscatter_trials.jsonl"), "pr_ship_w")
    if len(fwd["ocean"]) == 0 or len(bks["ocean"]) == 0:
        raise SystemExit("No valid power events: run both Configuration_ch11_*.ini ensembles first.")
    print(f"Excluded events (not plotted): forward {und_f} undefined + {zero_f} numerically zero; "
          f"backscatter {und_b} undefined + {zero_b} numerically zero")
    po_f, shadow, po_b, ship = fwd["ocean"], fwd["sig"], bks["ocean"], bks["sig"]
    lab_f = f"Forward ({len(po_f):,} events, {nt_f} trials)"
    lab_b = f"Backscatter ({len(po_b):,} events, {nt_b} trials)"
    ratio_f, scr_b = dbw(shadow) - dbw(po_f), dbw(ship) - dbw(po_b)  # 10log10(|dP_r|/P_glint), 10log10(P_ship/P_ocean)
    total_b = po_b + ship                                             # backscatter total = clutter + ship [W]

    # Figures 1-2: ship signature vs background per event; identical axis limits so the modes compare directly.
    levels = np.concatenate([dbw(po_f), dbw(shadow), dbw(po_b), dbw(ship)])
    limits = (5.0 * np.floor(levels.min() / 5.0) - 5.0, 5.0 * np.ceil(levels.max() / 5.0) + 5.0)
    signature_scatter(fwd, limits, r"Ocean glint power $P_\mathrm{glint}$ [dBW]", r"Ship shadow $|\Delta P_\mathrm{r}|$ [dBW]",
                      os.path.join(out_dir, "ch11_forward_shadow_vs_glint.png"), point_size=4)
    signature_scatter(bks, limits, r"Ocean clutter power $P_\mathrm{ocean}$ [dBW]", r"Ship power $P_\mathrm{ship}$ [dBW]",
                      os.path.join(out_dir, "ch11_backscatter_ship_vs_clutter.png"), point_size=8)

    # Figure 3: backscatter SCR vs distance of the BRP from the ship.
    dist_stats = scr_vs_distance(bks, os.path.join(out_dir, "ch11_scr_vs_distance.png"))

    # Figure 4: ship signature relative to its background (both modes).
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.histogram_bin_edges(np.concatenate([ratio_f, scr_b]), bins=60)
    hist_step(ax, ratio_f, bins, C_FWD, "-", lab_f + r": $|\Delta P_\mathrm{r}|/P_\mathrm{glint}$")
    hist_step(ax, scr_b, bins, C_BKS, "-", lab_b + r": $P_\mathrm{ship}/P_\mathrm{ocean}$")
    ax.axvline(0.0, color="k", ls="--", lw=1, label="ship signature = background")
    ax.set_xlabel(r"Ship signature relative to background [dB]")
    ax.set_ylabel("Fraction of events per bin")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, os.path.join(out_dir, "ch11_signal_ratio.png"))

    # Figure 5: absolute received power (total and ship signature).
    fig, ax = plt.subplots(figsize=(8, 5.5))
    series = [(dbw(po_f), C_FWD, "-", r"Forward total (glint) $P_\mathrm{glint}$"),
              (dbw(shadow), C_FWD, "--", r"Forward ship shadow $|\Delta P_\mathrm{r}|$"),
              (dbw(total_b), C_BKS, "-", r"Backscatter total $P_\mathrm{ocean}+P_\mathrm{ship}$"),
              (dbw(ship), C_BKS, "--", r"Backscatter ship $P_\mathrm{ship}$")]
    bins = np.histogram_bin_edges(np.concatenate([s[0] for s in series]), bins=70)
    for data, color, ls, label in series:
        hist_step(ax, data, bins, color, ls, label)
    ax.set_xlabel(r"Received power [dBW]")
    ax.set_ylabel("Fraction of events per bin")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, os.path.join(out_dir, "ch11_total_power.png"))

    stats = {
        "forward": {"n_events": int(len(po_f)), "n_trials": nt_f, "n_undefined": und_f, "n_numerically_zero": zero_f,
                    "median_glint_dbw": float(np.median(dbw(po_f))),
                    "median_shadow_dbw": float(np.median(dbw(shadow))),
                    "median_ratio_db": float(np.median(ratio_f)),
                    "ratio_db_p5_p95": [float(x) for x in np.percentile(ratio_f, [5, 95])],
                    "max_ratio_db": float(ratio_f.max())},
        "backscatter": {"n_events": int(len(po_b)), "n_trials": nt_b, "n_undefined": und_b, "n_numerically_zero": zero_b,
                        "median_total_dbw": float(np.median(dbw(total_b))),
                        "median_ocean_dbw": float(np.median(dbw(po_b))),
                        "median_ship_dbw": float(np.median(dbw(ship))),
                        "median_scr_db": float(np.median(scr_b)),
                        "scr_db_p5_p95": [float(x) for x in np.percentile(scr_b, [5, 95])],
                        "fraction_scr_above_0db": float(np.mean(scr_b > 0.0)),
                        "fraction_scr_above_10db": float(np.mean(scr_b > 10.0)),
                        "max_scr_db": float(scr_b.max()),
                        "scr_vs_distance": dist_stats},
    }
    path = os.path.join(out_dir, "ch11_signal_summary.json")
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
