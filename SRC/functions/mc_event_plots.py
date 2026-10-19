"""Event-level power persistence and plots for Monte Carlo moving-target runs."""

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Match each constellation across the power scatter and ratio histogram. Markers/line styles
# provide an additional distinction when colors are difficult to distinguish.
_STYLES = {
    "GPS": ("#4477aa", "o", "-"),
    "Galileo": ("#ee6677", "^", "--"),
    "GLONASS": ("#228833", "s", ":"),
    "BeiDou": ("#aa3377", "D", "-."),
    "QZSS": ("#ccbb44", "v", "-"),
    "IRNSS": ("#66ccee", "P", "--"),
}


def _signature_field(mode):
    if mode == "backscatter":
        return "pr_ship_w"
    if mode == "specular":
        return "forward_shadow_pr_w"
    raise ValueError(f"Unsupported MC mode: {mode}")


def _finite_float(value):
    """Return a JSON-safe power scalar, retaining nonpositive values for auditing."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def serialize_power_events(target_detect_dict, mode):
    """Serialize paired powers and identity for every footprint-overlap event.

    Parameters
    ----------
    target_detect_dict : dict
        Detector event lists keyed by constellation; no SINR selection is applied.
    mode : {"backscatter", "specular"}
        Select ship power or the positive forward-shadow magnitude, in watts.

    Returns
    -------
    list of dict
        Compact event records. Nonfinite or unavailable powers become JSON null;
        validation for plotting is performed on each complete pair.
    """
    field = _signature_field(mode)
    records = []
    for constellation, events in target_detect_dict.items():
        for event in events:
            timestamp = getattr(event, "time", None)
            record = {
                "constellation": constellation,
                "band": event.band,
                "time": timestamp.isoformat() if timestamp is not None else None,
                "pr_ocean_w": _finite_float(event.pr_ocean_w),
                field: _finite_float(getattr(event, field, None)),
            }
            for key in ("tx_satID", "rx_satID"):
                value = getattr(event, key, None)
                record[key] = value.item() if isinstance(value, np.generic) else value
            record["ship_dist_km"] = _ship_distance_km(event)
            records.append(record)
    return records


def _ship_distance_km(event):
    """Great-circle distance [km] from the reflection point to the ship, or None if either is unavailable."""
    point, ship = getattr(event, "reflection_point_ll", None), getattr(event, "pos_tgt_i", None)
    if point is None or ship is None:
        return None
    lat1, lon1 = np.radians(float(point[0])), np.radians(float(point[1]))  # reflection point [lat, lon]
    lat2, lon2 = np.radians(float(ship.y)), np.radians(float(ship.x))      # ship Point(x = lon, y = lat)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2  # haversine
    return _finite_float(2.0 * 6371.0 * np.arcsin(np.sqrt(a)))


def _collect_event_powers(results, mode):
    """Collect (ocean dBW, signature dBW, ratio dB) rows, always from one event."""
    field = _signature_field(mode)
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    counts = {
        "n_trials": len(results),
        "n_legacy_trials": 0,
        "n_contributing_trials": 0,
        "n_events": 0,
        "n_invalid_events": 0,
    }
    for result in results:
        events = result.get("power_events")
        if events is None:
            counts["n_legacy_trials"] += 1
            continue
        contributed = False
        for event in events:
            ocean = _finite_float(event.get("pr_ocean_w"))
            signature = _finite_float(event.get(field))
            if ocean is None or signature is None or ocean <= 0.0 or signature <= 0.0:
                counts["n_invalid_events"] += 1
                continue
            ocean_dbw = 10.0 * np.log10(ocean)
            signature_dbw = 10.0 * np.log10(signature)
            # Subtract logs instead of forming a ratio that might underflow/overflow.
            row = (ocean_dbw, signature_dbw, signature_dbw - ocean_dbw)
            constellation = event.get("constellation") or "Unknown"
            grouped.setdefault(constellation, []).append(row)
            counts["n_events"] += 1
            contributed = True
        counts["n_contributing_trials"] += int(contributed)
    return {key: np.asarray(rows) for key, rows in sorted(grouped.items())}, counts


def make_event_plots(results, mode, out_dir):
    """Write standalone power and ratio PNGs from saved MC event records.

    Parameters
    ----------
    results : list of dict
        Trial JSONL records, including legacy records without power_events.
    mode : {"backscatter", "specular"}
        Select the ship-power SCR or forward-shadow fraction.
    out_dir : str or os.PathLike
        Existing output directory.

    Returns
    -------
    list of str
        Saved paths, or an empty list when no finite, positive power pairs exist.

    Notes
    -----
    Selected bands are combined. Each event contributes one sample, so trials
    with more overlap events carry more weight. Sampled coherent integration time
    still affects delay-Doppler clutter area when that model is enabled.
    """
    grouped, counts = _collect_event_powers(results, mode)
    print(
        f"Event power plots: {counts['n_events']} valid events from "
        f"{counts['n_contributing_trials']}/{counts['n_trials']} trials; "
        f"excluded {counts['n_legacy_trials']} trials without saved event data "
        f"and {counts['n_invalid_events']} invalid/nonpositive power pairs."
    )
    if not grouped:
        print("No valid event power pairs; skipping new event plots.")
        return []

    stem = os.path.join(out_dir, f"mc_{mode}")
    saved = []
    title = (
        f"Monte Carlo footprint-overlap events — {mode}\n"
        f"Selected bands combined | {counts['n_events']:,} events\n"
        f"{counts['n_contributing_trials']:,} contributing trials / {counts['n_trials']:,} total trials"
    )
    note = (
        "Each valid event contributes one sample; trials with more events carry more weight.\n"
        "Sampled coherent integration time affects delay-Doppler clutter area when enabled."
    )
    if counts["n_legacy_trials"] or counts["n_invalid_events"]:
        note += f"\nExcluded: {counts['n_legacy_trials']:,} trials without event data; {counts['n_invalid_events']:,} invalid/nonpositive power pairs."

    def finish(fig, ax, suffix):
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
        fig.suptitle(title, fontsize=11)
        fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        path = f"{stem}_{suffix}.png"
        try:
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for constellation, rows in grouped.items():
        color, marker, _ = _STYLES.get(constellation, ("#666666", "x", ":"))
        ax.scatter(
            rows[:, 0],
            rows[:, 1],
            color=color,
            marker=marker,
            s=22,
            alpha=0.65,
            label=f"{constellation} ({len(rows):,} events)",
        )
    ax.set_xlabel(r"Ocean received power $P_\mathrm{ocean}$ [dBW]")
    if mode == "backscatter":
        all_powers = np.concatenate(list(grouped.values()))[:, :2]
        low, high = float(all_powers.min()), float(all_powers.max())
        pad = max(0.05 * (high - low), 1.0)
        limits = (low - pad, high + pad)
        ax.plot(limits, limits, color="#333333", linestyle="--", linewidth=1, label="Equal power")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_ylabel(r"Ship received power $P_\mathrm{ship}$ [dBW]")
        ax.set_title("Ship versus ocean received power")
    else:
        ax.set_ylabel(r"Forward-shadow magnitude $|\Delta P|$ [dBW]")
        ax.set_title("Forward-shadow magnitude versus ocean received power")
    finish(fig, ax, "received_power")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ratios = np.concatenate([rows[:, 2] for rows in grouped.values()])
    bins = np.histogram_bin_edges(ratios, bins=30).tolist()
    for constellation, rows in grouped.items():
        color, _, linestyle = _STYLES.get(constellation, ("#666666", "x", ":"))
        ax.hist(
            rows[:, 2],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=color,
            linestyle=linestyle,
            label=f"{constellation} ({len(rows):,} events)",
        )
    ax.set_ylabel("Events")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if mode == "backscatter":
        ax.set_xlabel(r"SCR: $10\log_{10}(P_\mathrm{ship}/P_\mathrm{ocean})$ [dB]")
        ax.set_title("Ship-to-ocean power ratio")
        suffix = "scr"
    else:
        ax.set_xlabel(r"Shadow fraction: $10\log_{10}(|\Delta P|/P_\mathrm{ocean})$ [dB]")
        ax.set_title("Forward-shadow fraction")
        suffix = "shadow_fraction"
    finish(fig, ax, suffix)
    return saved
