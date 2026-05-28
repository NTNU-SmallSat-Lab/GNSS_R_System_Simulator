from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from astropy.utils import iers

    iers.conf.auto_download = False
except Exception:
    pass

from __init__ import initialize
from functions.functions import compute_footprint
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna

L1_MHZ = 1575.42
L5_MHZ = 1176.45


def _nearest_pattern_key(pattern_files: dict, frequency_mhz: float) -> float:
    if not pattern_files:
        raise AssertionError("No antenna pattern files are available for the footprint diagnostic.")
    return min(pattern_files, key=lambda key: abs(key - frequency_mhz))


def _lightweight_antenna_settings(settings: dict, frequencies_mhz: list[float], footprint_db: float) -> dict:
    pattern_files = settings["antenna_pattern_files"]
    selected_patterns = {}
    for frequency_mhz in frequencies_mhz:
        key = _nearest_pattern_key(pattern_files, frequency_mhz)
        selected_patterns[key] = pattern_files[key]

    return {
        **settings,
        "antenna_pattern_files": selected_patterns,
        "footprint_dB": footprint_db,
    }


def _bundle_stats(bundle: np.ndarray) -> dict:
    norms = np.linalg.norm(bundle, axis=1)
    off_boresight_deg = np.degrees(np.arccos(np.clip(bundle[:, 2] / norms, -1.0, 1.0)))
    x_plane = bundle[:, 0] / bundle[:, 2]
    y_plane = bundle[:, 1] / bundle[:, 2]
    return {
        "points": int(bundle.shape[0]),
        "off_min_deg": float(np.min(off_boresight_deg)),
        "off_mean_deg": float(np.mean(off_boresight_deg)),
        "off_max_deg": float(np.max(off_boresight_deg)),
        "x_radius_at_1m": float(np.max(np.abs(x_plane))),
        "y_radius_at_1m": float(np.max(np.abs(y_plane))),
        "unit_min": float(np.min(norms)),
        "unit_max": float(np.max(norms)),
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _print_stats(label: str, stats: dict) -> None:
    formatted = ", ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in stats.items()
    )
    print(f"{label}: {formatted}")


def main(config_file: str = "Configuration_ship.ini") -> None:
    print(f"Footprint calculation diagnostic using {config_file}")
    _, settings = initialize(config_file)

    contour_levels = [-0.5, settings["footprint_dB"], -3.0]
    contour_levels = sorted({float(level) for level in contour_levels}, reverse=True)

    contour_stats = {}
    for contour_db in contour_levels:
        antenna = RxAntenna(_lightweight_antenna_settings(settings, [L1_MHZ, L5_MHZ], contour_db))
        l1_bundle = antenna._get_antenna_footprint(np.array([0.0, 0.0, 1.0]), L1_MHZ)
        contour_stats[contour_db] = _bundle_stats(l1_bundle)
        _print_stats(f"L1 contour {contour_db} dB", contour_stats[contour_db])

        norms = np.linalg.norm(l1_bundle, axis=1)
        _assert(np.allclose(norms, 1.0, atol=1e-10), f"Footprint vectors are not unit length for {contour_db} dB")

    ordered_levels = sorted(contour_stats, reverse=True)
    mean_angles = [contour_stats[level]["off_mean_deg"] for level in ordered_levels]
    _assert(
        all(a < b for a, b in zip(mean_angles, mean_angles[1:], strict=False)),
        "More negative footprint_dB should produce a wider contour.",
    )

    antenna = RxAntenna(_lightweight_antenna_settings(settings, [L1_MHZ, L5_MHZ], settings["footprint_dB"]))
    l1_stats = _bundle_stats(antenna._get_antenna_footprint(np.array([0.0, 0.0, 1.0]), L1_MHZ))
    l5_stats = _bundle_stats(antenna._get_antenna_footprint(np.array([0.0, 0.0, 1.0]), L5_MHZ))
    _print_stats("L1 stored-frequency footprint", l1_stats)
    _print_stats("L5 stored-frequency footprint", l5_stats)
    _assert(
        abs(l1_stats["off_mean_deg"] - l5_stats["off_mean_deg"]) > 0.1,
        "L1 and L5 footprint contours should differ for this antenna model.",
    )

    sat_pos = np.array([7_100_000.0, 0.0, 0.0])
    sat_vel = np.array([0.0, 7_500.0, 0.0])
    refl_pos = np.array([Constants.R_EARTH_MEAN, 0.0, 0.0])
    time_i = datetime(2025, 4, 2, 12, 0, 0, tzinfo=UTC)
    polygon, center = compute_footprint(sat_pos, sat_vel, refl_pos, time_i, antenna, L1_MHZ, earth_model="spherical")
    _assert(not polygon.is_empty, "Synthetic footprint polygon is empty.")
    _assert(polygon.is_valid, "Synthetic footprint polygon is invalid.")
    print(
        "Projected L1 polygon: "
        f"area_deg2={polygon.area:.6g}, bounds_lonlat={tuple(round(v, 6) for v in polygon.bounds)}, "
        f"center_latlon={tuple(round(float(v), 6) for v in center)}"
    )

    print("Footprint calculation diagnostic passed.")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "Configuration_ship.ini"
    main(config)
