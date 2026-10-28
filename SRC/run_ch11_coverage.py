"""Chapter-11 area coverage: specular and backscatter maps plus the per-cell visit counts, from one run.

Uses one area-coverage INI (platform, window, grid, backscatter cone) and overrides only the reflection
mode, so both maps and the counts table come from identical settings. Writes to OUTPUT/<out_subdir>/:
  Coverage_grid_specular.png, Coverage_grid_backscatter.png, coverage_counts.json
Run from the repo root:  python SRC/run_ch11_coverage.py [ini (Configuration_area.ini)] [out_subdir (ch11)]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np

from __init__ import initialize
import functions.functions as fn
from object_definitions.Constants_def import Constants
from object_definitions.Coverage_def import Coverage
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite


def run_mode(settings, mode, out_subdir):
    """Compute coverage for one reflection mode, save its map, and return the visit-count buckets."""
    settings = dict(settings)
    settings["reflection_type"] = mode
    gnss = fn.load_gnss_constellation_dict(settings["GNSS_constellations"])
    antenna = RxAntenna(settings) if settings.get("antenna_pattern_folder") is not None else None
    sats = [Satellite(i, orbit_n=settings["orbits"][i], rx_antenna=antenna) for i in range(settings["No_satellites"])]
    const = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sats)
    const.set_simulation_duration(settings["start_time"], settings["sim_duration"], settings["step_size"])
    fn.propagate_all_orbits(const, gnss)

    cov = Coverage(
        const, gnss, settings["shapefile_path"], grid_size_m=settings["grid_size_m"], earth_model=settings["earth_model"],
        link_budget_settings=settings.get("link_budget"), max_central_angle_deg=settings["max_central_angle_deg"],
        selected_signals=settings["selected_signals"], reflection_type=mode,
        max_incidence_angle_deg=settings["max_incidence_angle_deg"],
    )
    if not cov.coverage_constellations:
        raise ValueError("No usable constellations for coverage. Check [ANTENNA].bands.")
    for sig in cov.coverage_constellations:
        cov.compute_coverage_for_constellation(constellation=sig["constellation"], freq_hz=sig["freq_hz"],
                                               eirp_dbw=sig["eirp_dbw"], verbose=True)

    consts = sorted({sig["constellation"] for sig in cov.coverage_constellations})
    cov.plot_coverage_gridded_shapefile(gnss_constellation=", ".join(consts), file_name=f"{out_subdir}/Coverage_grid_{mode}.png", title="")  # no title: the caption carries it

    hits = np.array([sum(len(cov.coverage_dict[c][i].times) for c in consts) for i in range(len(cov.grid_gdf))])
    return {"mode": mode, "constellations": consts, "total_cells": int(hits.size),
            "hit_0": int((hits == 0).sum()), "hit_1": int((hits == 1).sum()), "hit_2": int((hits == 2).sum()),
            "hit_gt2": int((hits > 2).sum()), "covered": int((hits > 0).sum()), "max_hits": int(hits.max())}


def main():
    ini = sys.argv[1] if len(sys.argv) > 1 else "Configuration_area.ini"
    out_subdir = sys.argv[2] if len(sys.argv) > 2 else "ch11"
    _, settings = initialize(ini)
    os.makedirs(os.path.join(Constants.output_folder_path, out_subdir), exist_ok=True)
    out = {mode: run_mode(settings, mode, out_subdir) for mode in ("specular", "backscatter")}
    out["settings"] = {"ini": ini, "window_h": settings["sim_duration"], "grid_m": settings["grid_size_m"],
                       "max_incidence_deg": settings["max_incidence_angle_deg"],
                       "max_central_angle_deg": settings["max_central_angle_deg"]}
    path = os.path.join(Constants.output_folder_path, out_subdir, "coverage_counts.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
