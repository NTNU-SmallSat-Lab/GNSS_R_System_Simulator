"""Chapter-11 illustration figure: backscatter ship-vs-ocean SCR and SINR versus the distance of
the bistatic reflection point (BRP) from the tanker. Combines the representative ensemble
(illustration_events_backscatter.json, BRP tens-to-hundreds of km off the ship) with the phased
co-located near-retroreflection snapshot (illustration_events_colocated.json, BRP on the ship).

Run from repo root:  python SRC/plot_illustration.py
"""
import json

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

DET_THR_DB = 0.0  # configured detection SINR threshold


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ens = _load("illustration_events_backscatter.json")
    col = [r for r in _load("illustration_events_colocated.json") if r["refl_km_from_ship"] < 5.0]

    def arr(recs, key):
        return np.array([r[key] for r in recs], float)

    ens_km, ens_scr, ens_sinr = arr(ens, "refl_km_from_ship"), arr(ens, "scr_db"), arr(ens, "sinr_db")
    col_km, col_scr, col_sinr = arr(col, "refl_km_from_ship"), arr(col, "scr_db"), arr(col, "sinr_db")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ens_y, col_y, ylab in ((ax1, ens_scr, col_scr, "SCR [dB]"),
                                    (ax2, ens_sinr, col_sinr, "SINR [dB]")):
        ax.scatter(ens_km, ens_y, s=28, c="#4477aa")
        ax.scatter(col_km, col_y, s=28, c="#4477aa")
        ax.set_xlabel("BRP distance from ship [km]")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    ax2.axhline(DET_THR_DB, ls="--", c="grey", lw=1)
    ax2.text(ax2.get_xlim()[1], DET_THR_DB, " detection threshold", va="bottom", ha="right", color="grey", fontsize=8)
    ax1.set_title("Signal-to-clutter ratio")
    ax2.set_title("Signal-to-interference-plus-noise ratio")
    fig.suptitle("Backscatter ship detection vs bistatic reflection-point co-location\n"
                 "(dark-fleet tanker, open Skagerrak, GPS L1, 1 Oct 2025)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "OUTPUT/Illustration_SCR_SINR_vs_colocation.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
