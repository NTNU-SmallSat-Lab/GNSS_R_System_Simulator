"""Chapter-11 illustration figures: backscatter ship-vs-ocean SCR and SINR versus the distance of
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

    plots = (
        ("SCR", ens_scr, col_scr, "Signal-to-clutter ratio"),
        ("SINR", ens_sinr, col_sinr, "Signal-to-interference-plus-noise ratio"),
    )
    for metric, ens_y, col_y, title in plots:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.scatter(ens_km, ens_y, s=28, c="#4477aa")
        ax.scatter(col_km, col_y, s=28, c="#4477aa")
        ax.set_xlabel(r"$\mathrm{BRP}$ distance from ship [km]")
        ax.set_ylabel(rf"$\mathrm{{{metric}}}$ [dB]")
        ax.grid(True, alpha=0.3)
        if metric == "SINR":
            ax.axhline(DET_THR_DB, ls="--", c="grey", lw=1)
            ax.text(ax.get_xlim()[1], DET_THR_DB, " detection threshold", va="bottom", ha="right", color="grey", fontsize=8)
        ax.set_title(title)
        fig.suptitle("Backscatter ship detection vs BRP co-location\n"
                     "(dark-fleet tanker, open Skagerrak, GPS L1, 1 Oct 2025)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = f"OUTPUT/Illustration_{metric}_vs_colocation.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
