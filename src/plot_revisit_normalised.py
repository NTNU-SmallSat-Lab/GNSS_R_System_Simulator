"""
Replot revisit count maps normalised to a 24-hour equivalent rate.
Loads saved .npy count arrays from the 114.2h area coverage runs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

SIM_DURATION_H  = 114.2
NORM_DURATION_H = 24.0
SCALE           = NORM_DURATION_H / SIM_DURATION_H

SHAPEFILE = "data/International_hydrographic_organization/iho_norwegian_coast.shp"
OUT_DIR   = "OUTPUT"

CONSTELLATIONS = ["1sat", "2sat", "4sat", "8sat"]


def _map_panel(ax, grid_gdf, counts_norm, cmap, norm, title):
    pc = ccrs.PlateCarree()
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    for i, row in grid_gdf.iterrows():
        v = counts_norm[i]
        if v <= 0:
            continue
        xs, ys = row["geometry"].exterior.xy
        ax.fill(xs, ys, color=cmap(norm(v)), transform=pc, zorder=3)
    finite = counts_norm[counts_norm > 0]
    stats = (f"min={finite.min():.1f}  median={np.median(finite):.1f}  max={finite.max():.1f}"
             if len(finite) else "no coverage")
    ax.set_title(f"{title}\n{stats}", fontsize=9)


def plot_constellation(label, grid_gdf):
    tx   = np.load(f"{OUT_DIR}/counts_tx_area_{label}_cellcentric.npy").astype(float)
    txrx = np.load(f"{OUT_DIR}/counts_txrx_area_{label}_cellcentric.npy").astype(float)

    tx_norm   = tx   * SCALE
    txrx_norm = txrx * SCALE

    vmax = max(tx_norm.max(), 1)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             subplot_kw={"projection": ccrs.PlateCarree()})

    _map_panel(axes[0], grid_gdf, tx_norm,   cmap, norm,
               f"Tx beam only  (reference)\n[illumination count / 24 h equivalent]")
    _map_panel(axes[1], grid_gdf, txrx_norm, cmap, norm,
               f"Tx + steered Rx beam  (steer_fraction=0.2)\n[illumination count / 24 h equivalent]")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cb.set_label("Illumination count (30 s timesteps per 24 h)")

    fig.suptitle(
        f"Revisit count — {label}  |  normalised to 24 h  "
        f"(actual sim: {SIM_DURATION_H} h  ×  {SCALE:.3f})",
        fontsize=11, fontweight="bold",
    )
    out = f"{OUT_DIR}/Revisit_count_area_{label}_cc_24h.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    grid_gdf = gpd.read_file(SHAPEFILE)
    for label in CONSTELLATIONS:
        plot_constellation(label, grid_gdf)


if __name__ == "__main__":
    main()
