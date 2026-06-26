#!/usr/bin/env python3
"""
Unified revisit-count comparison across all three constellation scenarios.
All panels share the same colour scale (global max across every scenario).
Layout: 3 rows × 2 columns
  col 0 = Tx beam only (reference)
  col 1 = Tx + steered Rx beam
  rows  = 2-sat, 4-sat, 8-sat
"""

import sys
sys.path.insert(0, '/home/thebear/codes4fun/master/sim/GNSS_R_System_Simulator/src')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from __init__ import initialize

SCENARIOS = [
    {"tag": "1sat", "label": "1 pair   (SAT_01)"},
    {"tag": "2sat", "label": "2 pairs  (SAT_01 + SAT_05)"},
    {"tag": "4sat", "label": "4 pairs  (SAT_01, 03, 05, 07)"},
    {"tag": "8sat", "label": "8 pairs  (full fleet)"},
]

# ── Load saved count arrays ───────────────────────────────────────────────────
def _load(tag, kind):
    return np.load(f"OUTPUT/counts_{kind}_{tag}_cellcentric.npy")

data = {}
for sc in SCENARIOS:
    t = sc["tag"]
    data[t] = {
        "tx":   _load(t, "tx"),
        "txrx": _load(t, "txrx"),
    }

# ── Load grid geometry (same for all scenarios) ───────────────────────────────
_, settings = initialize("Configuration_2sat.ini")
grid_gdf = gpd.read_file(settings["shapefile_path"].replace(".shp", "_gridded_m.shp"))

# ── Compute global max for each column ───────────────────────────────────────
tx_max   = max(data[sc["tag"]]["tx"].max()   for sc in SCENARIOS)
txrx_max = max(data[sc["tag"]]["txrx"].max() for sc in SCENARIOS)

# Use a single vmax across BOTH columns so relative reduction is visible too
vmax = max(tx_max, txrx_max)

cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    len(SCENARIOS), 2,
    figsize=(18, 5 * len(SCENARIOS)),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

pc = ccrs.PlateCarree()

def _panel(ax, grid, counts, title):
    ax.add_feature(cfeature.LAND,      facecolor="#d8d8d8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    for i, row in grid.iterrows():
        if counts[i] == 0:
            continue
        xs, ys = row["geometry"].exterior.xy
        ax.fill(xs, ys, color=cmap(norm(counts[i])), transform=pc, zorder=3)
    nz = counts[counts > 0]
    stats = (f"min={nz.min() if len(nz) else 0}  "
             f"median={int(np.median(nz)) if len(nz) else 0}  "
             f"max={counts.max()}"
             if counts.max() > 0 else "no coverage")
    ax.set_title(f"{title}\n{stats}", fontsize=9)

for row_i, sc in enumerate(SCENARIOS):
    t = sc["tag"]
    _panel(axes[row_i, 0], grid_gdf, data[t]["tx"],
           f"[{sc['label']}]  Tx beam only (reference)")
    _panel(axes[row_i, 1], grid_gdf, data[t]["txrx"],
           f"[{sc['label']}]  Tx + steered Rx  (steer_fraction=0.2)")

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.04)
cb.set_label("Illumination count (timesteps)  —  same scale across all scenarios",
             fontsize=10)

fig.suptitle("Revisit count comparison — Norwegian Sea  |  24 h  |  30 s step",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out = "OUTPUT/Revisit_comparison_all.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
