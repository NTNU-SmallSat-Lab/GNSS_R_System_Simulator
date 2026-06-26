"""
SNR comparison: Galileo vs three LEO PNT Tx amplifier scenarios.

Usage (defaults):
    python src/plot_snr_comparison.py

Or with explicit paths:
    python src/plot_snr_comparison.py <galileo.npy> <leopnt.npy> <gaas.npy> <ldmos.npy> [shapefile.shp]
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

matplotlib.use("Agg", force=True)

SHAPEFILE       = "data/International_hydrographic_organization/iho_norwegian_coast_gridded_m.shp"
DEFAULT_GALILEO = "OUTPUT/snr_max_area_galileo_lb_cellcentric.npy"
DEFAULT_LEOPNT  = "OUTPUT/snr_max_area_leopnt_lb_cellcentric.npy"
DEFAULT_GAAS    = "OUTPUT/snr_max_area_gaas_lb_cellcentric.npy"
OUTPUT_FILE     = "OUTPUT/SNR_comparison_galileo_vs_leopnt.png"

PANELS = [
    ("galileo", "Galileo Tx  (1 sat, MEO ~23 200 km)\nE1  |  EIRP 27 dBW  (constant)"),
    ("leopnt",  "LEO PNT — isoflux baseline  (CELESTE IOD3, ~568 km)\nL1  |  35 dBm PA, isoflux 1–3 dBi  →  EIRP 5–8 dBW"),
    ("gaas",    "LEO PNT — GaAs PA  (CELESTE IOD3, ~568 km)\nL1  |  40 dBm PA, isoflux 1–3 dBi  →  EIRP 10–13 dBW"),
]


def _snr_panel(ax, grid_gdf, snr, cmap, norm, title, snr_threshold_db=0.0):
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    pc = ccrs.PlateCarree()
    for i, row in grid_gdf.iterrows():
        s = snr[i]
        if not np.isfinite(s):
            continue
        xs, ys = row["geometry"].exterior.xy
        ax.fill(xs, ys, color=cmap(norm(s)), transform=pc, zorder=3)

    finite = snr[np.isfinite(snr)]
    n_above = int(np.sum(finite >= snr_threshold_db))
    if len(finite):
        stats = (f"min={finite.min():.0f}  med={np.median(finite):.0f}  "
                 f"max={finite.max():.0f} dB  |  "
                 f"{n_above}/{len(finite)} cells ≥ {snr_threshold_db:.0f} dB")
    else:
        stats = "no covered cells"
    ax.set_title(f"{title}\n{stats}", fontsize=12)


def main():
    galileo_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GALILEO
    leopnt_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LEOPNT
    gaas_path    = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_GAAS
    shp_path     = sys.argv[4] if len(sys.argv) > 4 else SHAPEFILE

    for p in [galileo_path, leopnt_path, gaas_path, shp_path]:
        if not os.path.exists(p):
            print(f"Missing file: {p}")
            sys.exit(1)

    snr = {
        "galileo": np.load(galileo_path),
        "leopnt":  np.load(leopnt_path),
        "gaas":    np.load(gaas_path),
    }
    grid_gdf = gpd.read_file(shp_path)

    # Shared colour scale across all 3 SNR panels
    all_finite = np.concatenate([snr[k][np.isfinite(snr[k])] for k in snr])
    vmin = np.percentile(all_finite, 2)  if len(all_finite) else -120
    vmax = np.percentile(all_finite, 98) if len(all_finite) else -20
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Layout: 3 SNR panels + 2 diff panels (vs Galileo), colorbar column on right
    n_snr  = 3
    n_diff = 2
    n_rows = n_snr + n_diff

    fig = plt.figure(figsize=(13, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig,
                            width_ratios=[22, 1],
                            hspace=0.38, wspace=0.08)

    snr_axes  = [fig.add_subplot(gs[r, 0], projection=ccrs.PlateCarree()) for r in range(n_snr)]
    diff_axes = [fig.add_subplot(gs[n_snr + r, 0], projection=ccrs.PlateCarree()) for r in range(n_diff)]
    cax_snr  = fig.add_subplot(gs[:n_snr, 1])
    cax_diff = fig.add_subplot(gs[n_snr:, 1])

    # --- SNR panels ---
    for ax, (key, title) in zip(snr_axes, PANELS):
        _snr_panel(ax, grid_gdf, snr[key], cmap, norm, title)

    # --- Difference panels: LEO PNT variants − Galileo ---
    diff_specs = [
        ("leopnt", "ΔSNR = LEO PNT isoflux (35 dBm) − Galileo"),
        ("gaas",   "ΔSNR = GaAs isoflux (40 dBm) − Galileo"),
    ]
    all_diffs = []
    for key, _ in diff_specs:
        d = np.where(np.isfinite(snr[key]) & np.isfinite(snr["galileo"]),
                     snr[key] - snr["galileo"], np.nan)
        all_diffs.append(d)

    d_abs = max(
        max(np.nanmax(np.abs(d)) for d in all_diffs if np.any(np.isfinite(d))),
        1.0
    )
    d_norm = mcolors.TwoSlopeNorm(vmin=-d_abs, vcenter=0, vmax=d_abs)
    d_cmap = plt.cm.bwr

    pc = ccrs.PlateCarree()
    for ax, diff, (key, title) in zip(diff_axes, all_diffs, diff_specs):
        ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.xlabel_style = {"size": 11}
        gl.ylabel_style = {"size": 11}
        for i, row in grid_gdf.iterrows():
            d = diff[i]
            if not np.isfinite(d):
                continue
            xs, ys = row["geometry"].exterior.xy
            ax.fill(xs, ys, color=d_cmap(d_norm(d)), transform=pc, zorder=3)
        finite_d = diff[np.isfinite(diff)]
        stats = (f"mean={finite_d.mean():.1f}  med={np.median(finite_d):.1f}  "
                 f"range=[{finite_d.min():.0f}, {finite_d.max():.0f}] dB"
                 if len(finite_d) else "no common coverage")
        ax.set_title(f"{title}\n{stats}", fontsize=12)

    # --- Colorbars ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax_snr)
    cb.set_label("Peak SNR per cell [dB]", fontsize=12)
    cb.ax.tick_params(labelsize=11)

    sm2 = plt.cm.ScalarMappable(cmap=d_cmap, norm=d_norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cax_diff)
    cb2.set_label("ΔSNR  [LEO PNT] − Galileo [dB]", fontsize=12)
    cb2.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Bistatic SNR comparison — Galileo vs LEO PNT Tx  |  L1 (1575.42 MHz)\n"
        "1 Rx CELESTE satellite @ 568 km, 3×4 array, isoflux Tx antenna 1–3 dBi, "
        "σ₀ = −20 dB, A_cell = 900 km², T_sys = 290 K, B = 10 MHz",
        fontsize=14, y=1.005,
    )
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SNR comparison saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
