"""
Generate flowchart PNGs for every simulation in the GNSS-R System Simulator.
Uses pure matplotlib — no system graphviz needed.
Output: OUTPUT/flowchart_*.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "OUTPUT"

# ── Drawing primitives ──────────────────────────────────────────────────────

C_PROCESS  = ("#D6EAF8", "#2E86C1")   # fill, edge
C_DECISION = ("#FDEBD0", "#CA6F1E")
C_IO       = ("#D5F5E3", "#1E8449")
C_TERMINAL = ("#E8DAEF", "#7D3C98")
C_CLUSTER  = ("#F8F9FA", "#95A5A6")

FONT = "DejaVu Sans"


def _box(ax, xy, w, h, text, color, fontsize=8.5, style="round,pad=0.1"):
    fill, edge = color
    box = FancyBboxPatch((xy[0] - w/2, xy[1] - h/2), w, h,
                         boxstyle=style,
                         linewidth=1.4, edgecolor=edge, facecolor=fill)
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=fontsize, fontfamily=FONT, wrap=True,
            multialignment="center")


def _diamond(ax, xy, w, h, text, color, fontsize=8.5):
    fill, edge = color
    cx, cy = xy
    hw, hh = w/2, h/2
    pts = [(cx, cy+hh), (cx+hw, cy), (cx, cy-hh), (cx-hw, cy)]
    poly = plt.Polygon(pts, closed=True,
                       linewidth=1.4, edgecolor=edge, facecolor=fill)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize-0.5, fontfamily=FONT, multialignment="center")


def _arrow(ax, src, dst, label="", color="#444444"):
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, mutation_scale=12))
    if label:
        mx, my = (src[0]+dst[0])/2, (src[1]+dst[1])/2
        ax.text(mx+0.02, my, label, fontsize=7.5, color="#555555",
                fontfamily=FONT, ha="left", va="center")


def _cluster_rect(ax, xy, w, h, label, color="#95A5A6"):
    x0, y0 = xy[0]-w/2, xy[1]-h/2
    rect = mpatches.FancyBboxPatch((x0, y0), w, h,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1.0, edgecolor=color,
                                   facecolor="#F8F9FA", linestyle="dashed",
                                   zorder=0)
    ax.add_patch(rect)
    ax.text(x0+0.04, y0+h-0.04, label, fontsize=7, color=color,
            fontfamily=FONT, va="top", style="italic")


def _save(fig, name):
    path = os.path.join(OUT_DIR, f"flowchart_{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Area Coverage Simulation
# ═══════════════════════════════════════════════════════════════════════════
def flowchart_area_coverage():
    fig, ax = plt.subplots(figsize=(5.5, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    W, H = 0.55, 0.062
    X = 0.50
    steps = [
        (0.94, "START",                           C_TERMINAL),
        (0.84, "Load config\n(Configuration_area.ini)", C_IO),
        (0.74, "Load Tx GNSS constellation (TLE)",      C_PROCESS),
        (0.64, "Create Rx satellite constellation\n& RxAntenna", C_PROCESS),
        (0.54, "Generate Rx TLEs\n& set simulation duration",    C_PROCESS),
        (0.44, "Propagate all orbits (SGP4)",             C_PROCESS),
        (0.34, "Calculate sunlight fraction\nper Rx satellite",  C_PROCESS),
        (0.24, "Coverage object\n(shapefile + 30 km grid)",      C_PROCESS),
        (0.15, "Calculate reflection points\n& trajectory coverage\n(Rx × Tx × timestep)", C_PROCESS),
        (0.06, "Plot reflection points,\ncoverage maps & orbit tracks",  C_IO),
    ]

    for y, text, col in steps:
        if col == C_TERMINAL:
            _box(ax, (X, y), 0.28, 0.052, text, col, style="round,pad=0.15")
        else:
            _box(ax, (X, y), W, H, text, col)

    for i in range(len(steps)-1):
        y0 = steps[i][0] - H/2
        y1 = steps[i+1][0] + H/2
        _arrow(ax, (X, y0), (X, y1))

    # statistics box off to side
    _box(ax, (X, -0.02), W, H, "Calculate & print coverage statistics", C_PROCESS)
    _arrow(ax, (X, steps[-1][0]-H/2), (X, -0.02+H/2))

    fig.suptitle("Area Coverage Simulation\n(main_simulation_areaCoverage.py)",
                 fontsize=10, fontfamily=FONT, fontweight="bold", y=0.995)
    _save(fig, "area_coverage")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Area Coverage Cell-Centric
# ═══════════════════════════════════════════════════════════════════════════
def flowchart_area_coverage_cc():
    fig, ax = plt.subplots(figsize=(5.8, 15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    W, H = 0.60, 0.057
    X = 0.50

    _cluster_rect(ax, (X, 0.745), 0.74, 0.47, "Phase 1 — Setup & Propagation", "#2E86C1")
    ph1 = [
        (0.97, "START",                                     C_TERMINAL),
        (0.90, "Load config (+ optional link budget)",      C_IO),
        (0.83, "Load Tx GNSS constellation",                C_PROCESS),
        (0.77, "Create Rx constellation & RxAntenna",      C_PROCESS),
        (0.70, "Propagate all orbits",                      C_PROCESS),
        (0.63, "Calculate sunlight fraction",               C_PROCESS),
        (0.57, "Coverage object (shapefile + 30 km grid)",  C_PROCESS),
        (0.50, "Trajectory-based reflection points",        C_PROCESS),
    ]
    for y, text, col in ph1:
        _box(ax, (X, y), W if col != C_TERMINAL else 0.30, H, text, col)
    for i in range(len(ph1)-1):
        _arrow(ax, (X, ph1[i][0]-H/2), (X, ph1[i+1][0]+H/2))

    _cluster_rect(ax, (X, 0.325), 0.74, 0.32, "Phase 2 — Cell-Centric Analysis", "#1E8449")
    ph2 = [
        (0.44, "Reset coverage dict",                            C_PROCESS),
        (0.37, "CoverageCellCentric object\n(steer_fraction / nadir_pointing / link_budget)", C_PROCESS),
        (0.30, "Precompute Rx boresights\nfor all (Rx, Tx, timestep) pairs", C_PROCESS),
        (0.22, "For each grid cell × (Rx, Tx, timestep):\n"
               "  1. Tx beam gate  (nadir angle ≤ 60°)\n"
               "  2. Rx beam gate  (angle from boresight)\n"
               "  3. Compute SNR if link budget enabled\n"
               "  4. Record detection", C_PROCESS),
    ]
    _arrow(ax, (X, ph1[-1][0]-H/2), (X, ph2[0][0]+H/2))
    for y, text, col in ph2:
        h = H*1.8 if "For each" in text else H
        _box(ax, (X, y), W, h, text, col)
    for i in range(len(ph2)-1):
        h_prev = H*1.8 if "For each" in ph2[i][1] else H
        h_next = H*1.8 if "For each" in ph2[i+1][1] else H
        _arrow(ax, (X, ph2[i][0]-h_prev/2), (X, ph2[i+1][0]+h_next/2))

    # loop-back arrow from cell loop
    ax.annotate("", xy=(0.90, ph2[2][0]),
                xytext=(0.90, ph2[-1][0]),
                arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=1.2))
    _arrow(ax, (0.90, ph2[2][0]), (X+W/2, ph2[2][0]))
    ax.text(0.91, (ph2[2][0]+ph2[-1][0])/2, "next\ncell", fontsize=7.5,
            color="#7D3C98", va="center")

    _cluster_rect(ax, (X, 0.065), 0.74, 0.18, "Phase 3 — Output", "#CA6F1E")
    ph3 = [
        (0.13, "Revisit count plots (PNG)\nSave count & SNR arrays (.npy)", C_IO),
        (0.06, "Coverage statistics", C_PROCESS),
    ]
    h_last = H*1.8
    _arrow(ax, (X, ph2[-1][0]-h_last/2), (X, ph3[0][0]+H/2))
    for y, text, col in ph3:
        _box(ax, (X, y), W, H, text, col)
    _arrow(ax, (X, ph3[0][0]-H/2), (X, ph3[1][0]+H/2))

    fig.suptitle("Area Coverage Cell-Centric Simulation\n(main_simulation_areaCoverage_cellcentric.py)",
                 fontsize=10, fontfamily=FONT, fontweight="bold", y=0.995)
    _save(fig, "area_coverage_cc")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Moving Target Detection
# ═══════════════════════════════════════════════════════════════════════════
def flowchart_moving_target():
    fig, ax = plt.subplots(figsize=(6.5, 18))
    ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1.01); ax.axis("off")
    fig.patch.set_facecolor("white")

    W, H = 0.60, 0.054
    X = 0.50

    nodes = [
        # (y, text, kind)
        (0.97, "START",                                  "term"),
        (0.91, "Load config (Configuration_ship.ini)",   "io"),
        (0.84, "Build ship trajectory\n(A→B, speed, 10-min waypoints)", "proc"),
        (0.77, "Load Tx GNSS constellation",             "proc"),
        (0.70, "Create Rx constellation & RxAntenna",   "proc"),
        (0.63, "Set sim duration = ship travel time",    "proc"),
        (0.56, "MovingTargetDetection object\n(reflection_mode, link_budget,\ncutoff_angle, bistatic_angle)", "proc"),
    ]
    _cluster_rect(ax, (X, 0.76), 0.74, 0.47, "Phase 1 — Setup", "#2E86C1")

    for y, text, kind in nodes:
        col = {"term": C_TERMINAL, "io": C_IO, "proc": C_PROCESS}[kind]
        _box(ax, (X, y), W if kind != "term" else 0.30, H, text, col)
    for i in range(len(nodes)-1):
        _arrow(ax, (X, nodes[i][0]-H/2), (X, nodes[i+1][0]+H/2))

    _cluster_rect(ax, (X, 0.415), 0.74, 0.18, "Phase 2 — Optimised Propagation", "#1E8449")
    ph2 = [
        (0.49, "Find Rx visible time windows\n& resample ship positions", "proc"),
        (0.41, "Propagate only visible satellites (SGP4)", "proc"),
    ]
    _arrow(ax, (X, nodes[-1][0]-H/2), (X, ph2[0][0]+H/2))
    for y, text, kind in ph2:
        _box(ax, (X, y), W, H, text, C_PROCESS)
    _arrow(ax, (X, ph2[0][0]-H/2), (X, ph2[1][0]+H/2))

    _cluster_rect(ax, (X, 0.245), 0.74, 0.18, "Phase 3 — Cell-Centric Detection", "#7D3C98")
    ph3 = [
        (0.32, "MovingTargetCellCentric\n(steer_fraction=0.2, max_bistatic_angle)", "proc"),
        (0.24, "calculate():\nTx gate → Rx gate → bistatic gate → detect", "proc"),
        (0.17, "Detection map + timeline plots (PNG)", "io"),
    ]
    _arrow(ax, (X, ph2[-1][0]-H/2), (X, ph3[0][0]+H/2))
    for y, text, kind in ph3:
        col = C_IO if kind == "io" else C_PROCESS
        _box(ax, (X, y), W, H, text, col)
    for i in range(len(ph3)-1):
        _arrow(ax, (X, ph3[i][0]-H/2), (X, ph3[i+1][0]+H/2))

    _cluster_rect(ax, (X, 0.04), 0.74, 0.17,
                  "Phase 4 — Reflection-Point Intersection (traditional)", "#CA6F1E")
    ph4 = [
        (0.10, "Calculate reflection point\n& ship-footprint intersection", "proc"),
        (0.04, "Track + detection points plot (PNG)\n[Optional: animation MP4]", "io"),
    ]
    _arrow(ax, (X, ph3[-1][0]-H/2), (X, ph4[0][0]+H/2))
    for y, text, kind in ph4:
        col = C_IO if kind == "io" else C_PROCESS
        _box(ax, (X, y), W, H, text, col)
    _arrow(ax, (X, ph4[0][0]-H/2), (X, ph4[1][0]+H/2))

    fig.suptitle("Moving Target Detection Simulation\n(main_simulation_movingTarget.py)",
                 fontsize=10, fontfamily=FONT, fontweight="bold", y=1.0)
    _save(fig, "moving_target")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Beta Sweep
# ═══════════════════════════════════════════════════════════════════════════
def flowchart_beta_sweep():
    fig, ax = plt.subplots(figsize=(6.5, 14))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    W, H = 0.60, 0.057
    X = 0.50

    _box(ax, (X, 0.96), 0.30, H, "START", C_TERMINAL)
    _box(ax, (X, 0.89), W, H, "Configs: 1sat / 2sat / 4sat / 8sat", C_IO)
    _box(ax, (X, 0.82), W, H, "Beta values: None, 5°, 10°…45°, 60°", C_IO)
    _arrow(ax, (X, 0.96-H/2), (X, 0.89+H/2))
    _arrow(ax, (X, 0.89-H/2), (X, 0.82+H/2))

    # outer loop
    _cluster_rect(ax, (X, 0.615), 0.74, 0.35,
                  "For each constellation config (1sat → 8sat)", "#2E86C1")
    outer = [
        (0.73, "Load config & build ship / GNSS / Rx objects", "proc"),
        (0.66, "Propagate orbits (once per constellation)",     "proc"),
    ]
    _arrow(ax, (X, 0.82-H/2), (X, outer[0][0]+H/2))
    for y, text, _ in outer:
        _box(ax, (X, y), W, H, text, C_PROCESS)
    _arrow(ax, (X, outer[0][0]-H/2), (X, outer[1][0]+H/2))

    # inner loop
    _cluster_rect(ax, (X, 0.40), 0.66, 0.27, "For each beta value", "#7D3C98")
    inner = [
        (0.50, "MovingTargetCellCentric\n(max_bistatic_angle=beta)\ncalculate()", "proc"),
        (0.43, "Count total detections",   "proc"),
        (0.36, "Store result[label][beta]","proc"),
    ]
    _arrow(ax, (X, outer[-1][0]-H/2), (X, inner[0][0]+H/2))
    for y, text, _ in inner:
        _box(ax, (X, y), W, H, text, C_PROCESS)
    for i in range(len(inner)-1):
        _arrow(ax, (X, inner[i][0]-H/2), (X, inner[i+1][0]+H/2))

    # loop-back arrows
    ax.annotate("", xy=(0.87, inner[0][0]),
                xytext=(0.87, inner[-1][0]),
                arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=1.2))
    _arrow(ax, (0.87, inner[0][0]), (X+W/2, inner[0][0]))
    ax.text(0.89, (inner[0][0]+inner[-1][0])/2, "next\nbeta", fontsize=7,
            color="#7D3C98", va="center")

    ax.annotate("", xy=(0.93, outer[0][0]),
                xytext=(0.93, inner[-1][0]),
                arrowprops=dict(arrowstyle="-|>", color="#2E86C1", lw=1.2))
    _arrow(ax, (0.93, outer[0][0]), (X+W/2, outer[0][0]))
    ax.text(0.95, (outer[0][0]+inner[-1][0])/2, "next\nconstellation",
            fontsize=7, color="#2E86C1", va="center")

    # output
    out = [
        (0.24, "Plot: detections vs β\n(OUTPUT/Beta_sweep.png)", "io"),
        (0.16, "Print & save summary table (txt)", "io"),
        (0.08, "END", "term"),
    ]
    _arrow(ax, (X, inner[-1][0]-H/2), (X, out[0][0]+H/2))
    for y, text, kind in out:
        col = {"io": C_IO, "term": C_TERMINAL}[kind]
        w = 0.30 if kind == "term" else W
        _box(ax, (X, y), w, H, text, col)
    for i in range(len(out)-1):
        _arrow(ax, (X, out[i][0]-H/2), (X, out[i+1][0]+H/2))

    fig.suptitle("Beta Sweep (bistatic angle sensitivity)\n(run_beta_sweep.py)",
                 fontsize=10, fontfamily=FONT, fontweight="bold", y=0.995)
    _save(fig, "beta_sweep")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Geometric Coverage with Tracking
# ═══════════════════════════════════════════════════════════════════════════
def flowchart_geometric_tracking():
    fig, ax = plt.subplots(figsize=(6.0, 16))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    W, H = 0.60, 0.057
    X = 0.50

    main_flow = [
        (0.96, "START",                                           "term"),
        (0.89, "Load config\n(Configuration_geometric_1tx1rx.ini)", "io"),
        (0.82, "Load Tx GNSS constellation",                      "proc"),
        (0.75, "Create Rx constellation & RxAntenna",             "proc"),
        (0.68, "Propagate all orbits",                            "proc"),
        (0.61, "Coverage object\n(shapefile + 30 km grid)",       "proc"),
        (0.54, "GeometricCoverage object\n(min_tx_elev, max_bistatic_angle)", "proc"),
        (0.47, "calculate_geometric_coverage()\nfor each timestep × Rx × Tx × cell:\n  elevation gate → bistatic gate", "proc"),
        (0.39, "Extract reflection counts\nper cell & per timestep", "proc"),
    ]

    for y, text, kind in main_flow:
        col = {"term": C_TERMINAL, "io": C_IO, "proc": C_PROCESS}[kind]
        _box(ax, (X, y), W if kind != "term" else 0.30, H, text, col)
    for i in range(len(main_flow)-1):
        _arrow(ax, (X, main_flow[i][0]-H/2), (X, main_flow[i+1][0]+H/2))

    # tracking cluster
    _cluster_rect(ax, (X, 0.23), 0.72, 0.20,
                  "Reflection Tracking (100 sampled timesteps)", "#7D3C98")
    _arrow(ax, (X, main_flow[-1][0]-H/2), (X, 0.31+H/2))
    _box(ax, (X, 0.31), W, H, "Scan timesteps (every ~1% of sim)", C_PROCESS)
    _diamond(ax, (X, 0.24), W*0.75, H*0.85,
             "Rx or Tx in\ntarget region?", C_DECISION)
    _box(ax, (X, 0.17), W, H, "Log: time, positions, reflection count", C_PROCESS)

    _arrow(ax, (X, 0.31-H/2), (X, 0.24+H*0.43))
    _arrow(ax, (X, 0.24-H*0.43), (X, 0.17+H/2), "yes")
    # no → back to scan
    ax.annotate("", xy=(0.84, 0.31),
                xytext=(0.84, 0.24),
                arrowprops=dict(arrowstyle="-|>", color="#CA6F1E", lw=1.1))
    _arrow(ax, (0.84, 0.31), (X+W/2, 0.31))
    ax.text(0.86, (0.31+0.24)/2, "no /\nnext", fontsize=7,
            color="#CA6F1E", va="center")

    _arrow(ax, (X, 0.17-H/2), (X, 0.09+H/2))

    out = [
        (0.09, "Print coverage & reflection statistics", "proc"),
        (0.02, "Save results (.pkl)", "io"),
    ]
    for y, text, kind in out:
        col = C_IO if kind == "io" else C_PROCESS
        _box(ax, (X, y), W, H, text, col)
    _arrow(ax, (X, out[0][0]-H/2), (X, out[1][0]+H/2))

    fig.suptitle("Geometric Coverage with Reflection Tracking\n(run_simulation_with_tracking.py)",
                 fontsize=10, fontfamily=FONT, fontweight="bold", y=0.995)
    _save(fig, "geometric_tracking")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    flowchart_area_coverage()
    flowchart_area_coverage_cc()
    flowchart_moving_target()
    flowchart_beta_sweep()
    flowchart_geometric_tracking()
    print("\nAll flowcharts saved to OUTPUT/")
