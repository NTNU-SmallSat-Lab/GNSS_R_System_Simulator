"""Generate the two-plane steering-angle schematic for antenna.html.

Two panels (angle subscript = rotation axis):
  (a) elevation principal plane (x-z): beam tilted from boresight z^a by alpha_y (rotation about y; the
      peak moves along x^a, the flight direction).
  (b) azimuth principal plane (y-z): beam tilted from boresight z^a by alpha_x (rotation about x).

In each principal-plane cut the steering angle is measured directly from boresight z^a; a general command
applies both at once, giving beam direction (sin alpha_y, sin alpha_x, sqrt(1 - sin^2 - sin^2)). Boresight
points down to match the nadir-pointing orientation of the array-geometry figure. Output PNG ->
docs/source/images/.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: just write the PNG

from matplotlib.patches import Arc, FancyArrowPatch, Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# palette (consistent with the other docs schematics)
TOP_FILL = "#eaf2fb"     # aperture slab
ELEM_FILL = "#cfe0f2"    # array elements
EDGE = "#5b7fa6"         # outlines
AXIS = "#e08a1e"         # antenna-frame axes (orange)
BEAM = "#214f7b"         # steered beam
ANG = "#16212c"          # angle arcs / labels
TEXT = "#16212c"

ALPHA_X = 35.0    # example azimuth steering angle (rotation about x) [deg]
ALPHA_Y = 22.0    # example elevation steering angle (rotation about y) [deg]


def panel_cut(ax, axis_label, alpha_deg, angle_label, title):
    """One principal-plane side cut: boresight z^a down, transverse axis right, beam alpha_deg off z^a."""
    a = np.radians(alpha_deg)
    L = 1.0

    # array seen edge-on, with a few elements on the boresight-facing (lower) side
    ax.add_patch(Rectangle((-0.5, -0.035), 1.0, 0.07, facecolor=TOP_FILL, edgecolor=EDGE, lw=1.3, zorder=2))
    for xc in np.linspace(-0.4, 0.4, 5):
        ax.add_patch(Rectangle((xc - 0.05, -0.115), 0.1, 0.07, facecolor=ELEM_FILL, edgecolor=EDGE, lw=0.8, zorder=3))

    # antenna-frame axes: z^a down (boresight), transverse axis to the right
    ax.add_patch(FancyArrowPatch((0, 0), (0, -L * 1.18), arrowstyle="-|>", mutation_scale=15, color=AXIS, lw=2.2, zorder=6))
    ax.text(0.04, -L * 1.22, r"$z^{\mathrm{a}}$ (boresight)", color=AXIS, ha="left", va="top", fontsize=12)
    ax.add_patch(FancyArrowPatch((0, 0), (L * 1.02, 0), arrowstyle="-|>", mutation_scale=14, color=AXIS, lw=2.0, zorder=6))
    ax.text(L * 1.05, 0.05, axis_label, color=AXIS, ha="left", va="bottom", fontsize=14)

    # steered beam: tilted from the downward boresight toward the transverse axis by alpha
    bx, bz = np.sin(a), -np.cos(a)
    ax.add_patch(FancyArrowPatch((0, 0), (L * bx, L * bz), arrowstyle="-|>", mutation_scale=18, color=BEAM, lw=2.7, zorder=7))
    ax.text(L * bx + 0.06, L * bz - 0.02, "beam", color=BEAM, ha="left", va="center", fontsize=12)

    # angle arc, measured from boresight z^a (down, -90 deg) to the beam
    r = 0.55
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, angle=0.0, theta1=-90.0, theta2=-90.0 + alpha_deg, color=ANG, lw=1.7, zorder=8))
    am = np.radians(-90.0 + 0.5 * alpha_deg)
    ax.text(0.80 * np.cos(am), 0.80 * np.sin(am), angle_label, color=ANG, fontsize=15, ha="center", va="center", zorder=9)

    ax.set_xlim(-0.75, 1.3)
    ax.set_ylim(-1.4, 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=TEXT)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.6))
    panel_cut(ax1, r"$x^{\mathrm{a}}$", ALPHA_Y, r"$\alpha_{y}$", r"(a) elevation plane ($x$-$z$)")
    panel_cut(ax2, r"$y^{\mathrm{a}}$", ALPHA_X, r"$\alpha_{x}$", r"(b) azimuth plane ($y$-$z$)")
    fig.tight_layout(pad=0.6)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "source", "images", "steering_geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
