"""Generate the 2D specular-reflection geometry schematic for the specular mode page.

Side view of the law of reflection: a GNSS transmitter, a single bounce at the specular
reflection point on the sea, and the LEO receiver, with the surface normal and the equal
incidence / reflection angles. Output PNG lands in docs/source/images/.
"""

import os

import matplotlib

matplotlib.use("Agg")

from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

# palette (matches the existing docs SVGs)
RAY = "#3b6ea5"
SEA_FILL = "#dcebf7"
SEA_LINE = "#7fb0d6"
NORMAL = "#5a6b7a"
GLINT = "#f6d68a"
GLINT_EDGE = "#c8862a"
TEXT = "#16212c"
SUB = "#5a6b7a"

THETA_DEG = 40.0


def satellite(ax, x, y, scale=0.32):
    """Draw a small schematic satellite (body plus two solar panels) centred at (x, y)."""
    body = Rectangle((x - 0.5 * scale, y - 0.5 * scale), scale, scale,
                     facecolor="#e8f0fe", edgecolor=RAY, linewidth=1.4, zorder=5)
    ax.add_patch(body)
    for sign in (-1.0, 1.0):
        panel = Rectangle((x + sign * 1.5 * scale - 0.5 * scale, y - 0.3 * scale), scale, 0.6 * scale,
                          facecolor="#9fb8d4", edgecolor=RAY, linewidth=1.0, zorder=5)
        ax.add_patch(panel)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    s = np.array([5.0, 0.0])               # specular reflection point on the surface
    theta = np.radians(THETA_DEG)

    dir_tx = np.array([-np.sin(theta), np.cos(theta)])   # S -> transmitter (up-left)
    dir_rx = np.array([np.sin(theta), np.cos(theta)])    # S -> receiver (up-right)
    tx = s + 5.0 * dir_tx
    rx = s + 4.0 * dir_rx

    # sea surface
    x = np.linspace(0.2, 9.8, 400)
    wave = 0.06 * np.sin((x - 0.2) * 4.2)
    ax.fill_between(x, wave, -1.2, color=SEA_FILL, zorder=0)
    ax.plot(x, wave, color=SEA_LINE, linewidth=1.6, zorder=1)
    ax.text(0.4, -0.32, "Sea surface", color=SUB, fontsize=10, style="italic")

    # glistening zone
    ax.add_patch(Ellipse(s, width=3.0, height=0.42, facecolor=GLINT, edgecolor=GLINT_EDGE,
                         alpha=0.65, linewidth=1.2, zorder=2))
    ax.text(s[0], -0.55, "Glistening zone", color=SUB, fontsize=10, style="italic",
            ha="center", va="top")

    # surface normal
    ax.plot([s[0], s[0]], [0.0, 3.2], color=NORMAL, linewidth=1.4, linestyle=(0, (5, 4)), zorder=3)
    ax.text(s[0] + 0.12, 3.18, "surface normal", color=SUB, fontsize=10, style="italic", va="top")

    # incident and reflected rays (arrowheads at S and at the receiver)
    ax.add_patch(FancyArrowPatch(tx, s, arrowstyle="-|>", mutation_scale=16,
                                 color=RAY, linewidth=2.4, zorder=4, shrinkA=10, shrinkB=2))
    ax.add_patch(FancyArrowPatch(s, rx, arrowstyle="-|>", mutation_scale=16,
                                 color=RAY, linewidth=2.4, zorder=4, shrinkA=2, shrinkB=10))

    # equal-angle arcs about the normal
    ang_tx = np.degrees(np.arctan2(dir_tx[1], dir_tx[0]))   # 90 + theta
    ang_rx = np.degrees(np.arctan2(dir_rx[1], dir_rx[0]))   # 90 - theta
    ax.add_patch(Arc(s, 2.0, 2.0, angle=0.0, theta1=90.0, theta2=ang_tx,
                     color=GLINT_EDGE, linewidth=1.6, zorder=4))
    ax.add_patch(Arc(s, 2.0, 2.0, angle=0.0, theta1=ang_rx, theta2=90.0,
                     color=GLINT_EDGE, linewidth=1.6, zorder=4))
    mid_tx = np.radians(90.0 + 0.5 * THETA_DEG)
    mid_rx = np.radians(90.0 - 0.5 * THETA_DEG)
    ax.text(s[0] + 1.35 * np.cos(mid_tx), 1.35 * np.sin(mid_tx), r"$\theta_{in}$",
            color=GLINT_EDGE, fontsize=13, ha="center", va="center")
    ax.text(s[0] + 1.35 * np.cos(mid_rx), 1.35 * np.sin(mid_rx), r"$\theta_{out}$",
            color=GLINT_EDGE, fontsize=13, ha="center", va="center")
    ax.text(s[0], 2.05, r"$\theta_{in} = \theta_{out}$", color=TEXT, fontsize=12,
            ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    # satellites and the specular point
    satellite(ax, tx[0], tx[1])
    satellite(ax, rx[0], rx[1])
    ax.text(tx[0], tx[1] + 0.5, "GNSS transmitter", color=TEXT, fontsize=11, ha="center")
    ax.text(rx[0] + 0.1, rx[1] + 0.5, "LEO receiver", color=TEXT, fontsize=11, ha="center")
    ax.plot(s[0], s[1], "o", color=TEXT, markersize=6, zorder=6)
    ax.annotate("Specular reflection point", xy=(s[0] - 0.05, s[1]), xytext=(2.2, -1.0),
                color=TEXT, fontsize=11, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": TEXT, "linewidth": 0.9})

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-1.2, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.4)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "images")
    out_path = os.path.abspath(os.path.join(out_dir, "specular_geometry.png"))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
