"""Generate the phased-array antenna geometry schematic for antenna.html.

Axonometric technical drawing of the planar receiver array: the N_x x N_y element grid with
spacings d_x, d_y, the antenna coordinate frame (x^a, y^a in the aperture plane, z^a along
boresight), and the steered beam at polar angle theta off boresight and azimuth phi. Inspired
by the user's reference sketch, redrawn cleanly. Output PNG -> docs/source/images/.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: just write the PNG

from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# palette (consistent with the other docs schematics)
TOP_FILL = "#eaf2fb"     # panel top face
ELEM_FILL = "#cfe0f2"    # array elements
ELEM_EDGE = "#5b7fa6"
SIDE_FILL = "#c7d2de"    # panel thickness (side faces)
REF_FILL = "#a6c4e2"     # reference element (m, n) = (0, 0)
IDX = "#34495e"          # element-index labels
AXIS = "#e08a1e"         # antenna-frame axes (orange, after the reference)
BEAM = "#214f7b"         # steered beam
ANG = "#16212c"          # angle arcs / labels
DASH = "#8190a0"         # azimuth reference (dotted)
DIM = "#5a6b7a"          # dimension arrows (d_x, d_y)
TEXT = "#16212c"

# axonometric projection: world (x, y, z) -> screen (X, Y)
UX = np.array([0.95, 0.16])   # +x: right, slightly up
UY = np.array([-0.52, 0.34])  # +y: into the page (upper-left)
UZ = np.array([0.0, -1.0])    # +z: boresight, straight down (nadir-pointing receiver)

NX, NY = 4, 4   # elements along x, y
PITCH = 1.0     # element spacing (d_x = d_y here)
THICK = 0.18    # panel thickness
THETA = np.radians(34.0)   # beam polar angle off boresight
PHI = np.radians(40.0)     # beam azimuth in the aperture plane
CENTRE = np.array([NX * PITCH / 2.0, NY * PITCH / 2.0, 0.0])  # antenna-frame origin


def proj(x, y, z):
    """Project world coordinates (scalars or arrays) to screen X, Y."""
    return x * UX[0] + y * UY[0] + z * UZ[0], x * UX[1] + y * UY[1] + z * UZ[1]


def face(ax, pts3, facecolor, edgecolor, lw=1.2, z=1, alpha=1.0):
    """Draw a filled, projected polygon from a list of 3-D corners."""
    xy = [proj(p[0], p[1], p[2]) for p in pts3]
    ax.add_patch(Polygon(xy, closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=z, alpha=alpha))


def arrow(ax, p0, p1, color, lw=2.4, scale=20, z=10):
    """Draw a projected arrow from 3-D point p0 to p1."""
    a, b = proj(*p0), proj(*p1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def arc(ax, radius, vec, s0, s1, color, label, lab_r, lw=1.7, z=9, n=48):
    """Draw a 3-D angle arc of given radius about CENTRE, swept by direction vec(s)."""
    s = np.linspace(s0, s1, n)
    pts = np.array([CENTRE + radius * vec(si) for si in s])
    ax.plot(*proj(pts[:, 0], pts[:, 1], pts[:, 2]), color=color, lw=lw, zorder=z)
    mid = CENTRE + lab_r * vec(0.5 * (s0 + s1))
    ax.text(*proj(mid[0], mid[1], mid[2]), label, color=color, fontsize=15, ha="center", va="center", zorder=12)


def dim(ax, p0, p1, label, lab_off, color=DIM):
    """Double-headed dimension arrow between two 3-D points, with a label."""
    a, b = proj(*p0), proj(*p1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=10, color=color, lw=1.3, shrinkA=0, shrinkB=0, zorder=8))
    mid = 0.5 * (np.array(p0) + np.array(p1))
    ax.text(*proj(mid[0] + lab_off[0], mid[1] + lab_off[1], mid[2] + lab_off[2]), label, color=color, fontsize=12, ha="center", va="center")


def right_angle(ax, u, v, e=0.36, color=ANG, lw=1.3, z=9):
    """Small square corner marking a right angle between in-plane direction u and z^a (v) at CENTRE."""
    u, v = np.array(u, float), np.array(v, float)
    pts = np.array([CENTRE + e * u, CENTRE + e * (u + v), CENTRE + e * v])
    ax.plot(*proj(pts[:, 0], pts[:, 1], pts[:, 2]), color=color, lw=lw, zorder=z)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 6.4))

    lx, ly = NX * PITCH, NY * PITCH

    # panel thickness: with boresight z^a pointing down, the visible side faces are the far edges (back y=ly, right x=lx)
    face(ax, [(0, ly, 0), (lx, ly, 0), (lx, ly, -THICK), (0, ly, -THICK)], SIDE_FILL, ELEM_EDGE, z=1)
    face(ax, [(lx, 0, 0), (lx, ly, 0), (lx, ly, -THICK), (lx, 0, -THICK)], SIDE_FILL, ELEM_EDGE, z=1)

    # panel top face and the element grid
    face(ax, [(0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0)], TOP_FILL, ELEM_EDGE, lw=1.4, z=2)
    g = 0.16  # gap between elements
    for i in range(NX):
        for j in range(NY):
            fill = REF_FILL if (i == 0 and j == 0) else ELEM_FILL  # highlight the (0,0) reference element
            face(ax, [(i + g, j + g, 0), (i + 1 - g, j + g, 0), (i + 1 - g, j + 1 - g, 0), (i + g, j + 1 - g, 0)],
                 fill, ELEM_EDGE, lw=1.0, z=3)

    # element-spacing dimensions d_x, d_y (just outside the panel edges)
    dim(ax, (1.0, -0.55, 0), (2.0, -0.55, 0), r"$d_x$", (0.0, -0.32, 0.0))
    dim(ax, (-0.55, 1.0, 0), (-0.55, 2.0, 0), r"$d_y$", (-0.45, 0.0, 0.0))

    # element indices: m = 0..N_x-1 along x^a, n = 0..N_y-1 along y^a (0-based, matching the array-factor sum)
    for i in range(NX):
        ax.text(*proj(i + 0.5, -0.30, 0), str(i), color=IDX, fontsize=10.5, ha="center", va="center", zorder=12)
    for j in range(NY):
        ax.text(*proj(-0.30, j + 0.5, 0), str(j), color=IDX, fontsize=10.5, ha="center", va="center", zorder=12)
    ax.text(*proj(NX + 0.02, -0.30, 0), r"$m$", color=IDX, fontsize=13, style="italic", ha="center", va="center", zorder=12)
    ax.text(*proj(-0.30, NY + 0.02, 0), r"$n$", color=IDX, fontsize=13, style="italic", ha="center", va="center", zorder=12)
    ax.text(*proj(0.5, 0.5, 0.03), r"$(0,0)$", color=IDX, fontsize=8.5, ha="center", va="center", zorder=12)

    # azimuth reference: beam projected onto the aperture plane (dotted)
    az_end = CENTRE + 2.7 * np.array([np.cos(PHI), np.sin(PHI), 0.0])
    ax.plot(*proj(*np.array([CENTRE, az_end]).T), color=DASH, lw=1.4, ls=":", zorder=6)

    # angle arcs: theta (boresight -> beam) and phi (x^a -> azimuth reference)
    arc(ax, 1.15, lambda s: np.array([np.sin(s) * np.cos(PHI), np.sin(s) * np.sin(PHI), np.cos(s)]),
        0.0, THETA, ANG, r"$\theta$", 1.45)
    arc(ax, 1.5, lambda s: np.array([np.cos(s), np.sin(s), 0.0]), 0.0, PHI, ANG, r"$\phi$", 1.78)

    # right-angle markers: z^a is perpendicular to the aperture plane (square corner on both in-plane axes)
    right_angle(ax, [1, 0, 0], [0, 0, 1])
    right_angle(ax, [0, 1, 0], [0, 0, 1])

    # antenna-frame axes from the panel centre
    la = 2.7
    arrow(ax, CENTRE, CENTRE + [la, 0, 0], AXIS)
    arrow(ax, CENTRE, CENTRE + [0, la, 0], AXIS)
    arrow(ax, CENTRE, CENTRE + [0, 0, la], AXIS)
    ax.plot(*proj(*CENTRE), "o", color=AXIS, markersize=5, zorder=11)

    # steered beam
    beam_dir = np.array([np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)])
    arrow(ax, CENTRE, CENTRE + 3.1 * beam_dir, BEAM, lw=2.8, scale=22, z=11)

    # labels
    ax.text(*proj(*(CENTRE + [la + 0.28, 0, 0])), r"$x^{\mathrm{a}}$", color=AXIS, fontsize=16, ha="center", va="center")
    ax.text(*proj(*(CENTRE + [0, la + 0.30, 0])), r"$y^{\mathrm{a}}$", color=AXIS, fontsize=16, ha="center", va="center")
    ax.text(*proj(*(CENTRE + [0, 0.05, la + 0.30])), r"$z^{\mathrm{a}}$ (boresight)", color=AXIS, fontsize=14, ha="left", va="bottom")
    ax.text(*proj(*(CENTRE + 3.25 * beam_dir + [0.0, 0.0, 0.12])), "steered beam", color=BEAM, fontsize=12.5, ha="center", va="bottom")
    ax.text(*proj(-0.3, ly + 0.9, 0), r"$N_x \times N_y$ element array", color=TEXT, fontsize=12.5, ha="right", va="bottom")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    ax.margins(0.06)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "source", "images", "antenna_geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
