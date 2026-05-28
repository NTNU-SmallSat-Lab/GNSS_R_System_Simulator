"""Generate a static software-architecture diagram (PNG + SVG)."""
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt

COLORS = {
    "entry":      "#1f6feb",
    "mode":       "#2ea043",
    "domain":     "#fb8500",
    "foundation": "#6c757d",
}
BAND = {k: c + "22" for k, c in COLORS.items()}  # translucent band fill

BOX_W, BOX_H = 2.6, 0.85


def draw_box(ax, x, y, label, kind, fontsize=9):
    color = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=color, facecolor=color,
    ))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color="white", weight="bold")
    return (x, y)


def edge(ax, src_xy, dst_xy, style="solid", color="#333", rad=0.0):
    if style == "thick":
        kw = {"linewidth": 2.4, "color": color, "linestyle": "-"}
    elif style == "dashed":
        kw = {"linewidth": 1.0, "color": color, "linestyle": (0, (4, 3))}
    else:
        kw = {"linewidth": 1.4, "color": color, "linestyle": "-"}
    src = _box_edge(src_xy, dst_xy, "out")
    dst = _box_edge(dst_xy, src_xy, "in")
    ax.add_patch(FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=14,
        connectionstyle=f"arc3,rad={rad}",
        **kw,
    ))


def _box_edge(box_xy, other_xy, _which):
    """Find the box-edge point on the line from box_xy toward other_xy."""
    bx, by = box_xy
    ox, oy = other_xy
    dx, dy = ox - bx, oy - by
    if dx == 0 and dy == 0:
        return box_xy
    half_w, half_h = BOX_W / 2, BOX_H / 2
    if abs(dx) * half_h > abs(dy) * half_w:
        sx = half_w if dx > 0 else -half_w
        sy = dy * (sx / dx) if dx else 0
    else:
        sy = half_h if dy > 0 else -half_h
        sx = dx * (sy / dy) if dy else 0
    return (bx + sx, by + sy)


fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11.5)
ax.set_aspect("equal")
ax.axis("off")

Y_ENTRY = 10.2
Y_MODE = 8.2
Y_DOM_T = 6.1
Y_DOM_B = 4.6
Y_FOUND = 1.6

# Layer bands
def band(y_center, height, kind, label):
    ax.add_patch(Rectangle((0.2, y_center - height / 2), 17.6, height,
                           facecolor=BAND[kind], edgecolor="none", zorder=0))
    ax.text(0.45, y_center + height / 2 - 0.25, label, ha="left", va="top",
            fontsize=10, color=COLORS[kind], weight="bold", style="italic")

band(Y_ENTRY, 1.4, "entry", "1.  Entry orchestrators")
band(Y_MODE, 1.2, "mode", "2.  Mode-specific analysis")
band((Y_DOM_T + Y_DOM_B) / 2, 2.6, "domain", "3.  Shared domain model")
band(Y_FOUND, 1.4, "foundation", "4.  Foundation (imported broadly)")

# 1. Entry
init = draw_box(ax, 4.5, Y_ENTRY, "initialize\n(parses INI)", "entry")
ac   = draw_box(ax, 9.5, Y_ENTRY, "main_simulation_\nareaCoverage", "entry")
mt   = draw_box(ax, 14.5, Y_ENTRY, "main_simulation_\nmovingTarget", "entry")

# 2. Mode
cov = draw_box(ax, 9.5, Y_MODE, "Coverage", "mode")
mtd = draw_box(ax, 14.5, Y_MODE, "MovingTargetDetection", "mode")

# 3. Domain (top row: structural; bottom row: geometry and antenna)
sc   = draw_box(ax, 3.0, Y_DOM_T, "SatelliteConstellation", "domain")
sat  = draw_box(ax, 6.5, Y_DOM_T, "Satellite", "domain")
orb  = draw_box(ax, 10.0, Y_DOM_T, "Orbit_specification", "domain")
ship = draw_box(ax, 15.0, Y_DOM_T, "ShipTrajectory", "domain")
refl = draw_box(ax, 6.5, Y_DOM_B, "Reflection\n(specular)", "domain")
brp  = draw_box(ax, 10.0, Y_DOM_B, "BRP\n(backscatter)", "domain")
rx   = draw_box(ax, 15.0, Y_DOM_B, "RxAntenna", "domain")

# 4. Foundation
const = draw_box(ax, 2.5,  Y_FOUND, "Constants", "foundation")
fn    = draw_box(ax, 6.0,  Y_FOUND, "functions.functions", "foundation")
lb    = draw_box(ax, 9.5,  Y_FOUND, "functions.link_budget", "foundation")
plot  = draw_box(ax, 12.8, Y_FOUND, "Plotting", "foundation")
rw    = draw_box(ax, 15.8, Y_FOUND, "ReadWriteSatData", "foundation")

# Orchestration (thick)
edge(ax, ac, init, "thick")
edge(ax, mt, init, "thick")
edge(ax, ac, cov, "thick")
edge(ax, mt, mtd, "thick")
edge(ax, ac, sc,  "thick", rad=-0.15)
edge(ax, mt, sc,  "thick", rad=0.25)
edge(ax, mt, ship, "thick", rad=-0.05)
edge(ax, mt, rx,  "thick", rad=-0.20)

# Mode -> domain
edge(ax, cov, refl, "thick")
edge(ax, mtd, refl, "thick", rad=0.15)
edge(ax, mtd, brp,  "thick")
edge(ax, mtd, ship, "thick", rad=-0.10)

# Domain composition (thin solid)
edge(ax, sc, sat)
edge(ax, sat, orb)
edge(ax, refl, sat, rad=-0.10)
edge(ax, brp, sat, rad=0.20)
edge(ax, refl, rx, rad=-0.10)
edge(ax, brp,  rx, rad=-0.10)

# Foundation usage (dashed) -- show representative ones; everything imports Constants implicitly
edge(ax, sat,  rw,  "dashed")
edge(ax, sc,   rw,  "dashed", rad=0.20)
edge(ax, refl, fn,  "dashed", rad=-0.05)
edge(ax, refl, lb,  "dashed", rad=-0.20)
edge(ax, brp,  fn,  "dashed", rad=0.05)
edge(ax, brp,  lb,  "dashed", rad=-0.05)
edge(ax, ship, fn,  "dashed", rad=0.20)
edge(ax, cov,  plot, "dashed", rad=-0.20)

# Title + legend
ax.text(9, 11.15, "GNSS-R System Simulator  -  Software Architecture",
        ha="center", va="center", fontsize=14, weight="bold")

legend_handles = [
    Line2D([0], [0], color="#333", linewidth=2.4, label="orchestration / instantiation"),
    Line2D([0], [0], color="#333", linewidth=1.4, label="composition (has-a)"),
    Line2D([0], [0], color="#333", linewidth=1.0, linestyle=(0, (4, 3)),
           label="utility / foundation use"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True,
          fontsize=9, bbox_to_anchor=(0.985, 0.02))

ax.text(9, 0.35,
        "Note: Constants is imported by nearly every module (omitted from arrows for clarity).",
        ha="center", va="center", fontsize=8.5, color="#555", style="italic")

fig.savefig("architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
fig.savefig("architecture.svg", bbox_inches="tight", facecolor="white")
print("Wrote architecture.png and architecture.svg")
