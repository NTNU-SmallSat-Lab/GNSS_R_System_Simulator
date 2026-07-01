"""Generate a UML class diagram (drawio) of the GNSS-R simulator.

Output: architecture_uml.drawio

Open in:
  - https://app.diagrams.net   (File -> Open from device)
  - VS Code with the "Draw.io Integration" extension
  - Drawio Desktop

Drawio can export the result to PDF / PNG / SVG via File -> Export As.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

# ---------------------------------------------------------------------------
#  Drawio style strings - matches drawio's built-in "Class 2" template
# ---------------------------------------------------------------------------
TITLE_SIZE = 26          # title bar height in the swimlane
ROW_H = 26               # height of one attribute / method row
SEP_H = 8                # height of the horizontal separator between attrs & methods
BOX_W = 240              # default class box width

CLASS_STYLE = (
    "swimlane;fontStyle=2;childLayout=stackLayout;horizontal=1;"
    f"startSize={TITLE_SIZE};fillColor=none;horizontalStack=0;"
    "resizeParent=1;resizeParentMax=0;horizontalStackOrder=desc;"
    "collapsible=0;marginBottom=0;swimlaneFillColor=#ffffff;"
)

ROW_STYLE = (
    "text;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    "spacingLeft=4;spacingRight=4;overflow=hidden;rotatable=0;"
    "points=[[0,0.5],[1,0.5]];portConstraint=eastwest;fontSize=12;"
)

SEP_STYLE = (
    "line;strokeWidth=1;fillColor=none;align=left;verticalAlign=middle;"
    "spacingTop=-1;spacingLeft=3;spacingRight=3;rotatable=0;"
    "labelPosition=right;points=[[0,0.5],[1,0.5]];portConstraint=eastwest;"
)

# Edge styles (straight lines, no orthogonal routing)
ASSOC = "endArrow=none;html=1;rounded=0;"
DEPEND = "endArrow=open;html=1;dashed=1;endSize=10;rounded=0;"
INHERIT = "endArrow=block;endFill=0;html=1;endSize=14;rounded=0;"
COMPOSITION = "endArrow=none;html=1;startArrow=diamondThin;startFill=1;startSize=14;rounded=0;"
AGGREGATION = "endArrow=none;html=1;startArrow=diamondThin;startFill=0;startSize=14;rounded=0;"


# ---------------------------------------------------------------------------
#  Class data
# ---------------------------------------------------------------------------
@dataclass
class UClass:
    id: str
    name: str
    x: int
    y: int
    attrs: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    width: int = BOX_W


# Layout - similar spread to the example, expanded to fit all classes.
CLASSES: list[UClass] = [
    # ---- Top band: ShipTrajectory and the analyser classes
    UClass(
        id="ShipTraj", name="ShipTrajectory", x=80, y=40,
        attrs=[
            "ship_id",
            "ship_speed",
            "start_position / time",
            "end_position / time",
            "interval_minutes",
            "grid_resolution",
            "waypoints",
            "trajectory_points",
            "shapefile_path",
        ],
        methods=[
            "plan_route()",
            "generate_trajectory_equal_time()",
            "get_position_at_time(t)",
            "get_travel_duration_hours()",
        ],
    ),
    UClass(
        id="MTD", name="MovingTargetDetection", x=440, y=40,
        attrs=[
            "moving_target_object",
            "sat_constellation",
            "gnss_constellation",
            "sat_antenna",
            "reflection_dict_timeSorted",
            "target_detect_dict",
            "earth_model",
            "reflection_mode",
            "deltaT_sec",
        ],
        methods=[
            "calculate_moving_target_reflection_intersection()",
            "show_track_and_detection_points()",
            "create_reflection_animation()",
        ],
    ),
    UClass(
        id="Coverage", name="Coverage", x=820, y=40,
        attrs=[
            "sat_constellation",
            "gnss_constellation",
            "shp_input_file_path",
            "gridded_shapefile_path",
            "grid_type",
            "grid_size_deg / grid_size_m",
            "bounding_box",
            "reflection_dict",
            "coverage_dict",
            "earth_model",
        ],
        methods=[
            "calculate_reflection_points_and_coverage()",
            "calculate_statistics()",
            "plot_coverage_shapefile()",
            "plot_coverage_gridded_shapefile()",
        ],
    ),
    UClass(
        id="Orbit", name="Orbit", x=1180, y=40,
        attrs=[
            "i, e, a, RAAN",
            "p, M, n",
            "epoch_str",
            "dt_start",
        ],
        methods=[
            "set_start_time(dt)",
            "conv_DateTime_to_Epoch()",
            "generate_satID()",
        ],
    ),
    UClass(
        id="Skyfield", name="Skyfield satellite", x=1480, y=40,
        attrs=[
            "TLE-derived state",
            "skyfield.EarthSatellite",
        ],
        methods=[
            "at(time)",
            "is_sunlit(planets)",
        ],
    ),
    # ---- Mid band: Constants, RxAntenna, ReadWriteSatData
    UClass(
        id="Constants", name="Constants", x=1780, y=320,
        attrs=[
            "Physical constants",
            "GNSS band frequencies",
            "File paths",
            "Coordinate-system tools",
        ],
        methods=[],
    ),
    UClass(
        id="RxAntenna", name="GNSS_R_antenna", x=80, y=420,
        attrs=[
            "antenna_lookuptable",
            "steering_across_track / along_track",
            "frequencies, bands",
            "polarizations",
            "footprint_dB_contour",
            "R_antenna_to_body",
        ],
        methods=[
            "get_gain_dbi_for_direction()",
        ],
    ),
    UClass(
        id="LinkBudget", name="BistaticLinkBudgetSettings", x=440, y=420,
        attrs=[
            "tx_eirp_dbw",
            "system_losses_db",
            "polarization_losses_db",
            "ocean_sigma0_db",
            "ocean_sigma_slope_db_per_deg",
            "ocean_sigma_floor_db",
            "fresnel_rcs_scale",
        ],
        methods=[],
    ),
    UClass(
        id="RWSat", name="ReadWriteData", x=820, y=420,
        attrs=[
            "(static utility)",
        ],
        methods=[
            "generate_tle_data_string(sat)",
            "read_tle_data(path)",
            "download_and_read_tle(path)",
        ],
    ),
    UClass(
        id="Plotting", name="Plotting", x=1180, y=420,
        attrs=[
            "(static utility)",
        ],
        methods=[
            "plot_reflection_traces_gnss_const()",
            "plot_reflection_traces_total()",
            "plot_orbits(sat_const)",
        ],
    ),

    # ---- Lower band: SatelliteConstellation and Reflection-family
    UClass(
        id="SatConst", name="SatelliteConstellation", x=80, y=760,
        attrs=[
            "satConfigName",
            "satellites",
            "no_satellites",
            "orbits_array",
            "times_skyfield",
            "duration_sec",
            "propagation_step_size",
        ],
        methods=[
            "set_simulation_duration()",
            "propagate_constellation()",
            "add_satellite() / remove_satellite()",
        ],
    ),
    UClass(
        id="Reflection", name="Reflection", x=1180, y=760,
        attrs=[
            "tx_satID,  rx_satID",
            "earth_model",
            "max_central_angle_deg",
            "gnss_const",
            "time_vec",
            "refl_ECI_vec / refl_LL_vec",
            "refl_ang_vec",
            "link_budget",
        ],
        methods=[
            "compute_reflection_coordinates()",
            "get_reflection_point_ff()  (spherical, initial guess)",
        ],
    ),
    UClass(
        id="BRP", name="BRP   (backscatter)", x=1480, y=760,
        attrs=[
            "tx_satID,  rx_satID",
            "earth_model",
            "max_central_angle_deg",
            "max_incidence_angle_deg",
            "gnss_const",
            "time_vec",
            "refl_ECI_vec / refl_LL_vec",
            "refl_ang_vec",
            "link_budget",
        ],
        methods=[
            "compute_brp_coordinates()",
        ],
    ),

    # ---- Bottom: Satellite (the centre of the model)
    UClass(
        id="Satellite", name="Satellite", x=600, y=1080,
        attrs=[
            "satID,  satName",
            "classification",
            "orbit_n",
            "skf",
            "tle_file_path",
            "gnss_r_antenna",
            "battery, solarArray",
            "positions_eci, velocities_eci",
            "times_skyfield, time_prop_vec",
            "sunlit_count, sunlit_fraction",
        ],
        methods=[
            "propagate_orbit()",
            "propagate_orbit_for_times()",
            "calculate_sunlight_time()",
            "set_orbit() / set_epoch()",
            "set_timescale_simulation()",
            "write_tle_file()",
        ],
    ),
]


# ---------------------------------------------------------------------------
#  Edges
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    src: str
    tgt: str
    style: str = ASSOC
    label: str = ""
    mult_src: str = ""      # multiplicity at the source end (near src)
    mult_tgt: str = ""      # multiplicity at the target end (near tgt)


EDGES: list[Edge] = [
    # --- Satellite is composed of / aggregates these
    Edge("Satellite", "Orbit",      ASSOC, "lives at", "1", "0..1"),
    Edge("Satellite", "Skyfield",   ASSOC, "lives at", "1", "0..1"),
    Edge("Satellite", "RxAntenna",  ASSOC, "lives at", "1", "0..1"),

    # --- SatelliteConstellation contains Satellites
    Edge("SatConst",  "Satellite",  COMPOSITION, "contains",  "1", "1..*"),

    # --- Reflection / BRP need a Tx + Rx Satellite + an antenna + link-budget
    Edge("Reflection", "Satellite",  ASSOC,  "Tx + Rx",  "1", "2"),
    Edge("BRP",        "Satellite",  ASSOC,  "Tx + Rx",  "1", "2"),
    Edge("Reflection", "RxAntenna",  DEPEND, "uses",     "",  ""),
    Edge("BRP",        "RxAntenna",  DEPEND, "uses",     "",  ""),
    Edge("Reflection", "LinkBudget", DEPEND, "uses",     "",  ""),
    Edge("BRP",        "LinkBudget", DEPEND, "uses",     "",  ""),

    # --- Coverage uses LEO Rx + GNSS Tx constellations and stores Reflections
    Edge("Coverage",  "SatConst",   AGGREGATION, "Rx + Tx",  "1", "1..*"),
    Edge("Coverage",  "Reflection", AGGREGATION, "stores",   "1", "*"),
    Edge("Coverage",  "Plotting",   DEPEND, "uses", "", ""),

    # --- MovingTargetDetection orchestrates ship + constellations + geometry
    Edge("MTD", "SatConst",   AGGREGATION, "Rx + Tx",          "1", "1..*"),
    Edge("MTD", "ShipTraj",   AGGREGATION, "target",           "1", "1"),
    Edge("MTD", "RxAntenna",  AGGREGATION, "Rx antenna",       "1", "1"),
    Edge("MTD", "Reflection", AGGREGATION, "specular mode",    "1", "*"),
    Edge("MTD", "BRP",        AGGREGATION, "backscatter mode", "1", "*"),

    # --- Constants is shared everywhere ("lives at") - keep arrow count modest
    Edge("Satellite",  "Constants", DEPEND, "lives at", "", ""),
    Edge("SatConst",   "Constants", DEPEND, "lives at", "", ""),
    Edge("Reflection", "Constants", DEPEND, "lives at", "", ""),
    Edge("BRP",        "Constants", DEPEND, "lives at", "", ""),
    Edge("ShipTraj",   "Constants", DEPEND, "lives at", "", ""),

    # --- ReadWriteData used by Satellite and Constellation for TLE I/O
    Edge("Satellite", "RWSat",  DEPEND, "uses",   "", ""),
    Edge("SatConst",  "RWSat",  DEPEND, "uses",   "", ""),
]


# ---------------------------------------------------------------------------
#  XML emission
# ---------------------------------------------------------------------------
def class_height(c: UClass) -> int:
    rows = max(len(c.attrs), 1) + max(len(c.methods), 1)
    return TITLE_SIZE + ROW_H * rows + SEP_H


def class_xml(c: UClass) -> str:
    """Emit the swimlane + child rows for one class."""
    h = class_height(c)
    parts: list[str] = []

    parts.append(
        f'<mxCell id="{c.id}" value="{escape(c.name)}" style="{CLASS_STYLE}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{c.x}" y="{c.y}" width="{c.width}" height="{h}" as="geometry"/>'
        f'</mxCell>'
    )

    y_off = TITLE_SIZE
    attrs = c.attrs or [" "]
    for i, attr in enumerate(attrs):
        parts.append(
            f'<mxCell id="{c.id}_a{i}" value="{escape(attr)}" '
            f'style="{ROW_STYLE}" vertex="1" parent="{c.id}">'
            f'<mxGeometry y="{y_off}" width="{c.width}" height="{ROW_H}" as="geometry"/>'
            f'</mxCell>'
        )
        y_off += ROW_H

    parts.append(
        f'<mxCell id="{c.id}_sep" value="" style="{SEP_STYLE}" '
        f'vertex="1" parent="{c.id}">'
        f'<mxGeometry y="{y_off}" width="{c.width}" height="{SEP_H}" as="geometry"/>'
        f'</mxCell>'
    )
    y_off += SEP_H

    methods = c.methods or [" "]
    for i, m in enumerate(methods):
        parts.append(
            f'<mxCell id="{c.id}_m{i}" value="{escape(m)}" '
            f'style="{ROW_STYLE}" vertex="1" parent="{c.id}">'
            f'<mxGeometry y="{y_off}" width="{c.width}" height="{ROW_H}" as="geometry"/>'
            f'</mxCell>'
        )
        y_off += ROW_H

    return "\n".join(parts)


def edge_xml(e: Edge, idx: int) -> str:
    eid = f"e{idx}"
    parts = [
        f'<mxCell id="{eid}" style="{e.style}" edge="1" parent="1" '
        f'source="{e.src}" target="{e.tgt}">'
        f'<mxGeometry relative="1" as="geometry"/></mxCell>'
    ]
    if e.label:
        parts.append(
            f'<mxCell id="{eid}_l" value="{escape(e.label)}" '
            f'style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;'
            f'points=[];fontSize=11;" vertex="1" connectable="0" parent="{eid}">'
            f'<mxGeometry x="0" relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry>'
            f'</mxCell>'
        )
    if e.mult_src:
        parts.append(
            f'<mxCell id="{eid}_ms" value="{escape(e.mult_src)}" '
            f'style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;'
            f'points=[];fontSize=11;" vertex="1" connectable="0" parent="{eid}">'
            f'<mxGeometry x="-0.8" relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry>'
            f'</mxCell>'
        )
    if e.mult_tgt:
        parts.append(
            f'<mxCell id="{eid}_mt" value="{escape(e.mult_tgt)}" '
            f'style="edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;'
            f'points=[];fontSize=11;" vertex="1" connectable="0" parent="{eid}">'
            f'<mxGeometry x="0.8" relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry>'
            f'</mxCell>'
        )
    return "\n".join(parts)


def main() -> None:
    body: list[str] = []

    for c in CLASSES:
        body.append(class_xml(c))
    for i, e in enumerate(EDGES, 1):
        body.append(edge_xml(e, i))

    xml = f"""<mxfile host="app.diagrams.net" agent="claude-code">
  <diagram name="Architecture" id="arch1">
    <mxGraphModel dx="2400" dy="1600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="1800" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{chr(10).join(body)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    Path("architecture_uml.drawio").write_text(xml, encoding="utf-8")
    print("Wrote architecture_uml.drawio")


if __name__ == "__main__":
    main()
