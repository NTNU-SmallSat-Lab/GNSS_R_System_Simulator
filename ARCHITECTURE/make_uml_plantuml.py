"""Generate the UML class diagrams with PlantUML.

Two publication-sized figures share one style; stub classes mark the seam between them:
  architecture.puml / .svg / .png  - system architecture (modes, geometry, satellites, foundation)
  scattering.puml   / .svg / .png  - scattering and link-budget model (Target hierarchy, physics modules)

The SVGs are also copied to docs/source/images/ as uml_system.svg and uml_scattering.svg
(embedded by architecture.rst). Rendering uses the public PlantUML server; the sources can also be
pasted into https://www.plantuml.com/plantuml or https://kroki.io, or rendered locally with
plantuml.jar. For a publication, export PDF from the same source:
https://www.plantuml.com/plantuml/pdf/<encoded>.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request
import zlib

STYLE = r"""
skinparam shadowing false
skinparam dpi 150
skinparam ArrowColor #222222
skinparam packageStyle rectangle
skinparam PackageBorderColor #444444
skinparam PackageFontStyle bold
skinparam class {
    BackgroundColor #ffffff
    BorderColor #222222
    AttributeIconSize 0
    StereotypeFontColor #555555
}
hide empty members
"""

PUML_SYSTEM = r"""@startuml
title GNSS-R System Simulator -- system architecture (UML)
""" + STYLE + r"""
package "Analysis modes" #F7F7F7 {
    class Coverage {
        sat_constellation, gnss_constellation
        grid_size_deg / grid_size_m
        coverage_dict, selected_signals
        earth_model, max_central_angle_deg
        --
        initialize_coverage()
        compute_coverage_for_constellation()
        calculate_statistics()
        plot_coverage_gridded_shapefile()
    }

    class MovingTargetDetection {
        moving_target_object, sat_antenna
        sat_constellation, gnss_constellation
        reflection_mode <i>specular | backscatter</i>
        signals_per_constellation
        link_budget_settings, ship_dimensions
        target_detect_dict, deltaT_sec
        --
        calculate_moving_target_reflection_intersection()
        show_link_budget_contributions()
        show_track_and_detection_points()
        create_reflection_animation()
    }

    class ShipTrajectory {
        ship_id, ship_speed
        start / end position and time
        waypoints, trajectory_points
        shapefile_path
        --
        plan_route()
        generate_trajectory_equal_time()
        get_position_at_time(t)
    }
}

package "Reflection geometry" #F7F7F7 {
    class Reflection {
        tx_satID, rx_satID
        earth_model, max_central_angle_deg
        time_vec, refl_ECI_vec / refl_LL_vec
        refl_ang_vec, link_budget
        --
        compute_reflection_coordinates()
        get_reflection_point_ff()
    }

    class BRP <<backscatter>> {
        tx_satID, rx_satID
        earth_model
        max_central_angle_deg
        max_incidence_angle_deg
        time_vec, refl_ECI_vec / refl_LL_vec
        refl_ang_vec, link_budget
        --
        compute_brp_coordinates()
    }
}

package "Satellites and antenna" #F7F7F7 {
    class SatelliteConstellation {
        satConfigName, satellites
        times_skyfield, duration_sec
        propagation_step_size
        --
        set_simulation_duration()
        propagate_constellation()
        add_satellite() / remove_satellite()
    }

    class Satellite {
        satID, satName
        orbit_n, skf, tle_file_path
        positions_eci, velocities_eci
        --
        propagate_orbit()
        propagate_orbit_for_times()
        write_tle_file()
        calculate_sunlight_time()
    }

    class Orbit_specification {
        i, e, RAAN, p, nu <i>(true anomaly)</i>
        a, n <i>(mean motion)</i>
        epoch_str, dt_start
        --
        set_start_time(dt)
    }

    class RxAntenna {
        antenna_lookuptable
        max_across/along_track_steering_deg
        frequencies, bands, polarizations
        footprint_dB_contour, R_antenna_to_body
        --
        get_gain_dbi_for_direction()
        resolve_steering_angles_for_direction()
    }

    class AntennaPatternGenerator {
        elements_along_x/y, d_x, d_y
        freqs, steering sweeps
        --
        generate() <i>array-factor JSONs</i>
    }

    class "Skyfield EarthSatellite" as Skyfield <<external>> {
        SGP4 propagation from a TLE
        --
        at(time)
    }
}

package "Foundation" #F7F7F7 {
    class Constants <<static>> {
        physical constants, WGS-84
        BANDS_CENTER_FREQUENCIES_Hz
        GNSS_EIRP_DBW, GNSS_CHIP_RATE_HZ
        data / output paths
    }

    class ReadWriteSatData <<utility>> {
        --
        generate_tle_data_string() <i>SGP4 mean-element fit</i>
        read_tle_data()
        mean_to_true_anomaly_deg()
    }
}

class "functions.link_budget" as link_budget <<module>>
note right of link_budget
  Scattering and link-budget
  model: see companion diagram.
end note

' ---- Composition / aggregation -----------------------------------
SatelliteConstellation "1" *-- "1..*" Satellite : contains
Satellite "1" o-- "0..1" Orbit_specification
Satellite "1" o-- "1" Skyfield : wraps
Satellite "1" o-- "0..1" RxAntenna : may carry

' ---- Reflection / BRP need a (Tx, Rx) Satellite pair -------------
Reflection "*" -- "2" Satellite : Tx + Rx
BRP        "*" -- "2" Satellite : Tx + Rx

' ---- The two analysis modes orchestrate the model -----------------
Coverage "1" o-- "1..*" SatelliteConstellation : Rx + GNSS
Coverage "1" o-- "*"    Reflection             : specular mode
Coverage "1" o-- "*"    BRP                    : backscatter mode

MovingTargetDetection "1" o-- "1..*" SatelliteConstellation : Rx + GNSS
MovingTargetDetection "1" o-- "1"    ShipTrajectory          : target
MovingTargetDetection "1" o-- "1"    RxAntenna
MovingTargetDetection "1" o-- "*"    Reflection              : specular mode
MovingTargetDetection "1" o-- "*"    BRP                     : backscatter mode

' ---- Hand-off to the scattering / link-budget model ---------------
Reflection ..> link_budget : evaluates
BRP        ..> link_budget : evaluates
MovingTargetDetection ..> link_budget : per-band budgets

RxAntenna ..> AntennaPatternGenerator : loads pattern JSONs
Satellite ..> ReadWriteSatData : TLE fit + I/O

note bottom of Constants
  Imported nearly everywhere
  (kept implicit for readability);
  transmit EIRP and chip rates
  live here, not in the settings.
end note

legend right
  **Notation**
  *--   composition (owns)
  o--   aggregation (holds)
  ..>   dependency (uses)
  --|>  inheritance
  italic = abstract
endlegend

@enduml
"""

PUML_SCATTERING = r"""@startuml
title GNSS-R System Simulator -- scattering and link-budget model (UML)
""" + STYLE + r"""
package "Consumers (system diagram)" #F7F7F7 {
    class Reflection
    class BRP <<backscatter>>
    class MovingTargetDetection
    class RxAntenna
}

package "Link budget" #F7F7F7 {
    class "functions.link_budget" as link_budget <<module>> {
        --
        build_bistatic_geometry()
        compute_reflection_link_budget()
        compute_bistatic_link_budget()
        coherent_processing_gain_linear()
    }

    class BistaticGeometry <<dataclass>> {
        n_surface, n_tx2s, n_s2rx
        range_tx_m, range_rx_m
        incidence / scattering / bistatic angles
    }

    class BistaticLinkBudgetSettings <<dataclass>> {
        system_losses_db, polarization_losses_db
        noise_bandwidth_hz, system_noise_temperature_k
        detection_sinr_threshold_db
        {field} ocean sea state (H_s, U10, dir, T, S, Omega_c)
        delay_doppler_cell_enabled, coherent_integration_time_s
        ocean_patch_integration_enabled, num_incoherent_looks
    }

    class "functions.delay_doppler" as delay_doppler <<module>> {
        --
        delay_doppler_cell_area()
    }
}

package "Targets (RCS models)" #F7F7F7 {
    abstract class Target {
        --
        {abstract} compute_rcs(geometry, wavelength, area)
    }

    class TargetRcs <<dataclass>> {
        rcs_m2, info
    }

    class OceanTarget {
        wave_height_m, wind_speed_m_s
        wind_direction_deg, inverse_wave_age
        sea_temperature_c, salinity_psu
        --
        compute_rcs() <i>coherent + GO + Bragg</i>
        from_settings(settings)
    }

    class ShipTarget {
        length_m, beam_m, freeboard_m
        heading_eci
        --
        compute_rcs() <i>5 plates + 2 dihedrals</i>
    }
}

package "Physics helpers" #F7F7F7 {
    class "functions.ocean_physics" as ocean_physics <<module>> {
        --
        seawater_permittivity() <i>Klein-Swift</i>
        circular_pol_reflectivities() <i>Fresnel</i>
        large_scale_slope_variances() <i>Elfouhaily</i>
        two_scale_backscatter_nrcs() <i>Bragg</i>
    }

    class "functions.ship_physics" as ship_physics <<module>> {
        --
        flat_plate_rcs()
        dihedral_rcs()
    }
}

' ---- Consumers hand the geometry to the link budget ---------------
Reflection ..> link_budget : evaluates
BRP        ..> link_budget : evaluates
MovingTargetDetection ..> link_budget : per-band budgets
MovingTargetDetection ..> ShipTarget  : per-event ship RCS

' ---- The link budget builds geometry and asks a Target for RCS ----
link_budget ..> BistaticGeometry : builds
link_budget ..> BistaticLinkBudgetSettings : settings
link_budget ..> Target : compute_rcs()
link_budget ..> RxAntenna : steered gain
link_budget ..> delay_doppler : DDM cell area

' ---- Target hierarchy ---------------------------------------------
Target <|-- OceanTarget
Target <|-- ShipTarget
Target ..> TargetRcs : returns

OceanTarget ..> ocean_physics : uses
OceanTarget ..> BistaticLinkBudgetSettings : from_settings
ShipTarget  ..> ship_physics : uses

note bottom of link_budget
  Pr = EIRP G Rx lambda^2 sigma / ((4 pi)^3 Rt^2 Rr^2 L);
  transmit EIRP and chip rates come from Constants
  per (constellation, band).
end note

@enduml
"""


# --------------------------------------------------------------------------
#  PlantUML deflate + custom-base64 encoding (so we can hit the web renderer)
# --------------------------------------------------------------------------
def _encode6bit(b: int) -> str:
    if b < 10:
        return chr(48 + b)         # 0-9
    b -= 10
    if b < 26:
        return chr(65 + b)         # A-Z
    b -= 26
    if b < 26:
        return chr(97 + b)         # a-z
    b -= 26
    if b == 0:
        return "-"
    if b == 1:
        return "_"
    return "?"


def _encode64(data: bytes) -> str:
    out = []
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        b1 = chunk[0]
        b2 = chunk[1] if len(chunk) > 1 else 0
        b3 = chunk[2] if len(chunk) > 2 else 0
        out.append(_encode6bit(b1 >> 2))
        out.append(_encode6bit(((b1 & 0x3) << 4) | (b2 >> 4)))
        out.append(_encode6bit(((b2 & 0xF) << 2) | (b3 >> 6)))
        out.append(_encode6bit(b3 & 0x3F))
    return "".join(out)


def encode_plantuml(text: str) -> str:
    """Compress (raw deflate) + PlantUML-base64 encode."""
    raw = text.encode("utf-8")
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed = compressor.compress(raw) + compressor.flush()
    return _encode64(compressed)


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "claude-code/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    images = here.parent / "docs" / "source" / "images"
    base = "https://www.plantuml.com/plantuml"

    diagrams = [
        ("architecture", PUML_SYSTEM, images / "uml_system.svg"),
        ("scattering", PUML_SCATTERING, images / "uml_scattering.svg"),
    ]
    for stem, source, docs_svg in diagrams:
        (here / f"{stem}.puml").write_text(source, encoding="utf-8")
        print(f"Wrote {stem}.puml")

        encoded = encode_plantuml(source)
        for suffix in ("svg", "png"):
            url = f"{base}/{suffix}/{encoded}"
            try:
                data = _fetch(url)
            except Exception as e:                              # noqa: BLE001
                print(f"FAILED to fetch {stem}.{suffix}: {e}")
                print(f"  Render manually by pasting {stem}.puml into")
                print("  https://www.plantuml.com/plantuml or https://kroki.io")
                continue
            (here / f"{stem}.{suffix}").write_bytes(data)
            print(f"Wrote {stem}.{suffix} ({len(data):,} bytes)")
            if suffix == "svg":
                docs_svg.write_bytes(data)
                print(f"Wrote {docs_svg} (docs embed)")


if __name__ == "__main__":
    main()
