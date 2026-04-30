"""Generate a UML class diagram with PlantUML.

Outputs:
  architecture.puml  - PlantUML source (editable)
  architecture.svg   - rendered vector image
  architecture.png   - rendered raster image

Source can also be pasted into:
  https://www.plantuml.com/plantuml
  https://kroki.io
"""
from __future__ import annotations

import urllib.request
import zlib
from pathlib import Path


PUML = r"""@startuml
!pragma layout smetana
title GNSS-R System Simulator -- Software Architecture

skinparam shadowing false
skinparam ArrowColor #333333
skinparam packageStyle rectangle
skinparam class {
    BackgroundColor #ffffff
    BorderColor #000000
    FontStyle Italic
    AttributeFontStyle Plain
    AttributeIconSize 0
    StereotypeFontColor #555555
}
hide empty members

package "Analysis" <<Rectangle>> #F4FFF4 {
    class Coverage {
        sat_constellation
        gnss_constellation
        shp_input_file_path
        gridded_shapefile_path
        grid_size_deg / grid_size_m
        bounding_box
        reflection_dict
        coverage_dict
        earth_model
        --
        calculate_reflection_points_and_coverage()
        calculate_statistics()
        plot_coverage_shapefile()
        plot_coverage_gridded_shapefile()
    }

    class MovingTargetDetection {
        moving_target_object
        sat_constellation
        gnss_constellation
        sat_antenna
        reflection_dict_timeSorted
        target_detect_dict
        earth_model
        reflection_mode
        deltaT_sec
        --
        calculate_moving_target_reflection_intersection()
        show_track_and_detection_points()
        create_reflection_animation()
    }

    class ShipTrajectory {
        ship_id
        ship_speed
        start_position / time
        end_position / time
        waypoints
        trajectory_points
        shapefile_path
        ocean_polygons
        --
        plan_route()
        generate_trajectory_equal_time()
        get_position_at_time(t)
        get_travel_duration_hours()
    }
}

package "Geometry" <<Rectangle>> #FFF4F4 {
    class Reflection {
        tx_satID, rx_satID
        earth_model
        cutoff_angle
        gnss_const
        time_vec
        refl_ECI_vec / refl_LL_vec
        refl_ang_vec
        link_budget
        --
        compute_reflection_coordinates()
        get_reflection_point_ff() <i>spherical, initial guess</i>
    }

    class BRP <<backscatter>> {
        tx_satID, rx_satID
        earth_model
        cutoff_angle
        gnss_const
        time_vec
        refl_ECI_vec / refl_LL_vec
        refl_ang_vec
        link_budget
        --
        compute_brp_coordinates()
    }
}

package "Domain" <<Rectangle>> #F4F4FF {
    class SatelliteConstellation {
        satConfigName
        satellites
        no_satellites
        orbits_array
        times_skyfield
        duration_sec
        propagation_step_size
        --
        set_simulation_duration()
        propagate_constellation()
        generate_constellation_tle()
        add_satellite() / remove_satellite()
    }

    class Satellite {
        satID, satName
        classification
        orbit_n
        skf
        tle_file_path
        gnss_r_antenna
        battery, solarArray
        positions_eci, velocities_eci
        times_skyfield, time_prop_vec
        sunlit_count, sunlit_fraction
        --
        propagate_orbit()
        propagate_orbit_for_times()
        calculate_sunlight_time()
        set_orbit() / set_epoch()
        write_tle_file()
    }

    class Orbit {
        i, e, a
        RAAN, p, M
        n  <i>(mean motion)</i>
        epoch_str
        dt_start
        --
        set_start_time(dt)
        conv_DateTime_to_Epoch()
    }

    class GNSS_R_antenna {
        antenna_lookuptable
        steering_azimuth / elevation
        frequencies, bands
        polarizations
        footprint_dB_contour
        R_antenna_to_body
        --
        get_gain_dbi_for_direction()
    }

    class "Skyfield satellite" as Skyfield <<external>> {
        TLE-derived state propagator
        --
        at(time)
        is_sunlit(planets)
    }
}

package "Foundation" <<Rectangle>> #F5F5F5 {
    class Constants <<static>> {
        Physical constants
        GNSS band frequencies
        File paths
        Coordinate-system tools
    }

    class BistaticLinkBudgetSettings <<dataclass>> {
        tx_eirp_dbw
        system_losses_db
        polarization_losses_db
        ocean_sigma0_db
        ocean_sigma_slope_db_per_deg
        ocean_sigma_floor_db
        fresnel_rcs_scale
    }

    class ReadWriteData <<utility>> {
        --
        generate_tle_data_string(sat)
        read_tle_data(path)
        download_and_read_tle(path)
    }

    class Plotting <<static>> {
        --
        plot_reflection_traces_gnss_const()
        plot_reflection_traces_total()
        plot_orbits()
    }
}

' ---- Composition / aggregation -----------------------------------
SatelliteConstellation "1" *-- "1..*" Satellite : contains

Satellite "1" o-- "0..1" Orbit
Satellite "1" o-- "0..1" Skyfield : wraps
Satellite "1" o-- "0..1" GNSS_R_antenna : may carry

' ---- Reflection / BRP need a (Tx, Rx) Satellite pair -------------
Reflection "*" -- "2" Satellite : Tx + Rx
BRP        "*" -- "2" Satellite : Tx + Rx

' ---- Coverage and MovingTargetDetection orchestrate the model ----
Coverage "1" o-- "1..*" SatelliteConstellation : Rx + Tx
Coverage "1" o-- "*"    Reflection             : stores

MovingTargetDetection "1" o-- "1..*" SatelliteConstellation : Rx + Tx
MovingTargetDetection "1" o-- "1"    ShipTrajectory          : target
MovingTargetDetection "1" o-- "1"    GNSS_R_antenna          : Rx antenna
MovingTargetDetection "1" o-- "*"    Reflection              : specular mode
MovingTargetDetection "1" o-- "*"    BRP                     : backscatter mode

' ---- Dependencies (uses) -----------------------------------------
Reflection ..> GNSS_R_antenna : uses
BRP        ..> GNSS_R_antenna : uses
Reflection ..> BistaticLinkBudgetSettings : uses
BRP        ..> BistaticLinkBudgetSettings : uses

Satellite             ..> ReadWriteData : TLE I/O
SatelliteConstellation ..> ReadWriteData : uses
Coverage              ..> Plotting      : uses

' Constants is imported almost everywhere - kept implicit so the
' diagram stays readable.

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
    Path("architecture.puml").write_text(PUML, encoding="utf-8")
    print("Wrote architecture.puml")

    encoded = encode_plantuml(PUML)
    print(f"Encoded source ({len(encoded)} chars)")

    base = "https://www.plantuml.com/plantuml"
    targets = [
        ("architecture.svg", f"{base}/svg/{encoded}"),
        ("architecture.png", f"{base}/png/{encoded}"),
    ]
    for fname, url in targets:
        try:
            data = _fetch(url)
        except Exception as e:                                  # noqa: BLE001
            print(f"FAILED to fetch {fname}: {e}")
            print(f"  Render manually by pasting architecture.puml into")
            print(f"  https://www.plantuml.com/plantuml or https://kroki.io")
            continue
        Path(fname).write_bytes(data)
        print(f"Wrote {fname} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
