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
' Displayed identifiers use hyphens instead of underscores so the rendered SVG
' is LaTeX-safe: when the SVG is included via inkscape --export-latex or the
' `svg` package, text is passed through LaTeX and bare `_` triggers math-mode
' subscript. Class identifiers used in relationships keep their underscores
' (they are never rendered) via "Display name" as InternalName aliases.
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
        sat-constellation
        gnss-constellation
        shp-input-file-path
        gridded-shapefile-path
        grid-size-deg / grid-size-m
        bounding-box
        reflection-dict
        coverage-dict
        earth-model
        --
        calculate-reflection-points-and-coverage()
        calculate-statistics()
        plot-coverage-shapefile()
        plot-coverage-gridded-shapefile()
    }

    class MovingTargetDetection {
        moving-target-object
        sat-constellation
        gnss-constellation
        sat-antenna
        reflection-dict-timeSorted
        target-detect-dict
        earth-model
        reflection-mode
        deltaT-sec
        --
        calculate-moving-target-reflection-intersection()
        show-track-and-detection-points()
        create-reflection-animation()
    }

    class ShipTrajectory {
        ship-id
        ship-speed
        start-position / time
        end-position / time
        waypoints
        trajectory-points
        shapefile-path
        ocean-polygons
        --
        plan-route()
        generate-trajectory-equal-time()
        get-position-at-time(t)
        get-travel-duration-hours()
    }
}

package "Geometry" <<Rectangle>> #FFF4F4 {
    class Reflection {
        tx-satID, rx-satID
        earth-model
        cutoff-angle
        gnss-const
        time-vec
        refl-ECI-vec / refl-LL-vec
        refl-ang-vec
        link-budget
        --
        compute-reflection-coordinates()
        get-reflection-point-ff() <i>spherical, initial guess</i>
    }

    class BRP <<backscatter>> {
        tx-satID, rx-satID
        earth-model
        cutoff-angle
        gnss-const
        time-vec
        refl-ECI-vec / refl-LL-vec
        refl-ang-vec
        link-budget
        --
        compute-brp-coordinates()
    }
}

package "Domain" <<Rectangle>> #F4F4FF {
    class SatelliteConstellation {
        satConfigName
        satellites
        no-satellites
        orbits-array
        times-skyfield
        duration-sec
        propagation-step-size
        --
        set-simulation-duration()
        propagate-constellation()
        generate-constellation-tle()
        add-satellite() / remove-satellite()
    }

    class Satellite {
        satID, satName
        classification
        orbit-n
        skf
        tle-file-path
        gnss-r-antenna
        battery, solarArray
        positions-eci, velocities-eci
        times-skyfield, time-prop-vec
        sunlit-count, sunlit-fraction
        --
        propagate-orbit()
        propagate-orbit-for-times()
        calculate-sunlight-time()
        set-orbit() / set-epoch()
        write-tle-file()
    }

    class Orbit {
        i, e, a
        RAAN, p, M
        n  <i>(mean motion)</i>
        epoch-str
        dt-start
        --
        set-start-time(dt)
        conv-DateTime-to-Epoch()
    }

    class "GNSS-R-antenna" as GNSS_R_antenna {
        antenna-lookuptable
        steering-azimuth / elevation
        frequencies, bands
        polarizations
        footprint-dB-contour
        R-antenna-to-body
        --
        get-gain-dbi-for-direction()
    }

    class "Skyfield satellite" as Skyfield <<external>> {
        TLE-derived state propagator
        --
        at(time)
        is-sunlit(planets)
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
        tx-eirp-dbw
        system-losses-db
        polarization-losses-db
        ocean-sigma0-db
        ocean-sigma-slope-db-per-deg
        ocean-sigma-floor-db
        fresnel-rcs-scale
    }

    class ReadWriteData <<utility>> {
        --
        generate-tle-data-string(sat)
        read-tle-data(path)
        download-and-read-tle(path)
    }

    class Plotting <<static>> {
        --
        plot-reflection-traces-gnss-const()
        plot-reflection-traces-total()
        plot-orbits()
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
    out_dir = Path(__file__).resolve().parent
    (out_dir / "architecture.puml").write_text(PUML, encoding="utf-8")
    print(f"Wrote {out_dir / 'architecture.puml'}")

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
        out_path = out_dir / fname
        out_path.write_bytes(data)
        print(f"Wrote {out_path} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
