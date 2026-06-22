"""Generate the orbits figure for the satellites/constellations page, straight from the code.

Builds the LEO receiver constellation from Configuration_area.ini, propagates it over one
orbital period so the rings close, and calls the same Plotting.plot_orbits used by the area-
coverage run. That writes OUTPUT/Orbits.png (static) and OUTPUT/Orbits.html (a rotatable, self-
contained plotly view with a textured land/ocean globe); the HTML is copied into
docs/source/_static/orbits.html and embedded in the documentation as an interactive iframe.
"""

import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no Qt window, just the saved PNG

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)  # Constants resolves DATA/ and OUTPUT/ relative to the working directory
sys.path.insert(0, os.path.join(REPO_ROOT, "SRC"))  # entry-point scripts live inside SRC/

from __init__ import initialize  # noqa: E402
from object_definitions.SatConstellation_def import SatelliteConstellation  # noqa: E402
from object_definitions.Satellite_def import Satellite  # noqa: E402
from object_definitions.Static_classes_def import Plotting  # noqa: E402

ORBIT_PERIOD_HOURS = 1.8  # ~ one period at 7300 km semi-major axis, so each orbit closes into a ring


def main():
    _, sim = initialize("Configuration_area.ini")  # same parser the area-coverage run uses

    sats = [Satellite(i, orbit_n=sim["orbits"][i]) for i in range(sim["No_satellites"])]
    leo = SatelliteConstellation(satConfigName="LEO GNSS-R constellation", satArray=sats)
    leo.set_simulation_duration(sim["start_time"], ORBIT_PERIOD_HOURS, sim["step_size"])
    leo.propagate_constellation(sim["start_time"], leo.duration_sec, leo.propagation_step_size)

    Plotting.plot_orbits(leo)  # writes OUTPUT/Orbits.png and OUTPUT/Orbits.html

    src_html = os.path.join(REPO_ROOT, "OUTPUT", "Orbits.html")
    dst_html = os.path.join(REPO_ROOT, "docs", "source", "_static", "orbits.html")
    shutil.copyfile(src_html, dst_html)  # the interactive globe embedded by the docs iframe
    print(f"Copied {src_html} -> {dst_html}")


if __name__ == "__main__":
    main()
