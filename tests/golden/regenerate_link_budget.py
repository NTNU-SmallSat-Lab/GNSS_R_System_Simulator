"""Regenerate the link-budget golden file.

Run this manually when:
  * the bistatic radar equation, OceanTarget physics or BistaticGeometry helper
    intentionally change; OR
  * the fixture geometry below is updated.

Usage (from repo root):
    python tests/golden/regenerate_link_budget.py

This script intentionally lives outside the pytest collection (no `test_` prefix) so it never
runs as part of the test suite.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "SRC"))

import numpy as np

from functions.link_budget import compute_reflection_link_budget
from object_definitions.Constants_def import Constants
from object_definitions.OceanTarget_def import OceanTarget


def _fixture_scenario():
    """Deterministic, hand-built geometry for the golden regression.

    Three reflection samples, all at GPS L1:
      0: specular at  5 deg incidence
      1: specular at 30 deg incidence
      2: specular at 60 deg incidence
    All three share the same surface point on the +x axis at the Earth equator;
    Tx is at GPS MEO range, Rx at 720 km LEO altitude, both symmetric about the
    local surface normal.
    """
    R_earth = Constants.RADIUS_EARTH_EQUAT
    R_gps = 26.5e6
    R_leo = R_earth + 720e3

    thetas_deg = np.array([5.0, 30.0, 60.0])
    n = thetas_deg.size

    surf = np.tile(np.array([R_earth, 0.0, 0.0]), (n, 1))
    tx_pos = np.empty((3, n))
    rx_pos = np.empty((3, n))
    for k, th_deg in enumerate(thetas_deg):
        th = np.radians(th_deg)
        tx_dir = np.array([np.cos(th), 0.0, np.sin(th)])
        rx_dir = np.array([np.cos(th), 0.0, -np.sin(th)])
        tx_pos[:, k] = surf[k] + R_gps * tx_dir
        rx_pos[:, k] = surf[k] + (R_leo - R_earth) / np.cos(th) * rx_dir

    rx_vel = np.tile(np.array([[0.0], [7.5e3], [0.0]]), (1, n))

    target = OceanTarget(
        wave_height_m=1.5,
        wind_speed_m_s=7.0,
        wind_direction_deg=270.0,
        sea_temperature_c=15.0,
        salinity_psu=35.0,
    )
    return {
        "tx_positions_eci": tx_pos,
        "rx_positions_eci": rx_pos,
        "rx_velocities_eci": rx_vel,
        "reflection_positions_eci": surf,
        "incidence_deg": thetas_deg,
        "antenna": None,
        "frequency_hz": 1.57542e9,
        "tx_eirp_dbw": 27.0,
        "settings": None,
        "target": target,
    }


def main() -> None:
    inputs = _fixture_scenario()
    lb = compute_reflection_link_budget(**inputs)

    out_path = Path(__file__).parent / "reference_link_budget.npz"
    np.savez(
        out_path,
        received_power_w=lb["received_power_w"],
        received_power_dbm=lb["received_power_dbm"],
        rcs_m2=lb["rcs_m2"],
        fspl_sum_db=lb["fspl_sum_db"],
        incidence_deg=lb["incidence_deg"],
        scattering_deg=lb["scattering_deg"],
        bistatic_deg=lb["bistatic_deg"],
        tx_to_surface_range_m=lb["tx_to_surface_range_m"],
        surface_to_rx_range_m=lb["surface_to_rx_range_m"],
    )
    print(
        f"Wrote {out_path}\n"
        f"  Pr [dBm] = {lb['received_power_dbm']}\n"
        f"  RCS [m^2] = {lb['rcs_m2']}\n"
        f"  bistatic [deg] = {lb['bistatic_deg']}"
    )


if __name__ == "__main__":
    main()
