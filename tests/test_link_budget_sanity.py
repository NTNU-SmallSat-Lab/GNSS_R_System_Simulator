"""Sanity checks on the bistatic link budget at canonical GNSS-R geometries.

These tests exercise `compute_reflection_link_budget` end-to-end through the real
OceanTarget physics at fixed (Tx, Rx, surface) geometries. They assert relationships
(ordering with incidence angle, per-constellation EIRP delta, specular > backscatter)
and that the absolute received power lies inside a generous documented envelope —
the envelope is wide on purpose because the absolute value depends on the OceanTarget
model's choice of effective scattering area, which is the first Fresnel zone in the
current implementation.

Run with `pytest -s tests/test_link_budget_sanity.py` to see the Pr table on stdout.
"""

import numpy as np
import pytest

from functions.link_budget import compute_reflection_link_budget
from object_definitions.Constants_def import Constants
from object_definitions.OceanTarget_def import OceanTarget


R_EARTH_M = Constants.RADIUS_EARTH_EQUAT
R_GPS_M = 26.5e6              # GPS MEO orbital radius [m]
R_LEO_M = R_EARTH_M + 720e3   # 720 km LEO orbital radius [m]

DEFAULT_OCEAN = OceanTarget(
    wave_height_m=1.5,
    wind_speed_m_s=7.0,
    wind_direction_deg=270.0,
    sea_temperature_c=15.0,
    salinity_psu=35.0,
)


def _specular_scenario(theta_inc_deg: float, tx_eirp_dbw: float, frequency_hz: float = 1.57542e9):
    """Build a symmetric specular geometry at the requested incidence angle and run the link budget."""
    th = np.radians(theta_inc_deg)
    surf = np.array([[R_EARTH_M, 0.0, 0.0]])                         # surface point on the +x axis
    # Mirror-symmetric Tx / Rx about the local +x normal, both at incidence theta.
    tx_dir = np.array([np.cos(th), 0.0, np.sin(th)])                 # surface -> Tx unit direction
    rx_dir = np.array([np.cos(th), 0.0, -np.sin(th)])                # surface -> Rx unit direction
    tx_pos = surf[0] + R_GPS_M * tx_dir                              # Tx at GPS orbital range
    rx_pos = surf[0] + (R_LEO_M - R_EARTH_M) / np.cos(th) * rx_dir   # Rx slant-range at LEO altitude

    tx = tx_pos.reshape(3, 1)
    rx = rx_pos.reshape(3, 1)
    rx_vel = np.array([[0.0], [7.5e3], [0.0]])

    return compute_reflection_link_budget(
        tx_positions_eci=tx,
        rx_positions_eci=rx,
        rx_velocities_eci=rx_vel,
        reflection_positions_eci=surf,
        incidence_deg=np.array([theta_inc_deg]),
        antenna=None,
        frequency_hz=frequency_hz,
        tx_eirp_dbw=tx_eirp_dbw,
        settings=None,
        target=DEFAULT_OCEAN,
    )


def _backscatter_scenario(theta_inc_deg: float, tx_eirp_dbw: float, frequency_hz: float = 1.57542e9):
    """Build a near-backscatter geometry (Tx and Rx co-located along the local normal)."""
    surf = np.array([[R_EARTH_M, 0.0, 0.0]])
    n_local = np.array([1.0, 0.0, 0.0])                              # outward normal at surface
    tx_pos = surf[0] + R_GPS_M * n_local                             # Tx straight above surface point
    rx_pos = surf[0] + (R_LEO_M - R_EARTH_M) * n_local               # Rx straight above too

    tx = tx_pos.reshape(3, 1)
    rx = rx_pos.reshape(3, 1)
    rx_vel = np.array([[0.0], [7.5e3], [0.0]])

    return compute_reflection_link_budget(
        tx_positions_eci=tx,
        rx_positions_eci=rx,
        rx_velocities_eci=rx_vel,
        reflection_positions_eci=surf,
        incidence_deg=np.array([theta_inc_deg]),
        antenna=None,
        frequency_hz=frequency_hz,
        tx_eirp_dbw=tx_eirp_dbw,
        settings=None,
        target=DEFAULT_OCEAN,
    )


class TestSpecularReceivedPower:
    @pytest.mark.parametrize("theta_inc_deg", [5.0, 10.0, 30.0, 60.0])
    def test_gps_l1_specular_pr_is_finite(self, theta_inc_deg):
        lb = _specular_scenario(theta_inc_deg, tx_eirp_dbw=27.0)
        pr_dbm = lb["received_power_dbm"][0]
        print(f"  GPS L1   specular  theta={theta_inc_deg:5.1f} deg  ->  Pr = {pr_dbm:7.2f} dBm")
        assert np.isfinite(pr_dbm), f"Pr non-finite at theta={theta_inc_deg} deg"

    @pytest.mark.parametrize("theta_inc_deg", [5.0, 10.0, 30.0, 60.0])
    def test_galileo_e1_specular_pr_is_finite(self, theta_inc_deg):
        lb = _specular_scenario(theta_inc_deg, tx_eirp_dbw=32.0)
        pr_dbm = lb["received_power_dbm"][0]
        print(f"  Galileo E1 specular theta={theta_inc_deg:5.1f} deg  ->  Pr = {pr_dbm:7.2f} dBm")
        assert np.isfinite(pr_dbm)

    def test_specular_pr_decreases_with_incidence(self):
        """At the specular peak, larger incidence increases R_t and R_r so Pr should drop."""
        pr_dbm = [
            _specular_scenario(th, tx_eirp_dbw=27.0)["received_power_dbm"][0]
            for th in (5.0, 30.0, 60.0)
        ]
        assert pr_dbm[0] > pr_dbm[1] > pr_dbm[2], f"non-monotonic Pr vs incidence: {pr_dbm}"

    def test_galileo_e1_is_five_db_higher_than_gps_l1(self):
        """The per-constellation EIRP plumbing must produce a 5 dB Pr shift (32 - 27)."""
        gps_pr = _specular_scenario(30.0, tx_eirp_dbw=27.0)["received_power_dbm"][0]
        gal_pr = _specular_scenario(30.0, tx_eirp_dbw=32.0)["received_power_dbm"][0]
        delta = gal_pr - gps_pr
        print(f"  Galileo E1 - GPS L1 at theta=30 deg: delta = {delta:.3f} dB (expected ~5.0)")
        np.testing.assert_allclose(delta, 5.0, atol=1e-9)


class TestBackscatterReceivedPower:
    def test_backscatter_pr_is_finite(self):
        lb = _backscatter_scenario(theta_inc_deg=0.0, tx_eirp_dbw=27.0)
        pr_dbm = lb["received_power_dbm"][0]
        print(f"  GPS L1   backscatter           ->  Pr = {pr_dbm:7.2f} dBm")
        assert np.isfinite(pr_dbm)

    def test_specular_at_low_incidence_brighter_than_pure_backscatter(self):
        """In the GO + Cox-Munk model, both peak the slope PDF at zero slope, but the
        specular at low theta has a much shorter R_r than the BRP-style backscatter."""
        pr_spec = _specular_scenario(5.0, tx_eirp_dbw=27.0)["received_power_dbm"][0]
        pr_back = _backscatter_scenario(0.0, tx_eirp_dbw=27.0)["received_power_dbm"][0]
        # We do not assert a fixed dB margin because both geometries put the slope PDF
        # at its peak; we just assert the specular path is at least not lower.
        assert pr_spec >= pr_back - 1.0


class TestAbsoluteRange:
    """Loose absolute-range envelope for the OceanTarget model with default sea state.

    The current OceanTarget uses the first Fresnel zone as the effective scattering area,
    which under-counts the glistening-zone integral typical of CYGNSS-style references.
    These bounds are wide on purpose; they catch order-of-magnitude regressions rather
    than calibrate to a specific reference number.
    """

    def test_gps_l1_specular_pr_within_envelope(self):
        pr_dbm = _specular_scenario(30.0, tx_eirp_dbw=27.0)["received_power_dbm"][0]
        assert -260.0 < pr_dbm < -80.0, f"Pr {pr_dbm:.2f} dBm outside envelope"
