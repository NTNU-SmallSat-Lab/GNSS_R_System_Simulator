"""Delay-Doppler resolution-cell geometry tests.

Validates functions.delay_doppler.delay_doppler_cell_area against finite differences of the analytic
(delay, Doppler) -> surface map, plus chip-length, proportionality, footprint-cap, NaN-propagation,
order-of-magnitude, the Constants chip-rate table, and the settings validation that gates the cell.
"""

import numpy as np
import pytest

from functions.delay_doppler import delay_doppler_cell_area
from functions.link_budget import BistaticLinkBudgetSettings, build_bistatic_geometry
from object_definitions.Constants_def import Constants

C = Constants.SPEED_OF_LIGHT
WAVELENGTH_L1 = C / 1.575_42e9          # GPS L1 carrier wavelength [m]
CHIP_LENGTH_CA = C / 1.023e6           # GPS C/A one-chip path length [m] ~ 293.05 m
T_COH = 1.0e-3                         # 1 ms coherent integration


# A representative single-snapshot bistatic geometry: surface point on a spherical Earth, a LEO Rx at
# ~730 km, a GNSS Tx at MEO, with plausible orbital velocities. Exact values are arbitrary; the
# finite-difference test validates the math for whatever geometry is supplied.
R_E = 6.371e6
SURFACE = np.array([[R_E, 0.0, 0.0]])                       # (1, 3)
NORMAL = np.array([[1.0, 0.0, 0.0]])                        # (1, 3) outward normal
RX_POS = np.array([[R_E + 7.0e5], [1.0e6], [0.2e6]])       # (3, 1) LEO
TX_POS = np.array([[R_E + 2.0e7], [-3.0e6], [1.0e6]])      # (3, 1) MEO
RX_VEL = np.array([[-1.0e3], [7.0e3], [1.0e2]])            # (3, 1) ~7 km/s
TX_VEL = np.array([[-1.5e3], [2.5e3], [3.0e3]])            # (3, 1) ~4 km/s


def _geometry(tx_pos=TX_POS, rx_pos=RX_POS, surface=SURFACE, normal=NORMAL):
    return build_bistatic_geometry(tx_pos, rx_pos, surface, normal)


def _path_length(r, tx, rx):
    """L(r) = R_t + R_r for column-vector tx/rx and point r."""
    return np.linalg.norm(r - tx) + np.linalg.norm(rx - r)


def _doppler(r, tx, rx, v_tx, v_rx, wavelength_m):
    """f_D(r) = (1/lambda)(n_tx2s . v_tx - n_s2rx . v_rx)."""
    n_tx2s = (r - tx) / np.linalg.norm(r - tx)
    n_s2rx = (rx - r) / np.linalg.norm(rx - r)
    return (n_tx2s @ v_tx - n_s2rx @ v_rx) / wavelength_m


def _tangent_basis(normal):
    n = normal / np.linalg.norm(normal)
    seed = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


class TestJacobianAgainstFiniteDifferences:
    def test_jacobian_det_matches_central_difference(self):
        geom = _geometry()
        result = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)

        r0 = SURFACE[0]
        tx = TX_POS[:, 0]
        rx = RX_POS[:, 0]
        v_tx = TX_VEL[:, 0]
        v_rx = RX_VEL[:, 0]
        e1, e2 = _tangent_basis(NORMAL[0])
        h = 1.0  # 1 m perturbation; ranges ~1e6 m so the linearization is essentially exact

        dL_de1 = (_path_length(r0 + h * e1, tx, rx) - _path_length(r0 - h * e1, tx, rx)) / (2 * h)
        dL_de2 = (_path_length(r0 + h * e2, tx, rx) - _path_length(r0 - h * e2, tx, rx)) / (2 * h)
        dfD_de1 = (_doppler(r0 + h * e1, tx, rx, v_tx, v_rx, WAVELENGTH_L1) - _doppler(r0 - h * e1, tx, rx, v_tx, v_rx, WAVELENGTH_L1)) / (2 * h)
        dfD_de2 = (_doppler(r0 + h * e2, tx, rx, v_tx, v_rx, WAVELENGTH_L1) - _doppler(r0 - h * e2, tx, rx, v_tx, v_rx, WAVELENGTH_L1)) / (2 * h)

        det_fd = abs(dL_de1 * dfD_de2 - dL_de2 * dfD_de1)
        assert result["abs_jacobian_det_hz_m"][0] == pytest.approx(det_fd, rel=1e-5)

        area_fd = CHIP_LENGTH_CA * (1.0 / T_COH) / det_fd
        assert result["uncapped_cell_area_m2"][0] == pytest.approx(area_fd, rel=1e-5)

    def test_delay_gradient_matches_central_difference(self):
        geom = _geometry()
        result = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)
        r0, tx, rx = SURFACE[0], TX_POS[:, 0], RX_POS[:, 0]
        e1, e2 = _tangent_basis(NORMAL[0])
        h = 1.0
        dL_de1 = (_path_length(r0 + h * e1, tx, rx) - _path_length(r0 - h * e1, tx, rx)) / (2 * h)
        dL_de2 = (_path_length(r0 + h * e2, tx, rx) - _path_length(r0 - h * e2, tx, rx)) / (2 * h)
        grad_mag_fd = np.hypot(dL_de1, dL_de2)
        assert result["delay_gradient"][0] == pytest.approx(grad_mag_fd, rel=1e-5)


class TestChipLength:
    def test_gps_ca_chip_length_is_293_m(self):
        rate = Constants.get_chip_rate_hz_for_band("GPS", "L1")
        assert rate == pytest.approx(1.023e6)
        assert C / rate == pytest.approx(293.0522, abs=1e-3)


class TestScaling:
    def test_area_proportional_to_chip_length(self):
        geom = _geometry()
        a1 = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)
        a2 = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, 2.0 * CHIP_LENGTH_CA, T_COH)
        assert a2["uncapped_cell_area_m2"][0] == pytest.approx(2.0 * a1["uncapped_cell_area_m2"][0], rel=1e-12)

    def test_area_inversely_proportional_to_coherent_time(self):
        geom = _geometry()
        a1 = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)
        a2 = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, 2.0 * T_COH)
        assert a2["uncapped_cell_area_m2"][0] == pytest.approx(0.5 * a1["uncapped_cell_area_m2"][0], rel=1e-12)


class TestFootprintCap:
    def test_small_footprint_caps_cell(self):
        geom = _geometry()
        uncapped = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)["uncapped_cell_area_m2"][0]
        small = uncapped / 10.0
        result = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, footprint_area_m2=small)
        assert result["cell_area_m2"][0] == pytest.approx(small)
        assert bool(result["footprint_capped"][0]) is True

    def test_large_footprint_does_not_cap(self):
        geom = _geometry()
        uncapped = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)["uncapped_cell_area_m2"][0]
        result = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, footprint_area_m2=1e6 * uncapped)
        assert result["cell_area_m2"][0] == pytest.approx(uncapped, rel=1e-9)
        assert bool(result["footprint_capped"][0]) is False


class TestDegenerateAndInvalid:
    def test_near_specular_divergence_collapses_to_footprint(self):
        # A monostatic-style geometry where Tx == Rx makes grad L vanish (specular/backscatter
        # stationary point): the uncapped area diverges and must collapse to the footprint cap.
        geom = build_bistatic_geometry(RX_POS, RX_POS, SURFACE, NORMAL)
        footprint = 5.0e6
        result = delay_doppler_cell_area(geom, RX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, footprint_area_m2=footprint)
        assert result["cell_area_m2"][0] == pytest.approx(footprint)
        assert bool(result["footprint_capped"][0]) is True

    def test_invalid_geometry_propagates_nan(self):
        # Column 0 is the valid Tx position; column 1 carries a NaN Tx coordinate (invalid sample).
        tx = np.array([[TX_POS[0, 0], np.nan], [TX_POS[1, 0], 0.0], [TX_POS[2, 0], 0.0]])
        rx = np.repeat(RX_POS, 2, axis=1)
        surface = np.repeat(SURFACE, 2, axis=0)
        normal = np.repeat(NORMAL, 2, axis=0)
        geom = build_bistatic_geometry(tx, rx, surface, normal)
        result = delay_doppler_cell_area(geom, np.repeat(TX_VEL, 2, axis=1), np.repeat(RX_VEL, 2, axis=1), WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, footprint_area_m2=1e12)
        assert np.isfinite(result["cell_area_m2"][0])      # valid sample unaffected
        assert np.isnan(result["cell_area_m2"][1])         # NaN Tx position -> invalid


class TestOrderOfMagnitude:
    def test_cell_area_in_physical_range(self):
        # GPS C/A, 1 ms coherent integration, realistic LEO/MEO geometry: the DDM pixel should be
        # of order 0.001-100 km^2, far below the ~1e4 km^2 antenna footprint.
        geom = _geometry()
        result = delay_doppler_cell_area(geom, TX_VEL, RX_VEL, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, footprint_area_m2=1e12)
        area = result["cell_area_m2"][0]
        assert 1.0e3 < area < 1.0e8


class TestChipRateTable:
    def test_known_band_lookup(self):
        assert Constants.get_chip_rate_hz_for_band("GPS", "L5") == pytest.approx(10.23e6)
        assert Constants.get_chip_rate_hz_for_band("GALILEO", "E6") == pytest.approx(5.115e6)

    def test_unknown_band_raises(self):
        with pytest.raises(KeyError):
            Constants.get_chip_rate_hz_for_band("GPS", "E1")

    def test_unknown_constellation_raises(self):
        with pytest.raises(KeyError):
            Constants.get_chip_rate_hz_for_band("NOPE", "L1")

    def test_every_eirp_band_has_a_chip_rate(self):
        for constellation, bands in Constants.GNSS_EIRP_DBW.items():
            for band in bands:
                assert Constants.get_chip_rate_hz_for_band(constellation, band) > 0.0


def _settings(**overrides):
    base = {
        "system_losses_db": 2.0,
        "polarization_losses_db": 0.5,
        "ocean_wave_height_m": 1.5,
        "ocean_wind_speed_m_s": 7.0,
        "ocean_wind_direction_deg": 270.0,
        "ocean_sea_temperature_c": 10.0,
        "ocean_salinity_psu": 35.0,
        "ocean_inverse_wave_age": 0.84,
    }
    base.update(overrides)
    return BistaticLinkBudgetSettings(**base)


class TestSettingsValidation:
    def test_cell_enabled_requires_coherent_time(self):
        with pytest.raises(ValueError, match="coherent_integration_time_s is required"):
            _settings(delay_doppler_cell_enabled=True)

    def test_non_positive_coherent_time_raises(self):
        with pytest.raises(ValueError, match="coherent_integration_time_s must be"):
            _settings(coherent_integration_time_s=0.0)

    def test_cell_and_patch_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _settings(delay_doppler_cell_enabled=True, coherent_integration_time_s=1e-3, ocean_patch_integration_enabled=True)

    def test_valid_cell_settings_accepted(self):
        s = _settings(delay_doppler_cell_enabled=True, coherent_integration_time_s=1e-3)
        assert s.delay_doppler_cell_enabled is True
        assert s.coherent_integration_time_s == pytest.approx(1e-3)
