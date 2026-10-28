"""Delay-Doppler resolution-cell geometry tests.

Validates functions.delay_doppler.delay_doppler_cell_area against finite differences of the analytic
(delay, Doppler) -> surface map, plus chip-length, proportionality, footprint-cap, NaN-propagation,
order-of-magnitude, the Constants chip-rate table, and the settings validation that gates the cell.
"""

import numpy as np
import pymap3d
import pytest

from functions.delay_doppler import delay_doppler_cell_area, delay_doppler_cell_samples
from functions.link_budget import BistaticLinkBudgetSettings, build_bistatic_geometry
from object_definitions.Constants_def import Constants
from object_definitions.Reflection_def import Reflection

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


def _doppler_rotating_surface(r, tx, rx, v_tx, v_rx, wavelength_m):
    """Doppler with a surface point fixed to the rigidly rotating Earth."""
    n_tx2s = (r - tx) / np.linalg.norm(r - tx)
    n_s2rx = (rx - r) / np.linalg.norm(rx - r)
    omega = np.array([0.0, 0.0, Constants.EARTH_ROTATION_RATE_RAD_S])
    v_surface = np.cross(omega, r)
    return (n_tx2s @ (v_tx - v_surface) - n_s2rx @ (v_rx - v_surface)) / wavelength_m


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

    def test_rotating_surface_doppler_jacobian_matches_finite_difference(self):
        geom = _geometry()
        result = delay_doppler_cell_area(
            geom,
            TX_VEL,
            RX_VEL,
            WAVELENGTH_L1,
            CHIP_LENGTH_CA,
            T_COH,
            surface_positions_eci=SURFACE,
        )
        r0 = SURFACE[0]
        tx = TX_POS[:, 0]
        rx = RX_POS[:, 0]
        v_tx = TX_VEL[:, 0]
        v_rx = RX_VEL[:, 0]
        e1, e2 = _tangent_basis(NORMAL[0])
        h = 1.0

        dL_de1 = (_path_length(r0 + h * e1, tx, rx) - _path_length(r0 - h * e1, tx, rx)) / (2 * h)
        dL_de2 = (_path_length(r0 + h * e2, tx, rx) - _path_length(r0 - h * e2, tx, rx)) / (2 * h)
        dfD_de1 = (
            _doppler_rotating_surface(r0 + h * e1, tx, rx, v_tx, v_rx, WAVELENGTH_L1)
            - _doppler_rotating_surface(r0 - h * e1, tx, rx, v_tx, v_rx, WAVELENGTH_L1)
        ) / (2 * h)
        dfD_de2 = (
            _doppler_rotating_surface(r0 + h * e2, tx, rx, v_tx, v_rx, WAVELENGTH_L1)
            - _doppler_rotating_surface(r0 - h * e2, tx, rx, v_tx, v_rx, WAVELENGTH_L1)
        ) / (2 * h)
        det_fd = abs(dL_de1 * dfD_de2 - dL_de2 * dfD_de1)

        assert result["abs_jacobian_det_hz_m"][0] == pytest.approx(det_fd, rel=1e-5)
        assert result["surface_rotation_included"] is True


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


# --------------------------------------------------------------------------- #
# delay_doppler_cell_samples                                                  #
# --------------------------------------------------------------------------- #

def _enu(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    return east, np.cross(up, east), up


def _backscatter_scenario(tx_azimuth_deg):
    """ECEF Tx/Rx at 600 km over (60 N, 5 E), Rx heading north; the ship sits at the BRP of the Tx -> Rx line."""
    east, north, up = _enu(60.0, 5.0)
    rx = np.array(pymap3d.geodetic2ecef(60.0, 5.0, 0.0)) + 600e3 * up
    v_rx = 7.1e3 * north
    el, az = np.radians(50.0), np.radians(tx_azimuth_deg)
    dir_tx = np.cos(el) * (np.cos(az) * north + np.sin(az) * east) + np.sin(el) * up
    tx = rx + 2.2e7 * dir_tx
    v_tx = 3.0e3 * np.cross(dir_tx, up) / np.linalg.norm(np.cross(dir_tx, up))
    los = (rx - tx) / np.linalg.norm(rx - tx)
    pts = rx + np.linspace(0.0, 3.0e6, 300001)[:, None] * los
    lat, lon, alt = pymap3d.ecef2geodetic(pts[:, 0], pts[:, 1], pts[:, 2])
    k = int(np.argmax(alt < 0.0))
    ship = np.array(pymap3d.geodetic2ecef(lat[k], lon[k], 0.0))
    specular, _, _ = object.__new__(Reflection)._find_specular_point_ellipse_ecef(tx, rx)
    return tx, v_tx, rx, v_rx, specular, ship


def _first_order_area(tx, v_tx, rx, v_rx, point):
    lat, lon, _ = pymap3d.ecef2geodetic(*point, deg=False)
    normal = np.array([[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
    geom = build_bistatic_geometry(tx.reshape(3, 1), rx.reshape(3, 1), point.reshape(1, 3), normal)
    return float(delay_doppler_cell_area(geom, v_tx.reshape(3, 1), v_rx.reshape(3, 1), WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH)["uncapped_cell_area_m2"][0])


def _cell(tx, v_tx, rx, v_rx, specular, point):
    return delay_doppler_cell_samples(tx, v_tx, rx, v_rx, specular, point, WAVELENGTH_L1, CHIP_LENGTH_CA, T_COH, 500.0, 250.0)


class TestDelayDopplerCellSamples:
    def test_matches_first_order_area_away_from_degeneracy(self):
        # Broadside look: the ship's part of the cell is the first-order delta_L delta_f / |grad L x grad f_D|.
        tx, v_tx, rx, v_rx, specular, ship = _backscatter_scenario(90.0)
        cell = _cell(tx, v_tx, rx, v_rx, specular, ship)
        np.testing.assert_allclose(cell["point_segment_area_m2"], _first_order_area(tx, v_tx, rx, v_rx, ship), rtol=0.03)

    def test_ambiguity_twin_is_a_separate_part(self):
        tx, v_tx, rx, v_rx, specular, ship = _backscatter_scenario(90.0)
        cell = _cell(tx, v_tx, rx, v_rx, specular, ship)
        assert len(cell["segment_areas_m2"]) == 2
        assert cell["cell_area_m2"] > 1.5 * cell["point_segment_area_m2"]
        np.testing.assert_allclose(cell["area_m2"].sum(), cell["cell_area_m2"], rtol=1e-12)

    def test_bounded_where_first_order_diverges(self):
        # Ship on the Rx ground track: grad L || grad f_D, so the first-order area blows up; the contour
        # curvature bounds the integrated cell.
        tx, v_tx, rx, v_rx, specular, ship = _backscatter_scenario(179.0)
        cell = _cell(tx, v_tx, rx, v_rx, specular, ship)
        first_order = _first_order_area(tx, v_tx, rx, v_rx, ship)
        assert 0.0 < cell["cell_area_m2"] < 0.1 * first_order

    def test_cell_at_specular_point(self):
        # At the specular point the delay band is a disk (the first chip zone) cut by the Doppler bin.
        tx, v_tx, rx, v_rx, specular, _ = _backscatter_scenario(90.0)
        cell = _cell(tx, v_tx, rx, v_rx, specular, specular)
        assert cell["specular_in_cell"]
        assert np.isfinite(cell["cell_area_m2"]) and cell["cell_area_m2"] > 0.0
        first_order = _first_order_area(tx, v_tx, rx, v_rx, specular)
        assert not np.isfinite(first_order) or first_order > 1e3 * cell["cell_area_m2"]
