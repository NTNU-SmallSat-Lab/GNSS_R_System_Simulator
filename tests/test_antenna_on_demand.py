"""Diagnostic analytic array-factor helper tests.

Validates the closed-form AF added to RxAntenna against:
  - Peak normalization at the steering direction (= 0 dB)
  - First-null position of a uniform half-wavelength ULA
  - The closed-form ratio sin(N psi/2)/sin(psi/2), at boresight and steered
  - Vectorized input/output shape
  - Missing-geometry guard

This helper is intentionally kept out of the active fixed-nadir gain path; the golden
gain test exercises JSON-pattern sampling bit-for-bit.
"""

import numpy as np
import pytest

from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna


def _make_antenna(elements_x: int = 4, elements_y: int = 4, d_x: float = 0.1, d_y: float = 0.1) -> RxAntenna:
    antenna = RxAntenna.__new__(RxAntenna)
    antenna._array_geometry = {
        "elements_along_x": elements_x,
        "elements_along_y": elements_y,
        "d_x": d_x,
        "d_y": d_y,
    }
    return antenna


def _closed_form_af_db(
    elements_along_x: int,
    elements_along_y: int,
    d_x: float,
    d_y: float,
    freq_hz: float,
    beta_x: float,
    beta_y: float,
    theta_rad: float,
    phi_rad: float,
) -> float:
    """Reference closed form derived independently from the implementation under test."""
    wavelength = Constants.SPEED_OF_LIGHT / freq_hz
    k = 2.0 * np.pi / wavelength
    psi_x = k * d_x * np.sin(theta_rad) * np.cos(phi_rad) + beta_x
    psi_y = k * d_y * np.sin(theta_rad) * np.sin(phi_rad) + beta_y
    af_x = abs(np.sin(elements_along_x * psi_x / 2.0) / np.sin(psi_x / 2.0))
    af_y = abs(np.sin(elements_along_y * psi_y / 2.0) / np.sin(psi_y / 2.0))
    return 20.0 * np.log10((af_x * af_y) / (elements_along_x * elements_along_y) + 1e-12)


class TestPeakAtSteeringDirection:
    @pytest.mark.parametrize(
        "az_deg, el_deg",
        [
            (0.0, 0.0),
            (45.0, 10.0),
            (270.0, 25.0),
            (180.0, 30.0),
        ],
    )
    def test_steering_direction_gives_0db(self, az_deg: float, el_deg: float) -> None:
        antenna = _make_antenna()
        db = antenna.compute_array_factor_db_on_demand(
            theta_deg=el_deg,
            phi_deg=az_deg,
            steering_az_deg=az_deg,
            steering_el_deg=el_deg,
            freq_mhz=1575.42,
        )
        assert db == pytest.approx(0.0, abs=1e-9)


class TestNullPosition:
    def test_first_null_of_4element_half_wavelength_ula(self) -> None:
        # Single-row 4-element ULA along x at d = lambda/2. With boresight steering and phi=0,
        # the AF along x is sin(N psi/2)/sin(psi/2); the first null is at N psi/2 = pi, i.e.
        # k d sin(theta) = 2 pi / N -> sin(theta) = lambda / (N d) = 2/N for d = lambda/2.
        freq_mhz = 1575.42
        wavelength = Constants.SPEED_OF_LIGHT / (freq_mhz * 1e6)
        antenna = _make_antenna(elements_x=4, elements_y=1, d_x=wavelength / 2.0, d_y=wavelength / 2.0)
        theta_null_deg = float(np.degrees(np.arcsin(2.0 / 4.0)))
        db = antenna.compute_array_factor_db_on_demand(
            theta_deg=theta_null_deg,
            phi_deg=0.0,
            steering_az_deg=0.0,
            steering_el_deg=0.0,
            freq_mhz=freq_mhz,
        )
        assert db < -100.0


class TestClosedFormAgreement:
    def test_off_axis_boresight_steering_matches_closed_form(self) -> None:
        freq_mhz = 1575.42
        antenna = _make_antenna(elements_x=4, elements_y=4, d_x=0.1, d_y=0.1)

        theta_deg, phi_deg = 12.0, 47.0
        expected = _closed_form_af_db(
            elements_along_x=4,
            elements_along_y=4,
            d_x=0.1,
            d_y=0.1,
            freq_hz=freq_mhz * 1e6,
            beta_x=0.0,
            beta_y=0.0,
            theta_rad=np.deg2rad(theta_deg),
            phi_rad=np.deg2rad(phi_deg),
        )
        actual = antenna.compute_array_factor_db_on_demand(
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            steering_az_deg=0.0,
            steering_el_deg=0.0,
            freq_mhz=freq_mhz,
        )
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_steered_off_axis_matches_closed_form(self) -> None:
        freq_mhz = 1575.42
        freq_hz = freq_mhz * 1e6
        antenna = _make_antenna(elements_x=4, elements_y=4, d_x=0.1, d_y=0.1)

        steering_az_deg, steering_el_deg = 60.0, 15.0
        theta_deg, phi_deg = 20.0, 30.0

        wavelength = Constants.SPEED_OF_LIGHT / freq_hz
        k = 2.0 * np.pi / wavelength
        beta_x = -k * 0.1 * np.sin(np.deg2rad(steering_el_deg)) * np.cos(np.deg2rad(steering_az_deg))
        beta_y = -k * 0.1 * np.sin(np.deg2rad(steering_el_deg)) * np.sin(np.deg2rad(steering_az_deg))
        expected = _closed_form_af_db(
            elements_along_x=4,
            elements_along_y=4,
            d_x=0.1,
            d_y=0.1,
            freq_hz=freq_hz,
            beta_x=beta_x,
            beta_y=beta_y,
            theta_rad=np.deg2rad(theta_deg),
            phi_rad=np.deg2rad(phi_deg),
        )
        actual = antenna.compute_array_factor_db_on_demand(
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            steering_az_deg=steering_az_deg,
            steering_el_deg=steering_el_deg,
            freq_mhz=freq_mhz,
        )
        assert actual == pytest.approx(expected, abs=1e-9)


class TestVectorizedInputs:
    def test_array_inputs_broadcast(self) -> None:
        # _array_factor_db_at_direction must accept ndarrays of theta and phi; result shape follows numpy.
        elements_along_x, elements_along_y, d_x, d_y = 4, 4, 0.1, 0.1
        freq_hz = 1575.42 * 1e6
        theta_rad = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0]))
        phi_rad = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0]))

        result = RxAntenna._array_factor_db_at_direction(
            elements_along_x=elements_along_x,
            elements_along_y=elements_along_y,
            d_x=d_x,
            d_y=d_y,
            freq_hz=freq_hz,
            beta_x=0.0,
            beta_y=0.0,
            theta_rad=theta_rad,
            phi_rad=phi_rad,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == theta_rad.shape
        # Peak at theta=0
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        # AF decreases moving off boresight in this regime (no grating lobes for d=0.1m at 1575 MHz)
        assert result[1] < result[0]
        assert result[2] < result[0]


class TestMissingGeometryRaises:
    def test_missing_geometry_raises(self) -> None:
        antenna = RxAntenna.__new__(RxAntenna)
        antenna._array_geometry = None
        with pytest.raises(ValueError, match="Array geometry not configured"):
            antenna.compute_array_factor_db_on_demand(
                theta_deg=10.0,
                phi_deg=0.0,
                steering_az_deg=0.0,
                steering_el_deg=0.0,
                freq_mhz=1575.42,
            )
