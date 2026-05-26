"""Unit tests for ocean_physics helpers (Klein-Swift permittivity, Fresnel, Cox-Munk)."""

import numpy as np
import pytest
from scipy import integrate

from functions.ocean_physics import (
    circular_pol_reflectivities,
    cox_munk_slope_pdf,
    cox_munk_slope_variances,
    fresnel_coefficients,
    seawater_permittivity,
)


class TestSeawaterPermittivity:
    def test_l_band_reference_value(self):
        # At 1.575 GHz, 15 degC, 35 psu the published Klein-Swift result is roughly
        # eps_r ~ 73 - j 60 (within a few units).
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        assert 60.0 < eps_r.real < 90.0
        assert -80.0 < eps_r.imag < -40.0

    def test_imaginary_part_negative_under_exp_jwt_convention(self):
        # Loss tangent => negative imaginary part.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        assert eps_r.imag < 0.0

    def test_pure_water_limit_no_conductivity_term(self):
        # At S=0 the conductivity contribution must vanish.
        eps_pure = seawater_permittivity(1.57542e9, 15.0, 0.0)
        # Without salinity, imaginary part comes only from Debye loss -> small magnitude.
        assert abs(eps_pure.imag) < 10.0

    def test_higher_temperature_lowers_static_constant(self):
        # eps_s decreases with temperature (water becomes more polar at low T).
        eps_cold = seawater_permittivity(1.57542e9, 0.0, 35.0)
        eps_warm = seawater_permittivity(1.57542e9, 30.0, 35.0)
        assert eps_cold.real > eps_warm.real


class TestFresnelCoefficients:
    def test_normal_incidence_R_v_equals_minus_R_h(self):
        # Under the Fresnel sign convention used here, R_v(0) = -R_h(0).
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        R_v, R_h = fresnel_coefficients(0.0, eps_r)
        np.testing.assert_allclose(R_v, -R_h, rtol=1e-12)

    def test_normal_incidence_magnitude_consistent_with_sqrt_eps(self):
        # |R(0)| = |(1 - sqrt(eps)) / (1 + sqrt(eps))| with the sign convention used in fresnel_coefficients.
        # Independent of branch choice, |R(0)| must match this analytic form.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        R_v, _ = fresnel_coefficients(0.0, eps_r)
        sqrt_eps = np.sqrt(eps_r)
        expected = np.abs((1.0 - sqrt_eps) / (1.0 + sqrt_eps))
        np.testing.assert_allclose(np.abs(R_v), expected, rtol=1e-12)

    def test_grazing_R_v_and_R_h_approach_unity(self):
        # As theta -> 90 deg, |R| -> 1 for any lossy medium.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        R_v, R_h = fresnel_coefficients(np.radians(89.9), eps_r)
        assert np.abs(R_v) > 0.95
        assert np.abs(R_h) > 0.99


class TestCircularPolReflectivities:
    def test_normal_incidence_co_pol_vanishes(self):
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        R_RL_sq, R_RR_sq = circular_pol_reflectivities(0.0, eps_r)
        # |R_RR|^2 = |0.5 (R_v - R_h)|^2 and at normal incidence R_v == R_h, so R_RR = 0.
        np.testing.assert_allclose(R_RR_sq, 0.0, atol=1e-20)
        assert R_RL_sq > 0.5  # cross-pol carries the full reflectivity at normal incidence

    def test_grazing_co_pol_grows_relative_to_cross_pol(self):
        # Towards Brewster / grazing, |R_v| dips while |R_h| stays near 1, so R_RR rises.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        _, R_RR_low = circular_pol_reflectivities(np.radians(10.0), eps_r)
        _, R_RR_high = circular_pol_reflectivities(np.radians(80.0), eps_r)
        assert R_RR_high > R_RR_low


class TestCoxMunkSlopeVariances:
    def test_zero_wind_only_crosswind_residual(self):
        sigma_u_sq, sigma_c_sq = cox_munk_slope_variances(0.0)
        np.testing.assert_allclose(sigma_u_sq, 0.0, atol=1e-12)
        np.testing.assert_allclose(sigma_c_sq, 0.003, rtol=1e-12)

    def test_classical_10_m_s_values(self):
        # Cox and Munk (1954) clean sea, U_10 = 10 m/s:
        #     sigma_u_sq = 3.16e-3 * 10  = 0.0316
        #     sigma_c_sq = 0.003 + 1.92e-3 * 10 = 0.0222
        sigma_u_sq, sigma_c_sq = cox_munk_slope_variances(10.0)
        np.testing.assert_allclose(sigma_u_sq, 0.0316, rtol=1e-9)
        np.testing.assert_allclose(sigma_c_sq, 0.0222, rtol=1e-9)


class TestCoxMunkSlopePdf:
    def test_peak_at_origin(self):
        sigma_u_sq, sigma_c_sq = cox_munk_slope_variances(7.0)
        pdf_origin = cox_munk_slope_pdf(0.0, 0.0, sigma_u_sq, sigma_c_sq)
        pdf_off = cox_munk_slope_pdf(0.05, 0.05, sigma_u_sq, sigma_c_sq)
        assert pdf_origin > pdf_off

    def test_integrates_to_unity(self):
        # Integrate the PDF over a wide enough slope range to capture the full mass.
        sigma_u_sq, sigma_c_sq = cox_munk_slope_variances(7.0)

        def integrand(s_u, s_c):
            return cox_munk_slope_pdf(s_u, s_c, sigma_u_sq, sigma_c_sq)

        # +/- 6 sigma is more than enough for a Gaussian.
        bound_u = 6.0 * np.sqrt(sigma_u_sq)
        bound_c = 6.0 * np.sqrt(sigma_c_sq)
        total, _ = integrate.dblquad(integrand, -bound_c, bound_c, -bound_u, bound_u)
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_rejects_non_positive_variances(self):
        with pytest.raises(ValueError):
            cox_munk_slope_pdf(0.0, 0.0, 0.0, 0.01)
        with pytest.raises(ValueError):
            cox_munk_slope_pdf(0.0, 0.0, 0.01, 0.0)
