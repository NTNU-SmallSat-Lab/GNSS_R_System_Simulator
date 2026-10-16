"""Unit tests for ocean_physics helpers (Klein-Swift permittivity, Fresnel, Cox-Munk)."""

import numpy as np
import pytest
from scipy import integrate

from functions.ocean_physics import (
    bragg_backscatter_nrcs,
    bragg_scattering_coefficients,
    circular_pol_reflectivities,
    cox_munk_slope_pdf,
    cox_munk_slope_variances,
    elfouhaily_omnidirectional_spectrum,
    fresnel_coefficients,
    large_scale_slope_variances,
    seawater_permittivity,
    two_scale_backscatter_nrcs,
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

    def test_high_frequency_limit_approaches_eps_inf(self):
        # Debye limit: as omega*tau >> 1 and the ionic-conductivity term -> 0, the Debye pole
        # collapses and eps -> eps_inf = 4.9. A structural check on the relaxation form, independent
        # of the L-band coefficients. (1e13 Hz is outside the model's physical domain but the formula
        # limit must still hold.)
        eps_r = seawater_permittivity(1.0e13, 15.0, 35.0)
        np.testing.assert_allclose(eps_r.real, 4.9, atol=0.2)
        assert abs(eps_r.imag) < 0.5

    def test_real_part_decreases_monotonically_with_frequency(self):
        # Across the relaxation the real part falls from the static value toward eps_inf.
        freqs = np.array([1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12])
        eps_real = np.array([seawater_permittivity(f, 15.0, 35.0).real for f in freqs])
        assert np.all(np.diff(eps_real) < 0.0), f"eps_real not monotonically decreasing: {eps_real}"


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

    def test_passivity_reflection_magnitude_at_most_unity(self):
        # Energy conservation: a passive lossy half-space cannot reflect more power than is incident,
        # so |R_v|, |R_h| <= 1 at every incidence. Guards against a sign / branch error in the
        # Fresnel form that the limiting-case tests would miss between 0 and grazing.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        thetas = np.radians(np.linspace(0.0, 89.0, 90))
        R_v, R_h = fresnel_coefficients(thetas, eps_r)
        assert np.all(np.abs(R_v) <= 1.0 + 1e-12)
        assert np.all(np.abs(R_h) <= 1.0 + 1e-12)


class TestCircularPolReflectivities:
    def test_normal_incidence_co_pol_vanishes(self):
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        R_RL_sq, R_RR_sq = circular_pol_reflectivities(0.0, eps_r)
        # R_RR = 0.5 (R_h + R_v), and this Fresnel basis has R_v(0) = -R_h(0).
        np.testing.assert_allclose(R_RR_sq, 0.0, atol=1e-20)
        assert R_RL_sq > 0.5  # cross-pol carries the full reflectivity at normal incidence

    def test_grazing_co_pol_grows_relative_to_cross_pol(self):
        # Towards Brewster / grazing, |R_v| dips while |R_h| stays near 1, so R_RR rises.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        _, R_RR_low = circular_pol_reflectivities(np.radians(10.0), eps_r)
        _, R_RR_high = circular_pol_reflectivities(np.radians(80.0), eps_r)
        assert R_RR_high > R_RR_low

    def test_passivity_total_reflected_power_at_most_unity(self):
        # Energy conservation: the total reflected circular power (cross + co) equals
        # 0.5 (|R_h|^2 + |R_v|^2), which must stay <= 1 at every incidence for a passive surface.
        eps_r = seawater_permittivity(1.57542e9, 15.0, 35.0)
        thetas = np.radians(np.linspace(0.0, 89.0, 90))
        R_RL_sq, R_RR_sq = circular_pol_reflectivities(thetas, eps_r)
        assert np.all(R_RL_sq + R_RR_sq <= 1.0 + 1e-12)


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


class TestElfouhailySpectrum:
    # L-band Bragg ocean wavenumbers K_B = 2 k_radar sin(theta) at L1 (k_radar ~ 33 rad/m):
    #   theta = 30 deg -> ~33 rad/m,   theta = 45 deg -> ~47 rad/m.
    K_BRAGG_L1_30DEG = 33.0
    K_BRAGG_L1_45DEG = 47.0
    K_CAPILLARY = 200.0   # well above the L1 Bragg band; capillary branch dominates, wind response is clean

    def test_positive_and_finite_across_spectrum(self):
        k = np.logspace(-1, 3, 300)   # 0.1 .. 1000 rad/m
        S = elfouhaily_omnidirectional_spectrum(k, wind_speed_m_s=7.0, inverse_wave_age=0.84)
        assert np.all(np.isfinite(S))
        assert np.all(S > 0.0)

    def test_capillary_density_increases_with_wind(self):
        # At the capillary scale the short-wave branch dominates, so density rises monotonically
        # with wind. (At the L1 Bragg scale this is deliberately NOT asserted -- the short-gravity
        # tail competes with the capillary part and the omnidirectional response is non-monotonic.)
        S_calm = elfouhaily_omnidirectional_spectrum(self.K_CAPILLARY, 5.0, 0.84)
        S_mod = elfouhaily_omnidirectional_spectrum(self.K_CAPILLARY, 10.0, 0.84)
        S_high = elfouhaily_omnidirectional_spectrum(self.K_CAPILLARY, 15.0, 0.84)
        assert S_calm < S_mod < S_high

    def test_bragg_scale_curvature_order_of_magnitude(self):
        # Saturation/curvature B(k) = k^3 S(k) sits in the ~1e-3 equilibrium band at the L1 Bragg scales.
        for k_bragg in (self.K_BRAGG_L1_30DEG, self.K_BRAGG_L1_45DEG):
            S = elfouhaily_omnidirectional_spectrum(k_bragg, 7.0, 0.84)
            curvature = k_bragg**3 * S
            assert 1e-4 < curvature < 1e-1, f"curvature {curvature:.2e} out of band at k={k_bragg}"

    def test_array_matches_scalar(self):
        k = np.array([1.0, 10.0, 47.0, 200.0])
        S_arr = elfouhaily_omnidirectional_spectrum(k, 7.0, 0.84)
        S_scalar = np.array([elfouhaily_omnidirectional_spectrum(float(ki), 7.0, 0.84) for ki in k])
        np.testing.assert_allclose(S_arr, S_scalar, rtol=1e-12)
        assert S_arr.shape == k.shape

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            elfouhaily_omnidirectional_spectrum(33.0, 0.0, 0.84)    # zero wind
        with pytest.raises(ValueError):
            elfouhaily_omnidirectional_spectrum(33.0, 7.0, 0.0)     # zero inverse wave age
        with pytest.raises(ValueError):
            elfouhaily_omnidirectional_spectrum(-1.0, 7.0, 0.84)    # non-positive wavenumber


class TestBraggScatteringCoefficients:
    EPS_SEAWATER_L1 = seawater_permittivity(1.57542e9, 15.0, 35.0)

    def test_equal_at_normal_incidence(self):
        # By symmetry H and V are indistinguishable at normal incidence.
        g_hh, g_vv = bragg_scattering_coefficients(0.0, self.EPS_SEAWATER_L1)
        np.testing.assert_allclose(g_hh, g_vv, rtol=1e-12)

    def test_normal_incidence_closed_form(self):
        # g_pp(0) = (eps - 1) / (1 + sqrt(eps))^2.
        g_hh, _ = bragg_scattering_coefficients(0.0, self.EPS_SEAWATER_L1)
        expected = (self.EPS_SEAWATER_L1 - 1.0) / (1.0 + np.sqrt(self.EPS_SEAWATER_L1)) ** 2
        np.testing.assert_allclose(g_hh, expected, rtol=1e-12)

    def test_vv_exceeds_hh_at_oblique_incidence(self):
        # For a high-permittivity surface (seawater), |g_VV| > |g_HH| away from nadir.
        for theta_deg in (20.0, 40.0, 60.0):
            g_hh, g_vv = bragg_scattering_coefficients(np.radians(theta_deg), self.EPS_SEAWATER_L1)
            assert np.abs(g_vv) > np.abs(g_hh), f"|g_VV| !> |g_HH| at {theta_deg} deg"

    def test_finite_across_incidence(self):
        thetas = np.radians(np.linspace(0.0, 80.0, 17))
        g_hh, g_vv = bragg_scattering_coefficients(thetas, self.EPS_SEAWATER_L1)
        assert np.all(np.isfinite(g_hh)) and np.all(np.isfinite(g_vv))

    def test_array_matches_scalar(self):
        thetas = np.radians(np.array([10.0, 30.0, 50.0]))
        g_hh_arr, g_vv_arr = bragg_scattering_coefficients(thetas, self.EPS_SEAWATER_L1)
        for i, th in enumerate(thetas):
            g_hh_s, g_vv_s = bragg_scattering_coefficients(float(th), self.EPS_SEAWATER_L1)
            np.testing.assert_allclose(g_hh_arr[i], g_hh_s, rtol=1e-12)
            np.testing.assert_allclose(g_vv_arr[i], g_vv_s, rtol=1e-12)
        assert g_hh_arr.shape == thetas.shape


class TestBraggBackscatterNrcs:
    LAMBDA_L1_M = 299792458.0 / 1.57542e9
    EPS_SEAWATER_L1 = seawater_permittivity(1.57542e9, 15.0, 35.0)

    def _sigma0(self, theta_deg, wind=7.0, pol="RL"):
        sigma0_rl, sigma0_rr = bragg_backscatter_nrcs(
            np.radians(theta_deg), self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, wind, 0.84
        )
        return float(sigma0_rl if pol == "RL" else sigma0_rr)

    def test_prefactor_matches_kirchhoff_small_roughness_limit(self):
        # Normal-incidence Kirchhoff NRCS for a Gaussian-correlated surface, (k^2 |R|^2 / pi) * int
        # [exp(-q_z^2 s^2 (1 - rho)) - exp(-q_z^2 s^2)] d^2r, tends to C k^4 |R|^2 Psi(0) for small s,
        # with Psi the height spectrum normalised to s^2 (as W_iso). Its large-s limit is the GO NRCS.
        k = 2.0 * np.pi / self.LAMBDA_L1_M
        s, corr_len = 1.0e-4, 5.0
        integrand = lambda r: (np.exp(-4 * k**2 * s**2 * (1 - np.exp(-(r**2) / corr_len**2))) - np.exp(-4 * k**2 * s**2)) * 2 * np.pi * r
        sigma0_ka = k**2 / np.pi * integrate.quad(integrand, 0.0, 20.0 * corr_len, limit=400)[0]
        psi_0 = s**2 * corr_len**2 / (4.0 * np.pi)                    # Psi(0) = s^2 / (2 pi)^2 * int rho d^2r
        prefactor_ka = sigma0_ka / (k**4 * psi_0)

        theta = np.radians(30.0)
        k_bragg = 2.0 * k * np.sin(theta)
        w_iso = elfouhaily_omnidirectional_spectrum(k_bragg, 7.0, 0.84) / (2.0 * np.pi * k_bragg)
        g_hh, g_vv = bragg_scattering_coefficients(theta, self.EPS_SEAWATER_L1)
        unit_prefactor = k**4 * np.cos(theta) ** 4 * abs(0.5 * (g_hh + g_vv)) ** 2 * w_iso
        np.testing.assert_allclose(self._sigma0(30.0) / unit_prefactor, prefactor_ka, rtol=1e-3)

    def test_representative_l_band_value_stays_in_broad_envelope(self):
        # Broad plausibility/regression bound, not measurement calibration: circular-polarisation
        # sigma0_RL at 30 deg / 7 m/s should remain in an oceanic L-band range.
        sigma0_db = 10.0 * np.log10(self._sigma0(30.0))
        assert -25.0 < sigma0_db < -17.0, f"anchor sigma0_RL = {sigma0_db:.2f} dB outside L-band band"

    def test_representative_incidence_angles_stay_in_broad_l_band_envelope(self):
        # Broad plausibility only. Aquarius used linear VV/HH whereas this model is circular RL, so
        # these limits cannot be interpreted as cross-validation.
        for th in (29.0, 38.0, 46.0):
            sigma0_db = 10.0 * np.log10(self._sigma0(th, wind=7.0))
            assert -32.0 < sigma0_db < -12.0, f"sigma0_RL {sigma0_db:.1f} dB at {th} deg outside L-band envelope"

    def test_decreases_with_incidence_across_aquarius_beams(self):
        # L-band ocean backscatter falls off monotonically from the 29 deg to the 46 deg Aquarius beam.
        vals = [self._sigma0(th, wind=7.0) for th in (29.0, 38.0, 46.0)]
        assert vals[0] > vals[1] > vals[2], f"sigma0_RL not decreasing across Aquarius beams: {vals}"

    def test_vanishes_at_nadir(self):
        # No resonant ripple at nadir (k_B -> 0); the geometric-optics term owns that geometry.
        sigma0_rl, sigma0_rr = bragg_backscatter_nrcs(0.0, self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, 7.0, 0.84)
        assert float(sigma0_rl) == 0.0 and float(sigma0_rr) == 0.0

    def test_cross_pol_dominates_co_pol(self):
        # RHCP -> LHCP (handedness flip) is the strong channel; co-pol is well below at oblique incidence.
        assert self._sigma0(30.0, pol="RL") > self._sigma0(30.0, pol="RR")

    def test_decreases_with_incidence_in_valid_range(self):
        # In the SPM-valid oblique band, backscatter falls off toward grazing.
        vals = [self._sigma0(th) for th in (30.0, 40.0, 50.0, 60.0)]
        assert all(np.diff(vals) < 0.0), f"sigma0_RL not decreasing with incidence: {vals}"

    def test_increases_with_wind(self):
        # Sea-state sensitivity: stronger wind -> brighter backscatter, checked at 40 deg where the
        # leading isotropic model is monotonic (at lower incidence the gravity-tail crossover mutes it).
        assert self._sigma0(40.0, wind=15.0) > self._sigma0(40.0, wind=5.0)

    def test_array_matches_scalar(self):
        thetas = np.radians(np.array([25.0, 35.0, 55.0]))
        rl_arr, rr_arr = bragg_backscatter_nrcs(thetas, self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, 7.0, 0.84)
        for i, th in enumerate(thetas):
            rl_s, rr_s = bragg_backscatter_nrcs(float(th), self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, 7.0, 0.84)
            np.testing.assert_allclose(rl_arr[i], rl_s, rtol=1e-12)
            np.testing.assert_allclose(rr_arr[i], rr_s, rtol=1e-12)
        assert rl_arr.shape == thetas.shape


class TestLargeScaleSlopeVariances:
    LAMBDA_L1_M = 299792458.0 / 1.57542e9

    def test_below_total_cox_munk_mss(self):
        # The large-scale (k < k_radar / 3) slopes are a subset of the full wave field, so their mss
        # must be below the total Cox-Munk mss.
        su_L, sc_L = large_scale_slope_variances(self.LAMBDA_L1_M, 7.0, 0.84)
        su_tot, sc_tot = cox_munk_slope_variances(7.0)
        assert su_L > 0.0 and sc_L > 0.0
        assert su_L + sc_L < su_tot + sc_tot

    def test_increases_with_wind(self):
        low = sum(large_scale_slope_variances(self.LAMBDA_L1_M, 5.0, 0.84))
        high = sum(large_scale_slope_variances(self.LAMBDA_L1_M, 15.0, 0.84))
        assert high > low

    def test_upwind_variance_exceeds_crosswind(self):
        su_L, sc_L = large_scale_slope_variances(self.LAMBDA_L1_M, 7.0, 0.84)
        assert su_L > sc_L


class TestTwoScaleBackscatterNrcs:
    LAMBDA_L1_M = 299792458.0 / 1.57542e9
    EPS_SEAWATER_L1 = seawater_permittivity(1.57542e9, 15.0, 35.0)

    def _ts(self, theta_deg, azimuth_deg=0.0, wind=7.0):
        rl, _ = two_scale_backscatter_nrcs(
            np.radians(theta_deg), np.radians(azimuth_deg), self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, wind, 0.84
        )
        return float(rl)

    def _first_order(self, theta_deg, wind=7.0):
        rl, _ = bragg_backscatter_nrcs(np.radians(theta_deg), self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, wind, 0.84)
        return float(rl)

    def test_lifts_first_order_at_low_incidence(self):
        # Tilt modulation raises the low-incidence backscatter above the un-tilted first-order Bragg.
        assert self._ts(29.0) > self._first_order(29.0)

    def test_tilt_lift_larger_at_low_incidence(self):
        # The tilt lift shrinks toward grazing, where few facets improve the local geometry.
        ratio_low = self._ts(29.0) / self._first_order(29.0)
        ratio_high = self._ts(46.0) / self._first_order(46.0)
        assert ratio_low > ratio_high > 1.0

    def test_upwind_brighter_than_crosswind(self):
        # Anisotropic large-scale slopes produce the upwind/crosswind asymmetry first-order Bragg lacks.
        assert self._ts(29.0, azimuth_deg=0.0) > self._ts(29.0, azimuth_deg=90.0)

    def test_stays_in_l_band_envelope_across_aquarius_beams(self):
        # The tilt-averaged sigma0 stays in the L-band oceanic envelope (now a few dB closer to VV).
        for th in (29.0, 38.0, 46.0):
            sigma0_db = 10.0 * np.log10(self._ts(th))
            assert -30.0 < sigma0_db < -10.0, f"two-scale sigma0_RL {sigma0_db:.1f} dB at {th} deg outside envelope"

    def test_decreases_with_incidence(self):
        vals = [self._ts(th) for th in (29.0, 38.0, 46.0)]
        assert vals[0] > vals[1] > vals[2], f"two-scale sigma0_RL not decreasing with incidence: {vals}"

    def test_no_divergence_leak_near_gate_at_high_wind(self):
        # Mirror-ward facets with local k_B < k_radar / 3 add no Bragg return, so steep large-scale
        # slopes at high wind cannot leak the near-nadir divergence: sigma0 stays below 0 dB and
        # decreases with incidence from 20 deg outward.
        thetas = (20.0, 22.0, 25.0, 30.0, 35.0, 45.0)
        for wind in (4.0, 7.0, 10.0, 13.0, 16.0):
            vals = [self._ts(th, wind=wind) for th in thetas]
            assert max(vals) < 1.0, f"two-scale sigma0_RL above 0 dB at {wind} m/s: {10 * np.log10(max(vals)):.1f} dB"
            assert all(a > b for a, b in zip(vals, vals[1:])), f"not decreasing with incidence at {wind} m/s"

    def test_array_matches_scalar(self):
        thetas = np.radians(np.array([25.0, 35.0, 45.0]))
        azis = np.radians(np.array([0.0, 45.0, 90.0]))
        rl_arr, rr_arr = two_scale_backscatter_nrcs(thetas, azis, self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, 7.0, 0.84)
        for i in range(thetas.size):
            rl_s, rr_s = two_scale_backscatter_nrcs(
                float(thetas[i]), float(azis[i]), self.LAMBDA_L1_M, self.EPS_SEAWATER_L1, 7.0, 0.84
            )
            np.testing.assert_allclose(rl_arr[i], rl_s, rtol=1e-12)
            np.testing.assert_allclose(rr_arr[i], rr_s, rtol=1e-12)
        assert rl_arr.shape == thetas.shape
