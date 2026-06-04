"""Pure-function helpers for ocean surface bistatic scattering physics.

Includes:
    - Seawater complex permittivity (Klein and Swift, 1977).
    - Fresnel reflection coefficients for vertical / horizontal polarisation.
    - Circular-polarisation power reflectivities (RHCP -> LHCP cross-pol, RHCP -> RHCP co-pol).
    - Cox-Munk (1954) clean-sea slope variances and 2-D Gaussian slope PDF.

All functions accept numpy scalars or arrays and broadcast accordingly.
"""

from __future__ import annotations

import numpy as np

from object_definitions.Constants_def import Constants

EPS_0 = 8.854187817e-12  # vacuum permittivity [F/m]


def seawater_permittivity(frequency_hz, temperature_c, salinity_psu):
    """Complex relative permittivity of seawater (Klein and Swift, 1977).

    INPUTS:
        frequency_hz:   [scalar or array] Carrier frequency [Hz]
        temperature_c:  [scalar]          Sea temperature [degC] (valid range ~ -2 to 30)
        salinity_psu:   [scalar]          Salinity [practical salinity units / per mille] (valid ~ 0 to 40)

    OUTPUT:
        eps_r: [complex] Relative permittivity (real - j * imag). The imaginary part is negative
                         under the exp(+j omega t) convention used here (both Debye relaxation and
                         ionic conductivity are losses and add to the same negative-imaginary sign).
    """
    f_hz = np.asarray(frequency_hz, dtype=float)
    T = float(temperature_c)
    S = float(salinity_psu)

    # Static dielectric constant
    eps_s_pure = 87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3    # pure water (Klein-Swift eq. 12)
    a_TS = 1.0 + 1.613e-5 * T * S - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3    # salinity correction
    eps_s = eps_s_pure * a_TS

    # Debye relaxation time
    tau_pure = 1.768e-11 - 6.086e-13 * T + 1.104e-14 * T**2 - 8.111e-17 * T**3    # pure water [s]
    b_TS = 1.0 + 2.282e-5 * T * S - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3    # salinity correction
    tau = tau_pure * b_TS

    eps_inf = 4.9                                                       # high-frequency limit of seawater permittivity

    # Ionic conductivity (S/m)
    delta = 25.0 - T                                                    # delta = 25 - T [degC] (Klein-Swift convention)
    beta = (
        2.0333e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2
        - S * (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2)
    )                                                                   # temperature dependence of conductivity
    sigma_25 = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S**2 - 1.28205e-7 * S**3)    # conductivity at 25 degC
    sigma_ionic = sigma_25 * np.exp(-delta * beta)                      # conductivity at T

    omega = 2.0 * np.pi * f_hz                                          # angular frequency [rad/s]
    eps_r = eps_inf + (eps_s - eps_inf) / (1.0 + 1j * omega * tau) - 1j * sigma_ionic / (omega * EPS_0)    # Debye + ionic loss
    return eps_r


def fresnel_coefficients(theta_inc_rad, eps_r):
    """Fresnel reflection coefficients for vertical (TM) and horizontal (TE) polarisation.

    INPUTS:
        theta_inc_rad: [scalar or array]  Incidence angle from surface normal [rad]
        eps_r:         [complex]          Relative permittivity of the lower medium (seawater)

    OUTPUT:
        R_v, R_h: complex Fresnel coefficients for vertical and horizontal polarisation.
    """
    cos_t = np.cos(theta_inc_rad)                                       # cos(theta_inc)
    sin_sq = np.sin(theta_inc_rad) ** 2                                 # sin^2(theta_inc)
    with np.errstate(invalid="ignore", divide="ignore"):                # silence NaN propagation noise
        sqrt_term = np.sqrt(eps_r - sin_sq)                             # sqrt(eps_r - sin^2 theta) = n_2 cos(theta_t)
        R_h = (cos_t - sqrt_term) / (cos_t + sqrt_term)                 # TE (horizontal / s-pol) Fresnel
        R_v = (eps_r * cos_t - sqrt_term) / (eps_r * cos_t + sqrt_term) # TM (vertical / p-pol) Fresnel
    return R_v, R_h


def circular_pol_reflectivities(theta_inc_rad, eps_r):
    """Power reflectivities for circular polarisation at the sea surface.

    Uses the Fresnel sign convention R_v(0) = -R_h(0) (Ulaby, Moore and Fung 1982);
    consequently:
        R_RL = 0.5 * (R_h - R_v)   cross-pol  (RHCP -> LHCP, dominant at low incidence)
        R_RR = 0.5 * (R_h + R_v)   co-pol     (RHCP -> RHCP, grows toward grazing)

    Returns:
        R_RL_sq, R_RR_sq: |R_RL|^2 and |R_RR|^2 power reflectivities.
    """
    R_v, R_h = fresnel_coefficients(theta_inc_rad, eps_r)
    R_RL = 0.5 * (R_h - R_v)                                            # RHCP -> LHCP (cross-pol)
    R_RR = 0.5 * (R_h + R_v)                                            # RHCP -> RHCP (co-pol)
    return np.abs(R_RL) ** 2, np.abs(R_RR) ** 2


def cox_munk_slope_variances(wind_speed_m_s):
    """Cox-Munk (1954) clean-sea slope variances at 10 m wind speed.

    Returns:
        sigma_u_sq: upwind slope variance  [dimensionless]
        sigma_c_sq: crosswind slope variance [dimensionless]
    """
    U = float(wind_speed_m_s)
    sigma_u_sq = 3.16e-3 * U                                            # upwind variance:   3.16e-3 * U_10
    sigma_c_sq = 0.003 + 1.92e-3 * U                                    # crosswind variance: 0.003 + 1.92e-3 * U_10
    return sigma_u_sq, sigma_c_sq


def cox_munk_slope_pdf(s_u, s_c, sigma_u_sq, sigma_c_sq):
    """2-D anisotropic Gaussian slope PDF in the wind-aligned (upwind, crosswind) frame.

    INPUTS:
        s_u, s_c:                [scalar or array] Slope components along upwind / crosswind axes
        sigma_u_sq, sigma_c_sq:  [scalar]          Slope variances from cox_munk_slope_variances

    OUTPUT:
        pdf: probability density at the given slope, same shape as s_u / s_c
    """
    if sigma_u_sq <= 0.0 or sigma_c_sq <= 0.0:
        raise ValueError("Slope variances must be positive")
    pdf_norm = 1.0 / (2.0 * np.pi * np.sqrt(sigma_u_sq * sigma_c_sq))   # 2-D Gaussian normalisation constant
    return pdf_norm * np.exp(-0.5 * (s_u**2 / sigma_u_sq + s_c**2 / sigma_c_sq))


def elfouhaily_omnidirectional_spectrum(wavenumber_rad_m, wind_speed_m_s, inverse_wave_age):
    """Omnidirectional sea-surface elevation spectrum S(k) (Elfouhaily et al., 1997).

    Unified long- plus short-wave wind-wave spectrum. The omnidirectional curvature
    (saturation) spectrum is B(k) = B_low(k) + B_high(k); this returns S(k) = B(k) / k^3.
    Unlike the Cox-Munk slope variances (which integrate the whole wave field), this resolves
    the spectral density at a single wavenumber -- the quantity Bragg backscatter needs at the
    resonant wavenumber k_Bragg = 2 k sin(theta).

    Reference: Elfouhaily, Chapron, Katsaros, Vandemark, "A unified directional spectrum for
    long and short wind-driven waves", J. Geophys. Res. 102(C7), 15781-15796, 1997.

    INPUTS:
        wavenumber_rad_m: [scalar or array] Spatial wavenumber k = 2 pi / wavelength [rad/m] (k > 0)
        wind_speed_m_s:   [scalar]          Wind speed at 10 m reference height U10 [m/s] (> 0)
        inverse_wave_age: [scalar]          Inverse wave age Omega_c [-] (0.84 fully developed, up to ~5 young seas)

    OUTPUT:
        S_k: [scalar or array] Omnidirectional elevation spectrum [m^3], same shape as wavenumber_rad_m.
    """
    k = np.asarray(wavenumber_rad_m, dtype=float)
    U10 = float(wind_speed_m_s)
    omega_c = float(inverse_wave_age)
    if U10 <= 0.0:
        raise ValueError(f"wind_speed_m_s must be > 0; got {U10}")
    if omega_c <= 0.0:
        raise ValueError(f"inverse_wave_age must be > 0; got {omega_c}")
    if np.any(k <= 0.0):
        raise ValueError("wavenumber_rad_m must be strictly positive")

    g = Constants.GRAVITY_ACCEL_M_S2                           # standard gravitational acceleration [m/s^2]
    km = Constants.WAVE_KM_RAD_M                               # wavenumber of minimum phase speed [rad/m]
    cm = Constants.WAVE_CM_M_S                                 # minimum phase speed (at km) [m/s]

    # Spectral peak and phase speeds
    k0 = g / U10**2                                            # k0 = g / U10^2 [rad/m]
    kp = k0 * omega_c**2                                       # peak wavenumber kp = k0 * Omega_c^2
    cp = np.sqrt(g / kp)                                       # phase speed at the peak cp = sqrt(g / kp)
    c_k = np.sqrt((g / k) * (1.0 + (k / km) ** 2))             # phase speed c(k), incl. capillary term

    # Friction velocity from a 10 m neutral drag law (Wu, 1982)
    drag_coeff = 1.0e-3 * (0.81 + 0.065 * U10)                 # neutral drag coefficient Cd(U10)
    u_star = U10 * np.sqrt(drag_coeff)                         # friction velocity u_* [m/s]

    # JONSWAP peak shape (shared by both branches)
    Lpm = np.exp(-1.25 * (kp / k) ** 2)                        # Pierson-Moskowitz shape exp(-5/4 (kp/k)^2)
    gamma = 1.7 if omega_c <= 1.0 else 1.7 + 6.0 * np.log10(omega_c)   # JONSWAP peak-enhancement base
    sigma = 0.08 * (1.0 + 4.0 * omega_c**-3)                   # JONSWAP peak width
    Gamma = np.exp(-((np.sqrt(k / kp) - 1.0) ** 2) / (2.0 * sigma**2)) # peak-enhancement exponent
    Jp = gamma**Gamma                                          # JONSWAP peak enhancement Jp = gamma^Gamma

    # Long-wave (gravity) curvature B_low
    alpha_p = 0.006 * omega_c**0.55                            # long-wave equilibrium parameter (Phillips-Kitaigorodskii)
    Fp = np.exp(-(omega_c / np.sqrt(10.0)) * (np.sqrt(k / kp) - 1.0))  # long-wave side effect
    B_low = 0.5 * alpha_p * (cp / c_k) * Lpm * Jp * Fp         # B_l = 0.5 alpha_p (cp/c) LPM Jp Fp

    # Short-wave (gravity-capillary) curvature B_high
    if u_star <= cm:
        alpha_m = 0.01 * (1.0 + np.log(u_star / cm))           # light-wind branch
    else:
        alpha_m = 0.01 * (1.0 + 3.0 * np.log(u_star / cm))     # strong-wind branch
    alpha_m = max(alpha_m, 0.0)                                # Elfouhaily alpha_m underflows < 0 below ~2.6 m/s; clamp (no negative spectrum)
    Fm = np.exp(-0.25 * (k / km - 1.0) ** 2)                   # capillary side effect
    B_high = 0.5 * alpha_m * (cm / c_k) * Lpm * Jp * Fm        # B_h = 0.5 alpha_m (cm/c) LPM Jp Fm

    return (B_low + B_high) / k**3                             # S(k) = B(k) / k^3 [m^3]


def bragg_scattering_coefficients(theta_inc_rad, eps_r):
    """First-order small-perturbation (Bragg) scattering coefficients for H and V polarisation.

    These are the polarisation amplitudes in the SPM/Bragg backscatter cross-section
    sigma0_pp ~ k^4 cos^4(theta) |g_pp|^2 W(2 k sin theta). They are NOT the Fresnel
    reflection coefficients: they reduce to a common value at normal incidence
    (g_HH(0) = g_VV(0) = (eps - 1) / (1 + sqrt(eps))^2), and |g_VV| exceeds |g_HH| at
    oblique incidence for a high-permittivity surface such as seawater.

    Reference: Valenzuela (1978); Ulaby, Moore and Fung, "Microwave Remote Sensing" Vol. II.

    INPUTS:
        theta_inc_rad: [scalar or array]  Incidence angle from the surface normal [rad]
        eps_r:         [complex]          Relative permittivity of the lower medium (seawater)

    OUTPUT:
        g_hh, g_vv: complex Bragg scattering coefficients for horizontal and vertical polarisation.
    """
    cos_t = np.cos(theta_inc_rad)                                      # cos(theta_inc)
    sin_sq = np.sin(theta_inc_rad) ** 2                                # sin^2(theta_inc)
    with np.errstate(invalid="ignore", divide="ignore"):               # silence NaN propagation noise
        sqrt_term = np.sqrt(eps_r - sin_sq)                            # sqrt(eps_r - sin^2 theta)
        g_hh = (eps_r - 1.0) / (cos_t + sqrt_term) ** 2               # HH Bragg amplitude
        g_vv = (eps_r - 1.0) * ((eps_r - 1.0) * sin_sq + eps_r) / (eps_r * cos_t + sqrt_term) ** 2  # VV Bragg amplitude
    return g_hh, g_vv


def bragg_backscatter_nrcs(theta_inc_rad, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age):
    """First-order Bragg (small-perturbation) ocean backscatter NRCS for circular polarisation.

    Resonant scattering from the wind-driven ripple at the Bragg wavenumber k_B = 2 k sin(theta),
    with k = 2 pi / wavelength. This is the return the geometric-optics glistening model misses at
    oblique incidence (no facets tilted enough to retroreflect) and that dominates ocean backscatter
    beyond ~20 deg incidence.

        sigma0_pp = 8 pi k^4 cos^4(theta) |g_pp|^2 W_iso(k_B)

    with the isotropic 2-D spectrum W_iso(K) = S(K) / (2 pi K) from the Elfouhaily S(k), and the
    circular amplitudes formed from the linear Bragg coefficients:
        g_RL = (g_HH + g_VV) / 2   RHCP -> LHCP, handedness flipped  (carries all power at nadir)
        g_RR = (g_HH - g_VV) / 2   RHCP -> RHCP, handedness preserved (vanishes at nadir)

    The 8 pi prefactor is the textbook SPM constant; combined with the Elfouhaily spectrum
    normalisation it places sigma0_RL at the published L-band level (~ -21 dB at 30 deg, 7 m/s) with
    no free scaling -- the absolute level is pinned by TestBraggBackscatterNrcs, not a fudge factor.

    SPM validity holds for moderate incidence (~10..70 deg); near nadir k_B -> 0, the Bragg return
    vanishes and the geometric-optics term owns the geometry, so this returns 0 there.

    Reference: Valenzuela (1978); Ulaby, Moore and Fung, "Microwave Remote Sensing" Vol. II.

    INPUTS:
        theta_inc_rad:    [scalar or array]  Incidence angle from the surface normal [rad]
        wavelength_m:     [float]            Radar wavelength [m]
        eps_r:            [complex]          Seawater relative permittivity
        wind_speed_m_s:   [scalar]           Wind speed at 10 m [m/s]
        inverse_wave_age: [scalar]           Inverse wave age Omega_c [-]

    OUTPUT:
        sigma0_RL, sigma0_RR: RHCP->LHCP (cross-pol) and RHCP->RHCP (co-pol) NRCS [dimensionless],
                              same shape as theta_inc_rad; 0 at / below nadir.
    """
    theta = np.asarray(theta_inc_rad, dtype=float)
    k_radar = 2.0 * np.pi / wavelength_m                               # radar wavenumber k = 2 pi / lambda [rad/m]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    k_bragg = 2.0 * k_radar * sin_t                                    # resonant ocean wavenumber k_B = 2 k sin(theta)

    # Isotropic 2-D roughness spectrum W_iso(K) = S(K) / (2 pi K). At/below nadir k_B -> 0 (no resonant
    # ripple): evaluate S on a safe-positive wavenumber, then zero W_iso where k_B is not positive.
    positive_kb = k_bragg > 0.0
    k_safe = np.where(positive_kb, k_bragg, 1.0)
    S_bragg = elfouhaily_omnidirectional_spectrum(k_safe, wind_speed_m_s, inverse_wave_age)
    W_iso = np.where(positive_kb, S_bragg / (2.0 * np.pi * k_safe), 0.0)   # [m^4]

    g_hh, g_vv = bragg_scattering_coefficients(theta, eps_r)
    g_RL = 0.5 * (g_hh + g_vv)                                         # RHCP -> LHCP amplitude (handedness flip)
    g_RR = 0.5 * (g_hh - g_vv)                                         # RHCP -> RHCP amplitude (preserved)

    prefactor = 8.0 * np.pi * k_radar**4 * cos_t**4 * W_iso            # 8 pi k^4 cos^4(theta) W_iso
    sigma0_RL = prefactor * np.abs(g_RL) ** 2
    sigma0_RR = prefactor * np.abs(g_RR) ** 2
    return sigma0_RL, sigma0_RR
