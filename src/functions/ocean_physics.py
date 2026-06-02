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
