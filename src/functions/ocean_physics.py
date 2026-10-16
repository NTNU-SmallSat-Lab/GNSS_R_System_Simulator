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

    The imaginary part is negative under the ``exp(+j omega t)`` convention used here (both Debye
    relaxation and ionic conductivity are losses, on the same negative-imaginary sign).

    Parameters
    ----------
    frequency_hz : scalar or array
        Carrier frequency [Hz].
    temperature_c : scalar
        Sea temperature [degC] (valid range about -2 to 30).
    salinity_psu : scalar
        Salinity [PSU] (valid range about 0 to 40).

    Returns
    -------
    complex
        Relative permittivity (real minus j*imag).
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

    Parameters
    ----------
    theta_inc_rad : scalar or array
        Incidence angle from the surface normal [rad].
    eps_r : complex
        Relative permittivity of the lower medium (seawater).

    Returns
    -------
    tuple
        ``(R_v, R_h)`` complex Fresnel coefficients for vertical and horizontal polarisation.
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

    Uses the Fresnel sign convention ``R_v(0) = -R_h(0)`` (Ulaby, Moore and Fung 1982), so the cross-pol
    amplitude is ``R_RL = 0.5*(R_h - R_v)`` (RHCP to LHCP, dominant at low incidence) and the co-pol
    amplitude is ``R_RR = 0.5*(R_h + R_v)`` (RHCP to RHCP, grows toward grazing).

    Parameters
    ----------
    theta_inc_rad : scalar or array
        Incidence angle from the surface normal [rad].
    eps_r : complex
        Relative permittivity of seawater.

    Returns
    -------
    tuple
        ``(R_RL_sq, R_RR_sq)`` power reflectivities (the squared magnitudes).
    """
    R_v, R_h = fresnel_coefficients(theta_inc_rad, eps_r)
    R_RL = 0.5 * (R_h - R_v)                                            # RHCP -> LHCP (cross-pol)
    R_RR = 0.5 * (R_h + R_v)                                            # RHCP -> RHCP (co-pol)
    return np.abs(R_RL) ** 2, np.abs(R_RR) ** 2


def cox_munk_slope_variances(wind_speed_m_s):
    """Cox-Munk (1954) clean-sea slope variances at 10 m wind speed.

    Parameters
    ----------
    wind_speed_m_s : scalar
        Wind speed at 10 m [m/s].

    Returns
    -------
    tuple
        ``(sigma_u_sq, sigma_c_sq)`` upwind and crosswind slope variances [dimensionless].
    """
    U = float(wind_speed_m_s)
    sigma_u_sq = 3.16e-3 * U                                            # upwind variance:   3.16e-3 * U_10
    sigma_c_sq = 0.003 + 1.92e-3 * U                                    # crosswind variance: 0.003 + 1.92e-3 * U_10
    return sigma_u_sq, sigma_c_sq


def cox_munk_slope_pdf(s_u, s_c, sigma_u_sq, sigma_c_sq):
    """2-D anisotropic Gaussian slope PDF in the wind-aligned (upwind, crosswind) frame.

    Parameters
    ----------
    s_u, s_c : scalar or array
        Slope components along the upwind / crosswind axes.
    sigma_u_sq, sigma_c_sq : scalar
        Slope variances from :func:`cox_munk_slope_variances`.

    Returns
    -------
    scalar or array
        Probability density at the given slope.
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

    Parameters
    ----------
    wavenumber_rad_m : scalar or array
        Spatial wavenumber ``k = 2 pi / wavelength`` [rad/m] (> 0).
    wind_speed_m_s : scalar
        Wind speed at 10 m reference height U10 [m/s] (> 0).
    inverse_wave_age : scalar
        Inverse wave age Omega_c [-] (0.84 fully developed, up to about 5 for young seas).

    Returns
    -------
    scalar or array
        Omnidirectional elevation spectrum S(k) [m^3].
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
    alpha_p = 0.006 * np.sqrt(omega_c)                         # long-wave equilibrium parameter alpha_p = 6e-3 sqrt(Omega_c) (Elfouhaily eq. 34)
    Fp = np.exp(-(omega_c / np.sqrt(10.0)) * (np.sqrt(k / kp) - 1.0))  # long-wave side effect
    B_low = 0.5 * alpha_p * (cp / c_k) * Lpm * Jp * Fp         # B_l = 0.5 alpha_p (cp/c) LPM Jp Fp

    # Short-wave (gravity-capillary) curvature B_high
    if u_star <= cm:
        alpha_m = 0.01 * (1.0 + np.log(u_star / cm))           # light-wind branch
    else:
        alpha_m = 0.01 * (1.0 + 3.0 * np.log(u_star / cm))     # strong-wind branch
    alpha_m = max(alpha_m, 0.0)                                # Elfouhaily alpha_m underflows < 0 below ~2.6 m/s; clamp (no negative spectrum)
    Fm = np.exp(-0.25 * (k / km - 1.0) ** 2)                   # capillary side effect (Elfouhaily eq. 41)
    # Deliberate deviation from eq. 40: the paper's B_h has no low-k cutoff, so the capillary branch
    # would leak spectral density below the long-wave peak; Lpm suppresses that unphysical tail.
    B_high = 0.5 * alpha_m * (cm / c_k) * Lpm * Fm             # B_h = 0.5 alpha_m (cm/c) LPM Fm

    return (B_low + B_high) / k**3                             # S(k) = B(k) / k^3 [m^3]


def bragg_scattering_coefficients(theta_inc_rad, eps_r):
    """First-order small-perturbation (Bragg) scattering coefficients for H and V polarisation.

    These are the polarisation amplitudes in the SPM/Bragg backscatter cross-section
    ``sigma0_pp ~ k^4 cos^4(theta) |g_pp|^2 W(2 k sin theta)``. They are NOT the Fresnel
    reflection coefficients: they reduce to a common value at normal incidence
    (``g_HH(0) = g_VV(0) = (eps - 1) / (1 + sqrt(eps))^2``), and ``|g_VV|`` exceeds ``|g_HH|`` at
    oblique incidence for a high-permittivity surface such as seawater.

    Reference: Valenzuela (1978); Ulaby, Moore and Fung, "Microwave Remote Sensing" Vol. II.

    Parameters
    ----------
    theta_inc_rad : scalar or array
        Incidence angle from the surface normal [rad].
    eps_r : complex
        Relative permittivity of seawater.

    Returns
    -------
    tuple
        ``(g_hh, g_vv)`` complex Bragg scattering coefficients for horizontal and vertical polarisation.
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

    Resonant scattering from the wind-driven ripple at the Bragg wavenumber ``k_B = 2 k sin(theta)``,
    with ``k = 2 pi / wavelength``. This is the return the geometric-optics glistening model misses at
    oblique incidence (no facets tilted enough to retroreflect) and that dominates ocean backscatter
    beyond about 20 deg incidence. The cross-section is
    ``sigma0_pp = 16 pi k^4 cos^4(theta) |g_pp|^2 W_iso(k_B)`` with the isotropic 2-D spectrum
    ``W_iso(K) = S(K) / (2 pi K)`` from the Elfouhaily ``S(k)``, and the circular amplitudes
    ``g_RL = (g_HH + g_VV) / 2`` (RHCP to LHCP, handedness flipped) and ``g_RR = (g_HH - g_VV) / 2``
    (RHCP to RHCP, preserved). The 16-pi prefactor belongs to a height spectrum normalised to the
    height variance (``int W_iso d^2K = sigma_h^2``); it is the small-roughness limit of the same
    Kirchhoff integral whose large-roughness limit is the geometric-optics NRCS. SPM validity holds for moderate incidence
    (about 10 to 70 deg); near nadir
    ``k_B -> 0``, the Bragg return vanishes and the geometric-optics term owns the geometry, so this
    returns 0 there.

    Reference: Valenzuela (1978); Ulaby, Moore and Fung, "Microwave Remote Sensing" Vol. II.

    Parameters
    ----------
    theta_inc_rad : scalar or array
        Incidence angle from the surface normal [rad].
    wavelength_m : float
        Radar wavelength [m].
    eps_r : complex
        Seawater relative permittivity.
    wind_speed_m_s : scalar
        Wind speed at 10 m [m/s].
    inverse_wave_age : scalar
        Inverse wave age Omega_c [-].

    Returns
    -------
    tuple
        ``(sigma0_RL, sigma0_RR)`` cross-pol and co-pol NRCS [dimensionless]; 0 at or below nadir.
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

    prefactor = 16.0 * np.pi * k_radar**4 * cos_t**4 * W_iso           # 16 pi k^4 cos^4(theta) W_iso
    sigma0_RL = prefactor * np.abs(g_RL) ** 2
    sigma0_RR = prefactor * np.abs(g_RR) ** 2
    return sigma0_RL, sigma0_RR


def bragg_bistatic_coefficients(theta_inc_rad, theta_sca_rad, phi_sca_rad, eps_r):
    """First-order small-perturbation (Bragg) amplitudes for bistatic scattering.

    The linear bases are forward-scatter aligned for both waves: ``h = z x k / |z x k|`` and ``v = h x k``,
    with ``k`` the propagation direction. The incident wave travels along azimuth 0; the scattered wave
    leaves at azimuth ``phi_sca`` (0 = forward, pi = back toward the source). At backscatter
    ``a_hh = -g_hh`` and ``a_vv = g_vv`` of :func:`bragg_scattering_coefficients` (the sign is the basis flip
    ``h_s = -h_i``); for ``eps -> inf`` the amplitudes reduce to the perfect-conductor boundary perturbation.

    Reference: Valenzuela (1967); Ulaby, Moore and Fung, "Microwave Remote Sensing" Vol. II.

    Parameters
    ----------
    theta_inc_rad, theta_sca_rad : scalar or array
        Incidence and scattering angles from the surface normal [rad].
    phi_sca_rad : scalar or array
        Scattering azimuth relative to the incident propagation azimuth [rad].
    eps_r : complex
        Relative permittivity of seawater.

    Returns
    -------
    tuple
        ``(a_hh, a_hv, a_vh, a_vv)`` complex amplitudes; ``a_pq`` maps incident ``q`` to scattered ``p``.
    """
    cos_i = np.cos(theta_inc_rad)
    cos_s = np.cos(theta_sca_rad)
    sin_i = np.sin(theta_inc_rad)
    sin_s = np.sin(theta_sca_rad)
    cos_p = np.cos(phi_sca_rad)
    sin_p = np.sin(phi_sca_rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        root_i = np.sqrt(eps_r - sin_i**2)                             # sqrt(eps_r - sin^2 theta_i)
        root_s = np.sqrt(eps_r - sin_s**2)                             # sqrt(eps_r - sin^2 theta_s)
        den_h_i = cos_i + root_i                                       # TE factor, incident
        den_h_s = cos_s + root_s                                       # TE factor, scattered
        den_v_i = eps_r * cos_i + root_i                               # TM factor, incident
        den_v_s = eps_r * cos_s + root_s                               # TM factor, scattered
        a_hh = (eps_r - 1.0) * cos_p / (den_h_i * den_h_s)
        a_hv = (eps_r - 1.0) * root_i * sin_p / (den_v_i * den_h_s)   # incident v -> scattered h
        a_vh = (eps_r - 1.0) * root_s * sin_p / (den_h_i * den_v_s)   # incident h -> scattered v
        a_vv = (eps_r - 1.0) * (eps_r * sin_i * sin_s - root_i * root_s * cos_p) / (den_v_i * den_v_s)
    return a_hh, a_hv, a_vh, a_vv


def _circular_bragg_amplitudes(a_hh, a_hv, a_vh, a_vv):
    """RHCP -> LHCP and RHCP -> RHCP amplitudes from the linear forward-aligned Bragg amplitudes.

    Under ``exp(+j omega t)`` a wave along ``k`` has ``RHCP = (v - j h) / sqrt(2)`` and
    ``LHCP = (v + j h) / sqrt(2)`` (``v x h = k``). The amplitudes are the projections of the scattered
    field onto the scattered wave's own circular vectors, so a rotation of the linear bases about either
    propagation direction changes only their phase.
    """
    amp_RL = 0.5 * ((a_vv - a_hh) - 1j * (a_vh + a_hv))               # LHCP component for RHCP incidence
    amp_RR = 0.5 * ((a_vv + a_hh) + 1j * (a_hv - a_vh))               # RHCP component for RHCP incidence
    return amp_RL, amp_RR


def bragg_bistatic_nrcs(theta_inc_rad, theta_sca_rad, phi_sca_rad, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age):
    """First-order bistatic Bragg (small-perturbation) ocean NRCS for circular polarisation.

    ``sigma0 = 16 pi k^4 cos^2(theta_i) cos^2(theta_s) |a|^2 W_iso(K)``, where ``K`` is the magnitude of the
    horizontal scattering wavevector (the resonant ripple) and ``a`` the circular amplitude built from
    :func:`bragg_bistatic_coefficients`. At ``theta_s = theta_i``, ``phi_sca = pi`` it equals
    :func:`bragg_backscatter_nrcs`. In the specular direction ``K = 0`` and the return is 0.

    Parameters
    ----------
    theta_inc_rad, theta_sca_rad : scalar or array
        Incidence and scattering angles from the surface normal [rad].
    phi_sca_rad : scalar or array
        Scattering azimuth relative to the incident propagation azimuth [rad] (0 = forward).
    wavelength_m : float
        Radar wavelength [m].
    eps_r : complex
        Seawater relative permittivity.
    wind_speed_m_s, inverse_wave_age : scalar
        Sea-state inputs (as in :func:`bragg_backscatter_nrcs`).

    Returns
    -------
    tuple
        ``(sigma0_RL, sigma0_RR)`` cross-pol and co-pol NRCS [dimensionless].
    """
    k_radar = 2.0 * np.pi / wavelength_m
    sin_i = np.sin(theta_inc_rad)
    sin_s = np.sin(theta_sca_rad)
    cos_p = np.cos(phi_sca_rad)
    k_bragg = k_radar * np.sqrt(np.maximum(sin_i**2 + sin_s**2 - 2.0 * sin_i * sin_s * cos_p, 0.0))   # K = k |k_s,t - k_i,t|

    positive_kb = k_bragg > 0.0
    k_safe = np.where(positive_kb, k_bragg, 1.0)
    S_bragg = elfouhaily_omnidirectional_spectrum(k_safe, wind_speed_m_s, inverse_wave_age)
    W_iso = np.where(positive_kb, S_bragg / (2.0 * np.pi * k_safe), 0.0)   # [m^4]

    amp_RL, amp_RR = _circular_bragg_amplitudes(*bragg_bistatic_coefficients(theta_inc_rad, theta_sca_rad, phi_sca_rad, eps_r))
    prefactor = 16.0 * np.pi * k_radar**4 * np.cos(theta_inc_rad) ** 2 * np.cos(theta_sca_rad) ** 2 * W_iso   # 16 pi k^4 cos^2 cos^2 W_iso
    return prefactor * np.abs(amp_RL) ** 2, prefactor * np.abs(amp_RR) ** 2


# Two-scale division: large-scale "tilting" waves carry k < k_radar / 3; the shorter waves above that
# are the Bragg ripples. The factor-of-3 split follows Brown (1978). The facet-slope average uses a
# standardized Gaussian quadrature over +/- 3 sigma.
_TWO_SCALE_DIVISION_FACTOR = 3.0
_TWO_SCALE_SLOPE_GRID = 15        # facet-slope quadrature points per axis (odd -> the zero-slope node is sampled)
_TWO_SCALE_SLOPE_SPAN = 3.0       # average the slope PDF over +/- this many standard deviations


def large_scale_slope_variances(wavelength_m, wind_speed_m_s, inverse_wave_age):
    """Mean-square slope of the large-scale (tilting) waves below k_radar / 3, in the wind frame.

    Integrates the Elfouhaily slope spectrum k^2 S(k) up to the two-scale division wavenumber for the
    total large-scale slope variance, then splits it by the Cox-Munk upwind/crosswind anisotropy ratio.

    Parameters
    ----------
    wavelength_m : float
        Radar wavelength [m].
    wind_speed_m_s : scalar
        Wind speed at 10 m [m/s].
    inverse_wave_age : scalar
        Inverse wave age Omega_c [-].

    Returns
    -------
    tuple
        ``(s_u_L_sq, s_c_L_sq)`` large-scale upwind / crosswind slope variances [dimensionless].
    """
    k_radar = 2.0 * np.pi / wavelength_m
    k_div = k_radar / _TWO_SCALE_DIVISION_FACTOR                       # two-scale division wavenumber
    k = np.logspace(np.log10(k_div * 1.0e-4), np.log10(k_div), 600)   # from well below the spectral peak to k_div
    integrand = k**2 * elfouhaily_omnidirectional_spectrum(k, wind_speed_m_s, inverse_wave_age)
    mss_large = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(k)))   # trapezoid: mss = int k^2 S(k) dk
    s_u_sq, s_c_sq = cox_munk_slope_variances(wind_speed_m_s)
    total = s_u_sq + s_c_sq
    return mss_large * s_u_sq / total, mss_large * s_c_sq / total      # split total by Cox-Munk anisotropy


def two_scale_bistatic_nrcs(theta_inc_rad, theta_sca_rad, phi_rx_rad, azimuth_rad, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age):
    """Two-scale (tilt-modulated) bistatic ocean Bragg NRCS for circular polarisation.

    Averages the first-order bistatic Bragg NRCS over the large-scale facet-slope distribution. Each
    tilting facet has its own local incidence and scattering angles and local scattering azimuth; the
    Bragg return is evaluated in the facet frame and summed per unit mean-surface area, i.e. weighted by
    the facet area ``sqrt(1 + |grad z|^2)`` per unit horizontal area. Facets whose local Bragg wavenumber
    (the tangential scattering wavevector) falls below the two-scale division ``k / 3`` add no Bragg
    return: they are quasi-specular mirrors covered by the GO term. Facets hidden from the Tx or the Rx
    add nothing. All facets stay in the slope-PDF normalisation. This adds the tilt-modulation lift and the
    upwind/crosswind asymmetry that first-order Bragg misses, and reduces to :func:`bragg_bistatic_nrcs` as
    the large-scale slope -> 0. The circular amplitudes are taken in each facet frame; a tilt rotates the
    linear bases, which changes only the phase of a circular amplitude, so the tilt-induced polarisation
    rotation is included.

    Parameters
    ----------
    theta_inc_rad : scalar or array
        Incidence angle of the Tx ray at the mean surface [rad].
    theta_sca_rad : scalar or array
        Scattering angle of the ray toward the Rx at the mean surface [rad].
    phi_rx_rad : scalar or array
        Horizontal azimuth of the Rx direction relative to the Tx direction [rad]; 0 = backscatter,
        pi = forward-specular plane.
    azimuth_rad : scalar or array
        Horizontal azimuth of the Tx direction relative to the upwind axis [rad].
    wavelength_m : float
        Radar wavelength [m].
    eps_r : complex
        Seawater relative permittivity.
    wind_speed_m_s, inverse_wave_age : scalar
        Sea-state inputs (as in :func:`bragg_backscatter_nrcs`).

    Returns
    -------
    tuple
        ``(sigma0_RL, sigma0_RR)`` tilt-averaged cross-pol and co-pol NRCS.
    """
    angles = (theta_inc_rad, theta_sca_rad, phi_rx_rad, azimuth_rad)
    scalar_in = all(np.ndim(a) == 0 for a in angles)
    theta, theta_s, phi_rx, phi = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, dtype=float)) for a in angles))

    s_u_L_sq, s_c_L_sq = large_scale_slope_variances(wavelength_m, wind_speed_m_s, inverse_wave_age)
    cos2 = np.cos(phi) ** 2
    sin2 = np.sin(phi) ** 2
    var_inplane = s_u_L_sq * cos2 + s_c_L_sq * sin2                    # slope variance along the look azimuth
    var_cross = s_u_L_sq * sin2 + s_c_L_sq * cos2                      # slope variance perpendicular to it
    covariance = (s_c_L_sq - s_u_L_sq) * np.sin(phi) * np.cos(phi)     # rotation of the wind-frame covariance

    # Standardized facet-slope nodes (same for every sample; scaled per sample by the local sigma).
    u = np.linspace(-_TWO_SCALE_SLOPE_SPAN, _TWO_SCALE_SLOPE_SPAN, _TWO_SCALE_SLOPE_GRID)
    Ux, Uy = np.meshgrid(u, u, indexing="ij")
    Ux = Ux.ravel()                                                   # (M,)
    Uy = Uy.ravel()
    weight = np.exp(-0.5 * (Ux**2 + Uy**2))                           # (M,) standard-normal slope weight

    # Rotate the full anisotropic Gaussian covariance into the look/cross-look frame.  Away from
    # exact upwind or crosswind the two components are correlated; drawing them independently from
    # their marginal variances gives the wrong facet distribution.
    chol_11 = np.sqrt(var_inplane)
    chol_21 = covariance / chol_11
    chol_22 = np.sqrt(np.maximum(var_cross - chol_21**2, 0.0))
    zx = Ux[None, :] * chol_11[:, None]                               # (N, M) in-plane facet slope
    zy = Ux[None, :] * chol_21[:, None] + Uy[None, :] * chol_22[:, None]  # correlated cross-plane slope

    # Mean-surface frame: x = horizontal direction toward the Tx, z = up.
    to_tx = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)[:, None, :]    # (N, 1, 3) surface -> Tx
    to_rx = np.stack(
        [np.sin(theta_s) * np.cos(phi_rx), np.sin(theta_s) * np.sin(phi_rx), np.cos(theta_s)], axis=-1
    )[:, None, :]                                                     # (N, 1, 3) surface -> Rx
    area_ratio = np.sqrt(1.0 + zx**2 + zy**2)                         # facet area per unit horizontal area
    n_facet = np.stack([-zx, -zy, np.ones_like(zx)], axis=-1) / area_ratio[..., None]   # (N, M, 3) facet normal

    cos_i = np.sum(n_facet * to_tx, axis=-1)                          # local incidence cosine
    cos_s = np.sum(n_facet * to_rx, axis=-1)                          # local scattering cosine
    facet_visible = (cos_i > 1.0e-6) & (cos_s > 1.0e-6)               # facet faces both Tx and Rx (not self-shadowed)
    tx_t = to_tx - cos_i[..., None] * n_facet                         # facet-tangential part of surface -> Tx
    rx_t = to_rx - cos_s[..., None] * n_facet                         # facet-tangential part of surface -> Rx
    sin_i = np.linalg.norm(tx_t, axis=-1)
    sin_s = np.linalg.norm(rx_t, axis=-1)
    q_t = np.linalg.norm(tx_t + rx_t, axis=-1)                        # tangential scattering vector / k = local K_B / k

    # Facet frame: x_l = incident propagation azimuth (-tx_t); at normal incidence any tangent serves,
    # since the circular amplitudes are then independent of the azimuth reference.
    any_tangent = np.cross(n_facet, np.array([0.0, 1.0, 0.0]))        # facet tangent (n_facet is never along y)
    ref_t = np.where((sin_i > 1.0e-12)[..., None], -tx_t, np.where((sin_s > 1.0e-12)[..., None], rx_t, any_tangent))
    x_l = ref_t / np.linalg.norm(ref_t, axis=-1, keepdims=True)
    y_l = np.cross(n_facet, x_l)
    phi_local = np.arctan2(np.sum(rx_t * y_l, axis=-1), np.sum(rx_t * x_l, axis=-1))   # scattering azimuth, 0 = forward

    sigma0_RL_grid, sigma0_RR_grid = bragg_bistatic_nrcs(
        np.arctan2(sin_i, cos_i), np.arctan2(sin_s, cos_s), phi_local, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age
    )

    bragg_regime = q_t >= 1.0 / _TWO_SCALE_DIVISION_FACTOR            # local K_B >= k_radar / 3
    w = weight[None, :] * facet_visible * bragg_regime * area_ratio   # (N, M) quasi-specular and hidden facets add nothing
    w_total = np.sum(weight)                                          # every facet stays in the slope-PDF normalisation
    sigma0_RL = np.sum(np.where(w > 0.0, sigma0_RL_grid, 0.0) * w, axis=1) / w_total   # slope-averaged NRCS
    sigma0_RR = np.sum(np.where(w > 0.0, sigma0_RR_grid, 0.0) * w, axis=1) / w_total

    if scalar_in:
        return float(sigma0_RL[0]), float(sigma0_RR[0])
    return sigma0_RL, sigma0_RR


def two_scale_backscatter_nrcs(theta_inc_rad, azimuth_rad, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age):
    """Monostatic two-scale ocean backscatter NRCS: :func:`two_scale_bistatic_nrcs` with the Rx along the Tx direction."""
    return two_scale_bistatic_nrcs(theta_inc_rad, theta_inc_rad, 0.0, azimuth_rad, wavelength_m, eps_r, wind_speed_m_s, inverse_wave_age)
