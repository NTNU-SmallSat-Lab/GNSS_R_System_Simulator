"""Bistatic ocean RCS as a coherent (specular) plus incoherent (diffuse) contribution.

    Per sample the model returns sigma = sigma_coherent + sigma_incoherent, each an effective
    cross-section for the simulator's bistatic radar equation.

    Incoherent (rough-surface, geometric-optics over a Cox-Munk slope PDF):
    sigma_incoherent = pi * |R_pol(theta_local)|^2 * (|q| / q_z)^4 * P_slope(s_u, s_c) * A_glistening
    where:
    - q = k_hat_out - k_hat_in (scattering vector in surface frame)
    - theta_local = bistatic_angle / 2 (local incidence on the facet that mirrors Tx -> Rx)
    - (s_u, s_c) is the required facet slope in the wind-aligned frame
    - P_slope is the Cox-Munk 2-D Gaussian slope PDF
    - |R_pol|^2 is the RHCP -> LHCP power reflectivity at the carrier frequency, from a
      Klein-Swift seawater permittivity at the configured temperature and salinity
    - A_glistening is the effective glistening-zone area at the specular peak. In scalar fallback
      mode this is capped by the receiver illuminated area when supplied so a narrow antenna beam
      does not collect the full analytic glistening zone. In LB-2b patch-integration mode, the
      link-budget layer supplies each finite surface patch as its own illuminated area and sums
      the resulting per-patch power with per-patch antenna gain and ranges.

    Bragg (rough-surface resonant backscatter), added only at near-backscatter geometry where the GO
    term collapses: a first-order small-perturbation NRCS over the Elfouhaily spectrum, gated so the
    forward-specular path stays unchanged. See functions.ocean_physics.bragg_backscatter_nrcs.

    Coherent (smooth-surface specular reflection from the mean surface):
    sigma_coherent = 4 * pi * |R_pol|^2 * (R_t R_r / (R_t + R_r))^2 * exp(-kappa)
    - the flat-mirror cross-section that reproduces the two-ray coherent power
      Pr = Pt Gt Gr (lambda / (4 pi (R_t + R_r)))^2 |R|^2 through the bistatic radar equation
    - kappa = (4 pi sigma_h cos(theta_local) / lambda)^2 is the Rayleigh roughness attenuation
    - gated to specular geometry (required facet slope ~ 0); zero off-specular

    For typical sea state (sigma_h >> lambda) kappa is large and the coherent term collapses, leaving
    the incoherent glistening-zone return; for very calm water the coherent term dominates.

    NOTE: Delay-Doppler filtering is still not modelled here. LB-2b integrates over the receiver
    footprint; adding a delay-Doppler mask would further restrict which patches contribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from functions.ocean_physics import (
    circular_pol_reflectivities,
    cox_munk_slope_pdf,
    cox_munk_slope_variances,
    seawater_permittivity,
    two_scale_backscatter_nrcs,
)
from object_definitions.Constants_def import Constants
from object_definitions.Target_def import Target, TargetRcs

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry, BistaticLinkBudgetSettings


# Bragg backscatter is gated to near-backscatter geometry: the BRP retroreflection line is collinear
# (bistatic ~ 0 deg), while any forward-specular reflection sits at bistatic = 2 * incidence (>= ~10 deg).
# Gating below this keeps the specular path -- and its golden reference -- exactly unchanged.
_BRAGG_MAX_BISTATIC_DEG = 5.0
# First-order Bragg (SPM) is valid at oblique incidence; below this the quasi-specular geometric-optics
# term owns the geometry and the SPM formula overshoots, so the Bragg contribution is suppressed.
_BRAGG_MIN_INCIDENCE_DEG = 20.0


class OceanTarget(Target):
    """Ocean RCS model parameterised by sea state (wave height, wind), temperature and salinity."""

    def __init__(
        self,
        wave_height_m: float,
        wind_speed_m_s: float,
        wind_direction_deg: float,
        sea_temperature_c: float,
        salinity_psu: float,
        inverse_wave_age: float = 0.84,
    ) -> None:
        if wave_height_m < 0.0:
            raise ValueError(f"wave_height_m must be >= 0; got {wave_height_m}")
        if wind_speed_m_s <= 0.0:
            raise ValueError(f"wind_speed_m_s must be > 0; got {wind_speed_m_s}")
        if not (-90.0 <= sea_temperature_c <= 90.0):
            raise ValueError(f"sea_temperature_c outside plausible range: {sea_temperature_c}")
        if not (0.0 <= salinity_psu <= 50.0):
            raise ValueError(f"salinity_psu outside plausible range: {salinity_psu}")
        if not (0.84 <= inverse_wave_age <= 5.0):
            raise ValueError(f"inverse_wave_age (Omega_c) outside Elfouhaily range [0.84, 5.0]: {inverse_wave_age}")

        self.wave_height_m = float(wave_height_m)
        self.wind_speed_m_s = float(wind_speed_m_s)
        self.wind_direction_deg = float(wind_direction_deg)
        self.sea_temperature_c = float(sea_temperature_c)
        self.salinity_psu = float(salinity_psu)
        self.inverse_wave_age = float(inverse_wave_age)

    @classmethod
    def from_settings(cls, settings: BistaticLinkBudgetSettings) -> OceanTarget:
        return cls(
            wave_height_m=settings.ocean_wave_height_m,
            wind_speed_m_s=settings.ocean_wind_speed_m_s,
            wind_direction_deg=settings.ocean_wind_direction_deg,
            sea_temperature_c=settings.ocean_sea_temperature_c,
            salinity_psu=settings.ocean_salinity_psu,
            inverse_wave_age=settings.ocean_inverse_wave_age,
        )

    def compute_rcs(self, geometry: BistaticGeometry, wavelength_m: float, illuminated_area_m2: np.ndarray | None = None) -> TargetRcs:
        n_surface = geometry.n_surface          # (n, 3) outward surface unit normal
        n_tx2s = geometry.n_tx2s                # (n, 3) Tx -> surface unit vector
        n_s2rx = geometry.n_s2rx                # (n, 3) surface -> Rx unit vector
        range_tx = geometry.range_tx_m          # (n,) Tx-to-surface range
        range_rx = geometry.range_rx_m          # (n,) surface-to-Rx range
        bistatic_deg = geometry.bistatic_angle_deg
        incidence_deg = geometry.incidence_angle_deg
        n_samples = n_surface.shape[0]

        # Local geographic ENU basis at the surface point: north is the Earth rotation axis (+z)
        # projected onto the tangent plane, east = north x up. This anchors the Cox-Munk wind
        # direction to true geographic north for all latitudes. Only at the geographic poles
        # (|n . z| -> 1) is that projection singular and the wind azimuth physically undefined;
        # there we fall back to a +x projection to keep a finite, consistent basis.
        z_global = np.array([0.0, 0.0, 1.0])
        x_global = np.array([1.0, 0.0, 0.0])
        proj_z = n_surface @ z_global                             # +z component of n_surface = sin(latitude)
        north_from_z = z_global - proj_z[:, None] * n_surface     # +z projected onto tangent plane (-> geographic north)
        proj_x = n_surface @ x_global                             # +x component of n_surface
        north_from_x = x_global - proj_x[:, None] * n_surface     # +x projection, used only at the poles
        near_pole = np.abs(proj_z) > 1.0 - 1e-12                  # n_surface ~ +/-z: geographic north degenerate
        north_unnorm = np.where(near_pole[:, None], north_from_x, north_from_z)
        north_mag = np.linalg.norm(north_unnorm, axis=1)
        valid_basis = np.isfinite(north_mag) & (north_mag > 1e-9)

        n_north = np.full((n_samples, 3), np.nan)
        n_north[valid_basis] = north_unnorm[valid_basis] / north_mag[valid_basis, None]
        n_east = np.cross(n_north, n_surface)             # right-handed local ENU (east = north x up)
        east_mag = np.linalg.norm(n_east, axis=1)
        valid_east = valid_basis & (east_mag > 1e-9)
        n_east[valid_east] /= east_mag[valid_east, None]

        # Scattering vector
        q = n_s2rx - n_tx2s                               # q = n_s2rx - n_tx2s (dimensionless in this form)
        q_mag = np.linalg.norm(q, axis=1)
        q_z = np.einsum("ij,ij->i", q, n_surface)         # q component along surface normal
        q_n = np.einsum("ij,ij->i", q, n_north)           # q component along local north
        q_e = np.einsum("ij,ij->i", q, n_east)            # q component along local east

        # Required facet slope (tilt that mirrors Tx -> Rx)
        valid_q = valid_east & np.isfinite(q_z) & (np.abs(q_z) > 1e-9)
        s_east = np.full(n_samples, np.nan)
        s_north = np.full(n_samples, np.nan)
        s_east[valid_q] = -q_e[valid_q] / q_z[valid_q]    # slope along east  = -q_e / q_z
        s_north[valid_q] = -q_n[valid_q] / q_z[valid_q]   # slope along north = -q_n / q_z

        # Wind-aligned slope (meteorological FROM bearing -> upwind axis)
        theta_w = np.radians(self.wind_direction_deg)
        cos_w = np.cos(theta_w)
        sin_w = np.sin(theta_w)
        s_u = s_east * sin_w + s_north * cos_w            # slope along upwind axis
        s_c = s_east * cos_w - s_north * sin_w            # slope along crosswind axis

        # Cox-Munk slope PDF
        sigma_u_sq, sigma_c_sq = cox_munk_slope_variances(self.wind_speed_m_s)
        pdf = cox_munk_slope_pdf(s_u, s_c, sigma_u_sq, sigma_c_sq)

        # Fresnel reflectivity at the local facet incidence angle = bistatic / 2
        theta_local_rad = np.radians(bistatic_deg) / 2.0
        frequency_hz = Constants.SPEED_OF_LIGHT / wavelength_m
        eps_r = seawater_permittivity(frequency_hz, self.sea_temperature_c, self.salinity_psu)
        R_RL_sq, R_RR_sq = circular_pol_reflectivities(theta_local_rad, eps_r)

        # Effective areas
        range_sum = range_tx + range_rx
        denom = np.where(range_sum > 0.0, range_sum, np.nan)
        reduced_range = range_tx * range_rx / denom                      # R_t R_r / (R_t + R_r)
        fresnel_radius_sq = wavelength_m * reduced_range                 # r_F^2 = lambda * R_t * R_r / (R_t + R_r)
        fresnel_area = np.pi * fresnel_radius_sq                         # A_F: first Fresnel zone footprint (diagnostic)

        # Glistening-zone area for the incoherent regime: integration of the slope PDF
        # over the surface, expressed as an effective area at the specular peak. The
        # sec^3(theta_inc) factor is the projection of the surface patch onto the
        # ranges geometry; capped near grazing to avoid an unphysical sec divergence.
        cos_inc = np.cos(np.radians(incidence_deg))
        cos_inc_safe = np.clip(cos_inc, 0.05, 1.0)                        # cap below ~87 deg
        slope_var_total = sigma_u_sq + sigma_c_sq                         # total slope variance (Cox-Munk)
        glistening_area = (
            np.pi * (range_tx * range_rx / denom) ** 2 * slope_var_total / cos_inc_safe ** 3
        )

        # NRCS from the Z-V geometric-optics formula (per-unit-area, dimensionless)
        geom_factor = np.full(n_samples, np.nan)
        geom_factor[valid_q] = (q_mag[valid_q] / np.abs(q_z[valid_q])) ** 4    # (|q| / q_z)^4
        nrcs = np.pi * R_RL_sq * geom_factor * pdf

        # Two-scale (tilt-modulated) Bragg backscatter NRCS, gated to near-backscatter geometry at
        # oblique incidence. The GO formula above collapses there (no facets tilted enough to
        # retroreflect); Bragg fills it in. Forward-specular and near-nadir geometries get 0, so the
        # specular path is untouched and the quasi-specular GO term owns nadir. The look azimuth
        # relative to the upwind axis drives the upwind/crosswind asymmetry of the tilt average.
        look_e = np.einsum("ij,ij->i", n_s2rx, n_east)                    # look horizontal projection, east
        look_n = np.einsum("ij,ij->i", n_s2rx, n_north)                   # look horizontal projection, north
        azimuth_rel_wind = np.arctan2(look_e, look_n) - theta_w           # look azimuth relative to upwind axis
        bragg_active = (bistatic_deg < _BRAGG_MAX_BISTATIC_DEG) & (incidence_deg >= _BRAGG_MIN_INCIDENCE_DEG)
        nrcs_bragg = np.zeros(n_samples)                                  # RHCP -> LHCP cross-pol (consumed)
        nrcs_bragg_co = np.zeros(n_samples)                               # RHCP -> RHCP co-pol (polarization-hook diagnostic)
        active = np.where(bragg_active & np.isfinite(azimuth_rel_wind))[0]
        if active.size > 0:
            nrcs_bragg[active], nrcs_bragg_co[active] = two_scale_backscatter_nrcs(
                np.radians(incidence_deg[active]), azimuth_rel_wind[active],
                wavelength_m, eps_r, self.wind_speed_m_s, self.inverse_wave_age,
            )

        # Receiver-limited illuminated area. In scalar mode this cap prevents the analytic glistening zone
        # from contributing more diffuse clutter than the antenna footprint can observe. In LB-2b patch
        # mode the caller supplies one small patch area per sample, so this expression becomes the
        # per-patch diffuse RCS contribution used by the power integration.
        if illuminated_area_m2 is None:
            illuminated_area = None
            effective_glistening_area = glistening_area
            bragg_area = np.pi * reduced_range ** 2 / cos_inc_safe ** 3
        else:
            illuminated_area = np.asarray(illuminated_area_m2, dtype=float)
            if illuminated_area.shape != glistening_area.shape:
                try:
                    illuminated_area = np.broadcast_to(illuminated_area, glistening_area.shape)
                except ValueError as exc:
                    raise ValueError(
                        f"illuminated_area_m2 must broadcast to shape {glistening_area.shape}; got {illuminated_area.shape}"
                    ) from exc
            illuminated_area = np.where(np.isfinite(illuminated_area) & (illuminated_area > 0.0), illuminated_area, np.nan)
            effective_glistening_area = np.minimum(glistening_area, illuminated_area)
            bragg_area = illuminated_area

        # Incoherent contribution: forward GO over the glistening zone plus Bragg over the backscatter
        # footprint area.
        sigma_incoherent = nrcs * effective_glistening_area + nrcs_bragg * bragg_area    # rough-surface diffuse return

        # Coherent contribution: smooth-surface specular reflection from the mean surface, suppressed
        # by Rayleigh roughness exp(-kappa) and gated to geometries where the required facet slope is
        # zero (the macroscopic surface IS the mirror). Off-specular geometries get sigma_coh = 0.
        # sigma_coh is the flat-mirror cross-section that reproduces the two-ray coherent power
        # Pr = Pt Gt Gr (lambda / (4 pi (R_t + R_r)))^2 |R|^2 through the bistatic radar equation.
        rayleigh_kappa = (
            4.0 * np.pi * self.wave_height_m * np.cos(theta_local_rad) / wavelength_m
        ) ** 2                                                            # kappa = (4 pi sigma_h cos theta_local / lambda)^2
        coherent_mirror_rcs = 4.0 * np.pi * reduced_range ** 2           # 4 pi (R_t R_r / (R_t + R_r))^2
        slope_mag_sq = s_u ** 2 + s_c ** 2
        at_specular = np.isfinite(slope_mag_sq) & (slope_mag_sq < 1.0e-6) # required facet slope ~ 0
        sigma_coherent = np.where(
            at_specular,
            R_RL_sq * coherent_mirror_rcs * np.exp(-rayleigh_kappa),
            0.0,
        )

        # Combined RCS
        rcs_m2 = sigma_coherent + sigma_incoherent

        valid = (
            valid_q
            & np.isfinite(range_tx)
            & np.isfinite(range_rx)
            & (range_tx > 0.0)
            & (range_rx > 0.0)
            & np.isfinite(bistatic_deg)
        )
        rcs_m2 = np.where(valid, rcs_m2, np.nan)

        return TargetRcs(
            rcs_m2=rcs_m2,
            info={
                "fresnel_reflectivity_RL_sq": np.where(valid, R_RL_sq, np.nan),
                "fresnel_reflectivity_RR_sq": np.where(valid, R_RR_sq, np.nan),
                "slope_pdf": np.where(valid, pdf, np.nan),
                "slope_upwind": np.where(valid, s_u, np.nan),
                "slope_crosswind": np.where(valid, s_c, np.nan),
                "fresnel_area_m2": np.where(valid, fresnel_area, np.nan),
                "glistening_area_m2": np.where(valid, glistening_area, np.nan),
                "effective_glistening_area_m2": np.where(valid, effective_glistening_area, np.nan),
                "illuminated_area_m2": (
                    np.full(n_samples, np.nan)
                    if illuminated_area is None
                    else np.where(valid, illuminated_area, np.nan)
                ),
                "rayleigh_roughness_kappa": np.where(valid, rayleigh_kappa, np.nan),
                "nrcs": np.where(valid, nrcs, np.nan),
                "nrcs_bragg_RL": np.where(valid, nrcs_bragg, np.nan),
                "nrcs_bragg_RR": np.where(valid, nrcs_bragg_co, np.nan),
                "bragg_area_m2": np.where(valid, bragg_area, np.nan),
                "rcs_bragg_m2": np.where(valid, nrcs_bragg * bragg_area, np.nan),
                "rcs_coherent_m2": np.where(valid, sigma_coherent, np.nan),
                "rcs_incoherent_m2": np.where(valid, sigma_incoherent, np.nan),
            },
        )
