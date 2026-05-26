"""Bistatic ocean RCS using geometric-optics scattering from a Cox-Munk slope PDF.

Per sample, the model evaluates:
    sigma = pi * |R_pol(theta_local)|^2 * (|q| / q_z)^4 * P_slope(s_u, s_c) * A_eff
where:
    - q = k_hat_out - k_hat_in (scattering vector in surface frame)
    - theta_local = bistatic_angle / 2 (local incidence on the facet that mirrors Tx -> Rx)
    - (s_u, s_c) is the required facet slope in the wind-aligned frame
    - P_slope is the Cox-Munk 2-D Gaussian over the slope PDF
    - |R_pol|^2 is the RHCP -> LHCP power reflectivity at the carrier frequency, derived from
      a Klein-Swift seawater permittivity at the configured temperature and salinity
    - A_eff is the first-Fresnel-zone area (pi * lambda * R_t * R_r / (R_t + R_r))

`wave_height_m` is reported as a Rayleigh roughness diagnostic only in this revision; the
slope-driven Cox-Munk PDF carries the wind-dependent sea-state behaviour directly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from functions.ocean_physics import (
    circular_pol_reflectivities,
    cox_munk_slope_pdf,
    cox_munk_slope_variances,
    seawater_permittivity,
)
from object_definitions.Constants_def import Constants
from object_definitions.Target_def import Target, TargetRcs

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry, BistaticLinkBudgetSettings


class OceanTarget(Target):
    """Ocean RCS model parameterised by sea state (wave height, wind), temperature and salinity."""

    def __init__(
        self,
        wave_height_m: float,
        wind_speed_m_s: float,
        wind_direction_deg: float,
        sea_temperature_c: float,
        salinity_psu: float,
    ) -> None:
        if wave_height_m < 0.0:
            raise ValueError(f"wave_height_m must be >= 0; got {wave_height_m}")
        if wind_speed_m_s <= 0.0:
            raise ValueError(f"wind_speed_m_s must be > 0; got {wind_speed_m_s}")
        if not (-90.0 <= sea_temperature_c <= 90.0):
            raise ValueError(f"sea_temperature_c outside plausible range: {sea_temperature_c}")
        if not (0.0 <= salinity_psu <= 50.0):
            raise ValueError(f"salinity_psu outside plausible range: {salinity_psu}")

        self.wave_height_m = float(wave_height_m)
        self.wind_speed_m_s = float(wind_speed_m_s)
        self.wind_direction_deg = float(wind_direction_deg)
        self.sea_temperature_c = float(sea_temperature_c)
        self.salinity_psu = float(salinity_psu)

    @classmethod
    def from_settings(cls, settings: "BistaticLinkBudgetSettings") -> "OceanTarget":
        return cls(
            wave_height_m=settings.ocean_wave_height_m,
            wind_speed_m_s=settings.ocean_wind_speed_m_s,
            wind_direction_deg=settings.ocean_wind_direction_deg,
            sea_temperature_c=settings.ocean_sea_temperature_c,
            salinity_psu=settings.ocean_salinity_psu,
        )

    def compute_rcs(self, geometry: "BistaticGeometry", wavelength_m: float) -> TargetRcs:
        n_surface = geometry.n_surface          # (n, 3) outward surface unit normal
        n_tx2s = geometry.n_tx2s                # (n, 3) Tx -> surface unit vector
        n_s2rx = geometry.n_s2rx                # (n, 3) surface -> Rx unit vector
        range_tx = geometry.range_tx_m          # (n,) Tx-to-surface range
        range_rx = geometry.range_rx_m          # (n,) surface-to-Rx range
        bistatic_deg = geometry.bistatic_angle_deg
        incidence_deg = geometry.incidence_angle_deg
        n_samples = n_surface.shape[0]

        # Local tangent basis
        z_global = np.array([0.0, 0.0, 1.0])
        x_global = np.array([1.0, 0.0, 0.0])
        z_dot_abs = np.abs(n_surface @ z_global)          # |n_surface . z| per sample
        use_z = z_dot_abs < 0.95                          # use +z as reference except near the +/-z poles
        ref_axis = np.where(use_z[:, None], z_global, x_global)

        proj = np.einsum("ij,ij->i", n_surface, ref_axis)         # component of ref_axis along n_surface
        north_unnorm = ref_axis - proj[:, None] * n_surface       # ref_axis projected onto the tangent plane
        north_mag = np.linalg.norm(north_unnorm, axis=1)
        valid_basis = np.isfinite(north_mag) & (north_mag > 1e-9)

        n_north = np.full((n_samples, 3), np.nan)
        n_north[valid_basis] = north_unnorm[valid_basis] / north_mag[valid_basis, None]
        n_east = np.cross(n_north, n_surface)             # right-handed local ENU (east = north x n_surface)
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
        fresnel_radius_sq = wavelength_m * range_tx * range_rx / denom    # r_F^2 = lambda * R_t * R_r / (R_t + R_r)
        fresnel_area = np.pi * fresnel_radius_sq                          # A_F: first Fresnel zone (coherent regime)

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

        # Incoherent contribution: NRCS integrated over the glistening zone.
        sigma_incoherent = nrcs * glistening_area                         # rough-surface diffuse return

        # Coherent contribution: smooth-surface specular reflection, suppressed by Rayleigh
        # roughness exp(-kappa) and gated to geometries where the required facet slope is
        # zero (the macroscopic surface IS the mirror). Off-specular geometries get sigma_coh = 0.
        rayleigh_kappa = (
            4.0 * np.pi * self.wave_height_m * np.cos(theta_local_rad) / wavelength_m
        ) ** 2                                                            # kappa = (4 pi sigma_h cos theta_local / lambda)^2
        slope_mag_sq = s_u ** 2 + s_c ** 2
        at_specular = np.isfinite(slope_mag_sq) & (slope_mag_sq < 1.0e-6) # required facet slope ~ 0
        sigma_coherent = np.where(
            at_specular,
            R_RL_sq * fresnel_area * np.exp(-rayleigh_kappa),
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
                "rayleigh_roughness_kappa": np.where(valid, rayleigh_kappa, np.nan),
                "nrcs": np.where(valid, nrcs, np.nan),
                "rcs_coherent_m2": np.where(valid, sigma_coherent, np.nan),
                "rcs_incoherent_m2": np.where(valid, sigma_incoherent, np.nan),
            },
        )
