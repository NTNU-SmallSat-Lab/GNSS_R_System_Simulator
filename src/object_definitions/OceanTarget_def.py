"""Bistatic ocean RCS model.

Implements: sigma = sigma0(theta_i) * A_fresnel * fresnel_scale, where sigma0 is a calibrated
coefficient that decreases linearly in dB with incidence angle and is floored at
`sigma_floor_db`, and A_fresnel is a first-Fresnel-zone area approximation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from object_definitions.Target_def import Target, TargetRcs

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry, BistaticLinkBudgetSettings


class OceanTarget(Target):
    """Simplified ocean RCS: sigma0(theta_i) x A_fresnel x fresnel_scale."""

    def __init__(
        self,
        sigma0_db: float,
        slope_db_per_deg: float,
        sigma_floor_db: float,
        fresnel_scale: float,
    ) -> None:
        self.sigma0_db = float(sigma0_db)
        self.slope_db_per_deg = float(slope_db_per_deg)
        self.sigma_floor_db = float(sigma_floor_db)
        self.fresnel_scale = float(fresnel_scale)

    @classmethod
    def from_settings(cls, settings: "BistaticLinkBudgetSettings") -> "OceanTarget":
        """Build an OceanTarget from the ocean-related fields of BistaticLinkBudgetSettings."""
        return cls(
            sigma0_db=settings.ocean_sigma0_db,
            slope_db_per_deg=settings.ocean_sigma_slope_db_per_deg,
            sigma_floor_db=settings.ocean_sigma_floor_db,
            fresnel_scale=settings.fresnel_rcs_scale,
        )

    def compute_rcs(self, geometry: "BistaticGeometry", wavelength_m: float) -> TargetRcs:
        """Return ocean RCS [m^2] + diagnostics (sigma0_db, fresnel_area_m2) per sample."""
        incidence_deg = geometry.incidence_angle_deg
        range_tx_m = geometry.range_tx_m
        range_rx_m = geometry.range_rx_m

        incidence_pos = np.clip(incidence_deg, 0.0, None)
        sigma0_db = self.sigma0_db - self.slope_db_per_deg * incidence_pos
        sigma0_db = np.maximum(sigma0_db, self.sigma_floor_db)
        sigma0_lin = np.power(10.0, sigma0_db / 10.0)

        range_sum = range_tx_m + range_rx_m
        denom = np.where(range_sum > 0.0, range_sum, np.nan)
        fresnel_radius = np.sqrt(wavelength_m * range_tx_m * range_rx_m / denom)
        fresnel_area = np.pi * fresnel_radius ** 2
        rcs_m2 = sigma0_lin * fresnel_area * self.fresnel_scale

        valid = (
            np.isfinite(range_tx_m)
            & np.isfinite(range_rx_m)
            & np.isfinite(incidence_deg)
            & (range_tx_m > 0.0)
            & (range_rx_m > 0.0)
        )
        sigma0_db = np.where(valid, sigma0_db, np.nan)
        fresnel_area = np.where(valid, fresnel_area, np.nan)
        rcs_m2 = np.where(valid, rcs_m2, np.nan)

        return TargetRcs(
            rcs_m2=rcs_m2,
            info={"sigma0_db": sigma0_db, "fresnel_area_m2": fresnel_area},
        )
