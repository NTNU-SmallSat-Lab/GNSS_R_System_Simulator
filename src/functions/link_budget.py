from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from functions.rotations import eci_to_antenna_rotation_from_state
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna

if TYPE_CHECKING:
    from object_definitions.Target_def import Target


@dataclass(slots=True)
class BistaticLinkBudgetSettings:
    """
    Configuration values for the bistatic GNSS-R link budget.

    Field-by-field source of truth for the `[LINK BUDGET]` + `[OCEAN]` INI sections;
    field names, defaults and units defined here are picked up automatically by the loader.

    ATTRIBUTES:
        system_losses_db:          [float]  Lumped implementation losses [dB]
        polarization_losses_db:    [float]  Receiver-side polarization mismatch losses [dB]
        ocean_wave_height_m:       [float]  RMS wave height [m]  (Rayleigh roughness diagnostic)
        ocean_wind_speed_m_s:      [float]  Wind speed at 10 m [m/s]  (drives Cox-Munk slope variances)
        ocean_wind_direction_deg:  [float]  Wind direction (meteorological FROM bearing) [deg]
        ocean_sea_temperature_c:   [float]  Sea surface temperature [degC]
        ocean_salinity_psu:        [float]  Salinity (practical salinity units) [psu]

    NOTE: GNSS transmitter EIRP is per-(constellation, band) and lives in
    `Constants.GNSS_EIRP_DBW` — it is not a link-budget config knob.
    """

    system_losses_db: float = 2.0
    polarization_losses_db: float = 0.5
    ocean_wave_height_m: float = 1.5
    ocean_wind_speed_m_s: float = 7.0
    ocean_wind_direction_deg: float = 270.0
    ocean_sea_temperature_c: float = 15.0
    ocean_salinity_psu: float = 35.0


def coerce_link_budget_settings(settings: dict | BistaticLinkBudgetSettings | None) -> BistaticLinkBudgetSettings:
    if settings is None:
        return BistaticLinkBudgetSettings()
    if isinstance(settings, BistaticLinkBudgetSettings):
        return settings
    if isinstance(settings, dict):
        return BistaticLinkBudgetSettings(**settings)
    raise TypeError("settings must be None, dict, or BistaticLinkBudgetSettings")


@dataclass(frozen=True, slots=True)
class BistaticGeometry:
    """
    =========================================================================================================
    Per-sample bistatic geometry: surface normal, propagation directions, ranges and angles.

    Conventions:
        - `n_tx2s`  points from Tx to surface (incoming propagation direction, unit).
        - `n_s2rx`  points from surface to Rx (outgoing propagation direction, unit).
        - `n_surface` is the outward surface unit normal.
        - `incidence_angle_deg`  is between -n_tx2s and n_surface (0deg = normal incidence).
        - `scattering_angle_deg` is between  n_s2rx  and n_surface.
        - `bistatic_angle_deg`   is between -n_tx2s and  n_s2rx
          (0deg = backscatter, 2 * theta_i for the specular mirror geometry, 180deg = forward).

    Shapes: vectors are (n, 3); scalar quantities are (n,). Invalid samples carry NaN.
    =========================================================================================================
    """

    n_surface: np.ndarray
    n_tx2s: np.ndarray
    n_s2rx: np.ndarray
    range_tx_m: np.ndarray
    range_rx_m: np.ndarray
    incidence_angle_deg: np.ndarray
    scattering_angle_deg: np.ndarray
    bistatic_angle_deg: np.ndarray


def build_bistatic_geometry(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    surface_positions: np.ndarray,
    surface_normals: np.ndarray,
) -> BistaticGeometry:
    """
    =========================================================================================================
    Compute per-sample bistatic geometry from Tx, Rx, surface positions and surface normals.

    INPUTS:
        tx_positions:      [np.ndarray, shape (3, n)]   Transmitter positions [m]
        rx_positions:      [np.ndarray, shape (3, n)]   Receiver positions [m]
        surface_positions: [np.ndarray, shape (n, 3)]   Surface (reflection) positions [m]
        surface_normals:   [np.ndarray, shape (n, 3)]   Outward surface normals (any magnitude)

    All four arrays must share the same reference frame at the same time samples.
    Invalid entries (NaN, zero range, zero-length normal) propagate as NaN in the output.

    OUTPUT:
        geometry: [BistaticGeometry]   Per-sample directions, ranges and angles.
    =========================================================================================================
    """
    tx = np.asarray(tx_positions, dtype=float)
    rx = np.asarray(rx_positions, dtype=float)
    surf = np.asarray(surface_positions, dtype=float)
    normal = np.asarray(surface_normals, dtype=float)

    # Robustness checks
    if tx.ndim != 2 or tx.shape[0] != 3:
        raise ValueError(f"tx_positions must have shape (3, n); got {tx.shape}")
    if rx.ndim != 2 or rx.shape[0] != 3:
        raise ValueError(f"rx_positions must have shape (3, n); got {rx.shape}")
    if surf.ndim != 2 or surf.shape[1] != 3:
        raise ValueError(f"surface_positions must have shape (n, 3); got {surf.shape}")
    if normal.shape != surf.shape:
        raise ValueError(f"surface_normals must match surface_positions shape; got {normal.shape} vs {surf.shape}")
    n = surf.shape[0]
    if tx.shape[1] != n or rx.shape[1] != n:
        raise ValueError(f"tx_positions and rx_positions must have n={n} columns; got {tx.shape[1]} and {rx.shape[1]}")

    tx_T = tx.T
    rx_T = rx.T

    p_s2tx = tx_T - surf # Vectors from surface to Tx; n_tx2s = -p_s2tx / |p_s2tx|
    p_s2rx = rx_T - surf # Vectors from surface to Rx; n_s2rx =  p_s2rx / |p_s2rx|

    range_tx_m = np.linalg.norm(p_s2tx, axis=1)
    range_rx_m = np.linalg.norm(p_s2rx, axis=1)
    normal_mag = np.linalg.norm(normal, axis=1)

    # Validity mask
    valid = (
        np.isfinite(range_tx_m)
        & np.isfinite(range_rx_m)
        & np.isfinite(normal_mag)
        & (range_tx_m > 0.0)
        & (range_rx_m > 0.0)
        & (normal_mag > 0.0)
    )

    # Initialize outputs with NaN; valid samples get overwritten below, invalid samples stay NaN
    n_tx2s = np.full((n, 3), np.nan)
    n_s2rx = np.full((n, 3), np.nan)
    n_surface = np.full((n, 3), np.nan)

    # Unit vectors point in the physically correct directions for valid geometry; invalid geometry stays NaN.
    n_tx2s[valid] = -p_s2tx[valid] / range_tx_m[valid, None] # Unit vector from Tx to surface
    n_s2rx[valid] = p_s2rx[valid] / range_rx_m[valid, None]  # Unit vector from surface to Rx
    n_surface[valid] = normal[valid] / normal_mag[valid, None]  # Unit surface normal

    # Angle calculations: NaN for invalid geometry, valid angles in [0, 180] degrees for valid geometry.
    cos_inc = np.full(n, np.nan)
    cos_sca = np.full(n, np.nan)
    cos_bist = np.full(n, np.nan)

    # Valid geometry: compute cosines of angles using dot products, then clip to [-1, 1] to avoid numerical issues with arccos.
    cos_inc[valid] = -np.sum(n_tx2s[valid] * n_surface[valid], axis=1) # Cosine of incidence angle  = -n_tx2s . n_surface
    cos_sca[valid] = np.sum(n_s2rx[valid] * n_surface[valid], axis=1)  # Cosine of scattering angle =  n_s2rx . n_surface
    cos_bist[valid] = -np.sum(n_tx2s[valid] * n_s2rx[valid], axis=1)   # Cosine of bistatic angle   = -n_tx2s . n_s2rx

    # Convert to degrees; invalid geometry stays NaN.
    # Clipping to [-1, 1] above ensures that valid geometry yields real angles in [0, 180] degrees, even with small numerical errors.
    incidence_deg = np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
    scattering_deg = np.degrees(np.arccos(np.clip(cos_sca, -1.0, 1.0)))
    bistatic_deg = np.degrees(np.arccos(np.clip(cos_bist, -1.0, 1.0)))

    return BistaticGeometry(
        n_surface=n_surface,
        n_tx2s=n_tx2s,
        n_s2rx=n_s2rx,
        range_tx_m=np.where(valid, range_tx_m, np.nan),
        range_rx_m=np.where(valid, range_rx_m, np.nan),
        incidence_angle_deg=incidence_deg,
        scattering_angle_deg=scattering_deg,
        bistatic_angle_deg=bistatic_deg,
    )


def resolve_single_link_budget_frequency_hz(antenna) -> float:
    """
    =========================================================================================================
    Resolve the carrier frequency to use for the link budget.

    Multi-band support (one link budget per Rx band, across multiple GNSS constellations) is planned
    but not yet implemented. Until then, the antenna must be configured with exactly one band; this
    helper enforces that and falls back to L1 when no antenna is provided.

    INPUTS:
        antenna: [RxAntenna | None]   Receiver antenna, or None for the L1 default

    OUTPUT:
        frequency_hz: [float]   Carrier frequency [Hz]
    =========================================================================================================
    """
    if antenna is None or not getattr(antenna, "frequencies", None):
        return Constants.BANDS_CENTER_FREQUENCIES_Hz["L1"]
    if len(antenna.frequencies) > 1:
        raise NotImplementedError(
            f"Multi-band link budget is not yet supported "
            f"(antenna configured for bands {antenna.bands}). "
            "Configure exactly one band for now."
        )
    return float(antenna.frequencies[0]) * 1e6


def compute_reflection_link_budget(
    tx_positions_eci: np.ndarray,
    rx_positions_eci: np.ndarray,
    rx_velocities_eci: np.ndarray,
    reflection_positions_eci: np.ndarray,
    incidence_deg: np.ndarray,
    antenna: RxAntenna | None,
    frequency_hz: float,
    tx_eirp_dbw: float,
    settings: dict | BistaticLinkBudgetSettings | None = None,
    q_sat: np.ndarray | None = None,
    target: "Target | None" = None,
) -> dict:
    """
    =========================================================================================================
    Compute the common GNSS-R reflection link budget for specular and BRP paths.

    Position and velocity arrays are expected in the simulator's existing convention: satellite state
    vectors as shape ``(3, n)`` and reflection points as shape ``(n, 3)``.

    INPUTS:
        tx_positions_eci:         [np.ndarray, shape (3, n)]  Transmitter ECI positions [m]
        rx_positions_eci:         [np.ndarray, shape (3, n)]  Receiver ECI positions [m]
        rx_velocities_eci:        [np.ndarray, shape (3, n)]  Receiver ECI velocities [m/s]
        reflection_positions_eci: [np.ndarray, shape (n, 3)]  Reflection point ECI positions [m]
        incidence_deg:            [np.ndarray, shape (n,)]    Incidence angle at reflection point [deg]
        antenna:                  [RxAntenna | None]          Receiver antenna (None => isotropic 0 dBi)
        frequency_hz:             [float]                     Carrier frequency [Hz]
        tx_eirp_dbw:              [float]                     Transmitter EIRP for this (constellation, band) [dBW]
        settings:                 [dict | BistaticLinkBudgetSettings | None]  Rx-system link-budget settings
        q_sat:                    [np.ndarray, shape (4,) | None]  Satellite attitude quaternion [w, x, y, z]
        target:                   [Target | None]             RCS model; defaults to OceanTarget.from_settings(settings)

    OUTPUT:
        link_budget: [dict]   Link budget components and received power arrays
    =========================================================================================================
    """
    settings_obj = coerce_link_budget_settings(settings)
    if q_sat is None:
        q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if target is None:
        # Local import to avoid a load-time cycle (Target_def imports BistaticGeometry from here).
        from object_definitions.OceanTarget_def import OceanTarget

        target = OceanTarget.from_settings(settings_obj)

    tx_positions_eci = np.asarray(tx_positions_eci, dtype=float)
    rx_positions_eci = np.asarray(rx_positions_eci, dtype=float)
    rx_velocities_eci = np.asarray(rx_velocities_eci, dtype=float)
    reflection_positions_eci = np.asarray(reflection_positions_eci, dtype=float)
    incidence_deg = np.asarray(incidence_deg, dtype=float)

    num_positions = reflection_positions_eci.shape[0]

    # Spherical surface normal (the position vector itself) is used by default;
    # `scattering_angle_deg` and `bistatic_angle_deg` therefore carry up to ~0.19 deg residual
    # error when the upstream geometry was solved on the ellipsoid.
    geometry = build_bistatic_geometry(
        tx_positions_eci,
        rx_positions_eci,
        reflection_positions_eci,
        reflection_positions_eci,
    )
    # Override the incidence angle with the caller-supplied value to preserve elliptical-Earth accuracy.
    geometry = dataclasses.replace(geometry, incidence_angle_deg=incidence_deg)

    range_tx_m = geometry.range_tx_m
    range_rx_m = geometry.range_rx_m

    # No antenna pattern -> assume isotropic (0 dBi). Otherwise NaN gains
    # (e.g. reflection outside FOV) propagate to NaN power.
    if antenna is None:
        rx_gain_dbi = np.zeros(num_positions, dtype=float)
    else:
        rx_gain_dbi = np.full(num_positions, np.nan, dtype=float)
        freq_mhz = float(frequency_hz) / 1e6
        for i in range(num_positions):
            if not np.isfinite(range_rx_m[i]) or not np.isfinite(range_tx_m[i]):
                continue
            if np.isnan(reflection_positions_eci[i]).any():
                continue
            look_dir_a = compute_rx_look_direction_antenna_frame(
                rx_positions_eci[:, i],
                rx_velocities_eci[:, i],
                reflection_positions_eci[i, :],
                antenna,
                q_sat=q_sat,
            )
            steering_angles = antenna.resolve_steering_angles_for_direction(look_dir_a)
            if steering_angles is None:
                continue
            steering_azimuth_deg, steering_elevation_deg = steering_angles
            rx_gain_dbi[i] = antenna.get_gain_dbi_for_direction(
                look_dir_a,
                freq_mhz,
                azimuth_deg=steering_azimuth_deg,
                elevation_deg=steering_elevation_deg,
            )

    wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz
    target_rcs = target.compute_rcs(geometry, wavelength_m)

    link_budget = compute_bistatic_link_budget(
        range_tx_m=range_tx_m,
        range_rx_m=range_rx_m,
        rx_gain_dbi=rx_gain_dbi,
        rcs_m2=target_rcs.rcs_m2,
        frequency_hz=frequency_hz,
        tx_eirp_dbw=tx_eirp_dbw,
        settings=settings_obj,
    )
    link_budget["incidence_deg"] = geometry.incidence_angle_deg
    link_budget["scattering_deg"] = geometry.scattering_angle_deg
    link_budget["bistatic_deg"] = geometry.bistatic_angle_deg
    link_budget["target_info"] = target_rcs.info
    return link_budget


def _linear_from_db(values_db):
    return np.power(10.0, np.asarray(values_db, dtype=float) / 10.0)


def _db_from_linear(values_linear):
    values_linear = np.asarray(values_linear, dtype=float)
    values_db = np.full(values_linear.shape, np.nan, dtype=float)
    valid = np.isfinite(values_linear) & (values_linear > 0.0)
    values_db[valid] = 10.0 * np.log10(values_linear[valid])
    return values_db


def compute_rx_look_direction_antenna_frame(
    sat_pos_eci: np.ndarray,
    sat_vel_eci: np.ndarray,
    refl_pos_eci: np.ndarray,
    antenna: RxAntenna,
    q_sat: np.ndarray | None = None,
) -> np.ndarray:
    """
    =========================================================================================================
    Compute reflection look direction in the antenna frame.

    INPUTS:
        sat_pos_eci:  [np.ndarray, shape (3,)]   Satellite ECI position [m]
        sat_vel_eci:  [np.ndarray, shape (3,)]   Satellite ECI velocity [m/s]
        refl_pos_eci: [np.ndarray, shape (3,)]   Reflection point ECI position [m]
        antenna:      [RxAntenna]                Receiver antenna
        q_sat:        [np.ndarray, shape (4,) | None]   Attitude quaternion [w, x, y, z]
                                                        (default: identity [1, 0, 0, 0])

    OUTPUT:
        n_ar_a: [np.ndarray, shape (3,)]   Unit vector in antenna frame, or NaNs if geometry is invalid
    =========================================================================================================
    """
    invalid = np.array([np.nan, np.nan, np.nan], dtype=float)
    if antenna is None:
        return invalid

    sat_pos_eci = np.asarray(sat_pos_eci, dtype=float)
    sat_vel_eci = np.asarray(sat_vel_eci, dtype=float)
    refl_pos_eci = np.asarray(refl_pos_eci, dtype=float)

    if q_sat is not None:
        q_sat = np.asarray(q_sat, dtype=float)

    if sat_pos_eci.shape != (3,) or sat_vel_eci.shape != (3,) or refl_pos_eci.shape != (3,):
        return invalid
    if not (np.isfinite(sat_pos_eci).all() and np.isfinite(sat_vel_eci).all() and np.isfinite(refl_pos_eci).all()):
        return invalid

    sat_to_refl = refl_pos_eci - sat_pos_eci
    sat_to_refl_norm = np.linalg.norm(sat_to_refl)
    if sat_to_refl_norm <= 0.0:
        return invalid
    n_ar_eci = sat_to_refl / sat_to_refl_norm

    R_eci_to_antenna = eci_to_antenna_rotation_from_state(sat_pos_eci, sat_vel_eci, antenna, q_sat)
    if R_eci_to_antenna is None:
        return invalid
    n_ar_a = R_eci_to_antenna @ n_ar_eci
    n_ar_a_norm = np.linalg.norm(n_ar_a)
    if n_ar_a_norm <= 0.0:
        return invalid
    return n_ar_a / n_ar_a_norm


def compute_bistatic_link_budget(
    range_tx_m: np.ndarray,
    range_rx_m: np.ndarray,
    rx_gain_dbi: np.ndarray,
    rcs_m2: np.ndarray,
    frequency_hz: float,
    tx_eirp_dbw: float,
    settings: dict | BistaticLinkBudgetSettings | None = None,
) -> dict:
    """
    =========================================================================================================
    Compute simplified bistatic GNSS-R received power from a precomputed RCS.

    Pr = EIRP * G_rx * lambda^2 * sigma / ((4 pi)^3 * R_t^2 * R_r^2 * L)

    `fspl_sum_db` is the sum of one-way FSPL terms (Tx->surface plus surface->Rx).
    `bistatic_spreading_loss_db` is the corresponding spreading term used in the simplified
    bistatic radar equation implementation.

    INPUTS:
        range_tx_m:    [np.ndarray]   Distance transmitter to reflection point [m]
        range_rx_m:    [np.ndarray]   Distance reflection point to receiver [m]
        rx_gain_dbi:   [np.ndarray]   Receiver antenna gain along look direction [dBi]
        rcs_m2:        [np.ndarray]   Target RCS per sample [m^2]; supplied by a Target implementation
        frequency_hz:  [float]        Carrier frequency [Hz]
        tx_eirp_dbw:   [float]        Transmitter EIRP for the specific (constellation, band) [dBW]
        settings:      [dict | BistaticLinkBudgetSettings | None]   Rx-system link-budget settings

    OUTPUT:
        link_budget: [dict]   Link budget components and received power arrays
    =========================================================================================================
    """
    settings_obj = coerce_link_budget_settings(settings)

    range_tx_m = np.asarray(range_tx_m, dtype=float)
    range_rx_m = np.asarray(range_rx_m, dtype=float)
    rx_gain_dbi = np.asarray(rx_gain_dbi, dtype=float)
    rcs_m2 = np.asarray(rcs_m2, dtype=float)

    if range_tx_m.shape != range_rx_m.shape:
        raise ValueError("range_tx_m and range_rx_m must have identical shape")
    if rx_gain_dbi.shape != range_tx_m.shape:
        raise ValueError("rx_gain_dbi must match range arrays shape")
    if rcs_m2.shape != range_tx_m.shape:
        raise ValueError("rcs_m2 must match range arrays shape")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz

    fspl_tx_db = np.full(range_tx_m.shape, np.nan, dtype=float)
    fspl_rx_db = np.full(range_rx_m.shape, np.nan, dtype=float)
    valid_tx = np.isfinite(range_tx_m) & (range_tx_m > 0.0)
    valid_rx = np.isfinite(range_rx_m) & (range_rx_m > 0.0)
    fspl_tx_db[valid_tx] = 20.0 * np.log10((4.0 * np.pi * range_tx_m[valid_tx]) / wavelength_m)
    fspl_rx_db[valid_rx] = 20.0 * np.log10((4.0 * np.pi * range_rx_m[valid_rx]) / wavelength_m)

    fspl_sum_db = fspl_tx_db + fspl_rx_db
    bistatic_spreading_loss_db = np.full(range_tx_m.shape, np.nan, dtype=float)
    valid_bistatic = valid_tx & valid_rx
    if np.any(valid_bistatic):
        bistatic_spreading_loss_db[valid_bistatic] = fspl_sum_db[valid_bistatic] - 10.0 * np.log10((4.0 * np.pi) / (wavelength_m**2))

    total_losses_lin = _linear_from_db(settings_obj.system_losses_db + settings_obj.polarization_losses_db)
    tx_eirp_w = _linear_from_db(tx_eirp_dbw)
    rx_gain_lin = _linear_from_db(rx_gain_dbi)

    denom = ((4.0 * np.pi) ** 3) * (range_tx_m**2) * (range_rx_m**2) * total_losses_lin
    pr_w = tx_eirp_w * rx_gain_lin * (wavelength_m**2) * rcs_m2 / denom

    valid = np.isfinite(pr_w) & (pr_w > 0.0)
    pr_w = np.where(valid, pr_w, np.nan)
    pr_dbw = _db_from_linear(pr_w)
    pr_dbm = pr_dbw + 30.0

    return {
        "frequency_hz": float(frequency_hz),
        "wavelength_m": float(wavelength_m),
        "tx_to_surface_range_m": range_tx_m,
        "surface_to_rx_range_m": range_rx_m,
        "fspl_tx_db": fspl_tx_db,
        "fspl_rx_db": fspl_rx_db,
        "fspl_sum_db": fspl_sum_db,
        "bistatic_spreading_loss_db": bistatic_spreading_loss_db,
        "rx_gain_dbi": rx_gain_dbi,
        "rcs_m2": rcs_m2,
        "received_power_w": pr_w,
        "received_power_dbw": pr_dbw,
        "received_power_dbm": pr_dbm,
    }
