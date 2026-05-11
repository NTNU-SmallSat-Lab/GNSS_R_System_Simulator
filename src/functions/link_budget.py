from dataclasses import dataclass

import numpy as np

from functions.rotations import quat_to_rotation_matrix
from object_definitions.Constants_def import Constants
from object_definitions.Rx_antenna_def import RxAntenna


@dataclass(slots=True)
class BistaticLinkBudgetSettings:
    """
    =========================================================================================================
    BistaticLinkBudgetSettings class:
        - Configuration values for a simplified bistatic GNSS-R link budget.
        - Single source of truth for the `[LINK BUDGET]` INI section: field names, defaults
          and units defined here are picked up automatically by the configuration loader.

        ATTRIBUTES:
            tx_eirp_dbw:                  [float]  GNSS transmitter EIRP [dBW]
            system_losses_db:             [float]  Lumped implementation losses [dB]
            polarization_losses_db:       [float]  Polarization mismatch losses [dB]
            ocean_sigma0_db:              [float]  Effective sigma0 at nadir [dB]
            ocean_sigma_slope_db_per_deg: [float]  Linear incidence-angle slope of sigma0 [dB/deg]
            ocean_sigma_floor_db:         [float]  Lower bound on effective sigma0 [dB]
            fresnel_rcs_scale:            [float]  Scalar multiplier on Fresnel-area RCS [-]

        NOTES:
            - Ocean RCS is modeled as `sigma = sigma0(theta_i) * A_fresnel * fresnel_rcs_scale`.
            - `ocean_sigma*` parameters are effective calibration terms in this simplified model,
              not a strict geophysical NRCS split into coherent and diffuse components.
            - This is intentionally a first-order engineering model for rapid trade studies.
    =========================================================================================================
    """

    tx_eirp_dbw: float = 27.0
    system_losses_db: float = 2.0
    polarization_losses_db: float = 0.5
    ocean_sigma0_db: float = -20.0
    ocean_sigma_slope_db_per_deg: float = 0.25
    ocean_sigma_floor_db: float = -45.0
    fresnel_rcs_scale: float = 1.0


def coerce_link_budget_settings(settings: dict | BistaticLinkBudgetSettings | None) -> BistaticLinkBudgetSettings:
    if settings is None:
        return BistaticLinkBudgetSettings()
    if isinstance(settings, BistaticLinkBudgetSettings):
        return settings
    if isinstance(settings, dict):
        return BistaticLinkBudgetSettings(**settings)
    raise TypeError("settings must be None, dict, or BistaticLinkBudgetSettings")


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
    settings: dict | BistaticLinkBudgetSettings | None = None,
    q_sat: np.ndarray | None = None,
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
        settings:                 [dict | BistaticLinkBudgetSettings | None]  Link budget settings
        q_sat:                    [np.ndarray, shape (4,) | None]  Satellite attitude quaternion [w, x, y, z]

    OUTPUT:
        link_budget: [dict]   Link budget components and received power arrays
    =========================================================================================================
    """
    settings_obj = coerce_link_budget_settings(settings)
    if q_sat is None:
        q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    tx_positions_eci = np.asarray(tx_positions_eci, dtype=float)
    rx_positions_eci = np.asarray(rx_positions_eci, dtype=float)
    rx_velocities_eci = np.asarray(rx_velocities_eci, dtype=float)
    reflection_positions_eci = np.asarray(reflection_positions_eci, dtype=float)
    incidence_deg = np.asarray(incidence_deg, dtype=float)

    num_positions = reflection_positions_eci.shape[0]
    range_rx_m = np.linalg.norm((reflection_positions_eci - rx_positions_eci.T) * 1e-3, axis=1) * 1e3
    range_tx_m = np.linalg.norm((reflection_positions_eci - tx_positions_eci.T) * 1e-3, axis=1) * 1e3

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

    return compute_bistatic_link_budget(
        range_tx_m=range_tx_m,
        range_rx_m=range_rx_m,
        incidence_deg=incidence_deg,
        rx_gain_dbi=rx_gain_dbi,
        frequency_hz=frequency_hz,
        settings=settings_obj,
    )


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

    if q_sat is None:
        q_sat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
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

    sat_speed = np.linalg.norm(sat_vel_eci)
    sat_radius = np.linalg.norm(sat_pos_eci)
    if sat_speed <= 0.0 or sat_radius <= 0.0:
        return invalid

    v_hat = sat_vel_eci / sat_speed
    h_vec = np.cross(sat_pos_eci, v_hat)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= 0.0:
        return invalid
    n_hat = h_vec / h_norm

    c_hat = np.cross(v_hat, n_hat)
    c_norm = np.linalg.norm(c_hat)
    if c_norm <= 0.0:
        return invalid
    c_hat = c_hat / c_norm

    R_eci_to_vnc = np.vstack([v_hat, n_hat, c_hat])
    R_attitude = quat_to_rotation_matrix(q_sat)
    R_vnc_to_body = R_attitude @ Constants.R_body_to_vnc_normal.T

    n_ar_a = antenna.R_antenna_to_body.T @ R_vnc_to_body @ R_eci_to_vnc @ n_ar_eci
    n_ar_a_norm = np.linalg.norm(n_ar_a)
    if n_ar_a_norm <= 0.0:
        return invalid
    return n_ar_a / n_ar_a_norm


def compute_simple_ocean_rcs(
    range_tx_m: np.ndarray,
    range_rx_m: np.ndarray,
    incidence_deg: np.ndarray,
    wavelength_m: float,
    settings: BistaticLinkBudgetSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    =========================================================================================================
    Compute simple ocean RCS model for a bistatic link budget.

    The model is `sigma = sigma0(theta_i) * A_fresnel` where A_fresnel is an isotropic first
    Fresnel-zone area approximation.

    INPUTS:
        range_tx_m:    [np.ndarray]                  Tx-to-surface range [m]
        range_rx_m:    [np.ndarray]                  Surface-to-Rx range [m]
        incidence_deg: [np.ndarray]                  Incidence angle at the surface [deg]
        wavelength_m:  [float]                       Carrier wavelength [m]
        settings:      [BistaticLinkBudgetSettings]  Calibration parameters

    OUTPUT:
        result: [tuple]   (sigma0_db, fresnel_area_m2, rcs_m2) — each np.ndarray same shape as inputs

    NOTES:
        - `sigma0` here is treated as an effective coefficient in the simplified model above.
        - The function does not explicitly separate coherent/specular and diffuse/incoherent
          scattering physics.
    =========================================================================================================
    """
    range_tx_m = np.asarray(range_tx_m, dtype=float)
    range_rx_m = np.asarray(range_rx_m, dtype=float)
    incidence_deg = np.asarray(incidence_deg, dtype=float)

    incidence_pos = np.clip(incidence_deg, 0.0, None)
    sigma0_db = settings.ocean_sigma0_db - settings.ocean_sigma_slope_db_per_deg * incidence_pos
    sigma0_db = np.maximum(sigma0_db, settings.ocean_sigma_floor_db)
    sigma0_lin = _linear_from_db(sigma0_db)

    range_sum = range_tx_m + range_rx_m
    denom = np.where(range_sum > 0.0, range_sum, np.nan)
    fresnel_radius = np.sqrt(wavelength_m * range_tx_m * range_rx_m / denom)
    fresnel_area = np.pi * fresnel_radius**2
    rcs_m2 = sigma0_lin * fresnel_area * settings.fresnel_rcs_scale

    valid = np.isfinite(range_tx_m) & np.isfinite(range_rx_m) & np.isfinite(incidence_deg) & (range_tx_m > 0.0) & (range_rx_m > 0.0)
    sigma0_db = np.where(valid, sigma0_db, np.nan)
    fresnel_area = np.where(valid, fresnel_area, np.nan)
    rcs_m2 = np.where(valid, rcs_m2, np.nan)

    return sigma0_db, fresnel_area, rcs_m2


def compute_bistatic_link_budget(
    range_tx_m: np.ndarray,
    range_rx_m: np.ndarray,
    incidence_deg: np.ndarray,
    rx_gain_dbi: np.ndarray,
    frequency_hz: float,
    settings: dict | BistaticLinkBudgetSettings | None = None,
) -> dict:
    """
    =========================================================================================================
    Compute simplified bistatic GNSS-R received power.

    `fspl_sum_db` is the sum of one-way FSPL terms (Tx->surface plus surface->Rx).
    `bistatic_spreading_loss_db` is the corresponding spreading term used in the simplified
    bistatic radar equation implementation.

    INPUTS:
        range_tx_m:    [np.ndarray]   Distance transmitter to reflection point [m]
        range_rx_m:    [np.ndarray]   Distance reflection point to receiver [m]
        incidence_deg: [np.ndarray]   Incidence angle at the ocean surface [deg]
        rx_gain_dbi:   [np.ndarray]   Receiver antenna gain along look direction [dBi]
        frequency_hz:  [float]        Carrier frequency [Hz]
        settings:      [dict | BistaticLinkBudgetSettings | None]   Link budget settings

    OUTPUT:
        link_budget: [dict]   Link budget components and received power arrays
    =========================================================================================================
    """
    settings_obj = coerce_link_budget_settings(settings)

    range_tx_m = np.asarray(range_tx_m, dtype=float)
    range_rx_m = np.asarray(range_rx_m, dtype=float)
    incidence_deg = np.asarray(incidence_deg, dtype=float)
    rx_gain_dbi = np.asarray(rx_gain_dbi, dtype=float)

    if range_tx_m.shape != range_rx_m.shape:
        raise ValueError("range_tx_m and range_rx_m must have identical shape")
    if incidence_deg.shape != range_tx_m.shape:
        raise ValueError("incidence_deg must match range arrays shape")
    if rx_gain_dbi.shape != range_tx_m.shape:
        raise ValueError("rx_gain_dbi must match range arrays shape")
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

    sigma0_db, fresnel_area_m2, rcs_m2 = compute_simple_ocean_rcs(range_tx_m, range_rx_m, incidence_deg, wavelength_m, settings_obj)

    total_losses_lin = _linear_from_db(settings_obj.system_losses_db + settings_obj.polarization_losses_db)
    tx_eirp_w = _linear_from_db(settings_obj.tx_eirp_dbw)
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
        "incidence_deg": incidence_deg,
        "rx_gain_dbi": rx_gain_dbi,
        "ocean_sigma0_db": sigma0_db,
        "ocean_fresnel_area_m2": fresnel_area_m2,
        "ocean_rcs_m2": rcs_m2,
        "received_power_w": pr_w,
        "received_power_dbw": pr_dbw,
        "received_power_dbm": pr_dbm,
    }
