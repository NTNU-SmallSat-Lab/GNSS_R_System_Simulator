"""
Shared bistatic geometry helpers used by both CoverageCellCentric
and MovingTargetCellCentric.
"""

import numpy as np

EARTH_RADIUS_M         = 6_371_000.0
TX_BEAM_HALF_ANGLE_DEG = 60.0
SPEED_OF_LIGHT_M_S     = 299_792_458.0
BOLTZMANN_K            = 1.380649e-23


def get_rx_beam_half_angle(antenna, db_threshold=-3.0):
    """
    Scan the antenna radiation pattern and return the nadir angle (degrees)
    at which gain first drops to db_threshold below the peak.
    Returns the maximum such angle across all phi slices.
    """
    if antenna is None:
        return 17.2

    lookup_key = list(antenna.antenna_lookuptable.keys())[0]
    pattern_db = antenna.antenna_lookuptable[lookup_key]["radiation_pattern_dB"]
    n_theta    = pattern_db.shape[0]
    theta_rad  = np.linspace(0, np.pi, n_theta)
    angles     = []

    for phi_i in range(pattern_db.shape[1]):
        slc  = pattern_db[:, phi_i]
        peak = int(np.argmax(slc))
        for d in [1, -1]:
            for i in range(peak + d, (n_theta if d == 1 else -1), d):
                if slc[i] <= db_threshold:
                    prev = i - d
                    t = ((db_threshold - slc[prev]) / (slc[i] - slc[prev])
                         if slc[prev] != slc[i] else 0.0)
                    angles.append(np.degrees(theta_rad[prev]
                                             + t * (theta_rad[i] - theta_rad[prev])))
                    break
            else:
                continue
            break

    return float(np.max(angles)) if angles else 17.2


def steer_target_eci(rx_pos_eci, tx_pos_eci, steer_fraction, R=EARTH_RADIUS_M):
    """
    Return the steered Rx boresight target on Earth's surface (ECI metres).
    steer_fraction = 0 → Tx sub-satellite point (nadir direction from Tx)
    steer_fraction = 1 → geometric horizon from Rx in the Tx azimuth
    """
    rx_up         = rx_pos_eci / np.linalg.norm(rx_pos_eci)
    tx_nadir_unit = tx_pos_eci / np.linalg.norm(tx_pos_eci)

    # Component of Tx nadir projected onto the Rx local horizontal plane
    tx_horiz = tx_nadir_unit - np.dot(tx_nadir_unit, rx_up) * rx_up
    norm = np.linalg.norm(tx_horiz)
    if norm < 1e-10:
        return tx_nadir_unit * R
    tx_horiz /= norm

    rx_r = np.linalg.norm(rx_pos_eci)
    eta  = np.arccos(np.clip(R / rx_r, -1.0, 1.0))
    horizon_unit = np.cos(eta) * rx_up + np.sin(eta) * tx_horiz

    target_unit = (1.0 - steer_fraction) * tx_nadir_unit + steer_fraction * horizon_unit
    target_unit /= np.linalg.norm(target_unit)
    return target_unit * R


# ── Link budget helpers ────────────────────────────────────────────────────────

def tx_eirp_dbw(tx_ecef, cell_ecef, lb: dict) -> float:
    """
    Tx EIRP toward cell_ecef in dBW.

    lb keys used:
      tx_type              : "gnss" | "leo_pnt" | "fixed_gain"
      tx_eirp_dbw          : constant EIRP for tx_type="gnss"
      tx_pa_output_dbm     : PA output power [dBm] for "leo_pnt" and "fixed_gain"
      tx_gain_dbi          : fixed peak antenna gain [dBi] for tx_type="fixed_gain"
      tx_isoflux_nadir_dbi : gain at 0° nadir for isoflux model
      tx_isoflux_edge_dbi  : gain at max nadir angle for isoflux model
      tx_isoflux_max_nadir_deg : nadir angle at which edge gain is reached
    """
    tx_type = lb.get("tx_type", "gnss")

    if tx_type == "gnss":
        return float(lb.get("tx_eirp_dbw", 27.0))

    if tx_type == "fixed_gain":
        # Constant EIRP: PA output + fixed peak gain (beam assumed pointed at target)
        return (float(lb.get("tx_pa_output_dbm", 0.0)) - 30.0) + float(lb.get("tx_gain_dbi", 0.0))

    # LEO PNT isoflux: gain varies linearly in dB from nadir to edge
    tx_up         = tx_ecef / np.linalg.norm(tx_ecef)
    tx_to_cell    = cell_ecef - tx_ecef
    tx_to_cell_u  = tx_to_cell / np.linalg.norm(tx_to_cell)
    nadir_deg     = float(np.degrees(np.arccos(np.clip(np.dot(-tx_up, tx_to_cell_u), -1.0, 1.0))))
    theta_max     = float(lb.get("tx_isoflux_max_nadir_deg", 60.0))
    g_nadir       = float(lb.get("tx_isoflux_nadir_dbi", 1.0))
    g_edge        = float(lb.get("tx_isoflux_edge_dbi", 3.0))
    g_tx          = g_nadir + (g_edge - g_nadir) * min(nadir_deg / theta_max, 1.0)
    return (float(lb.get("tx_pa_output_dbm", 35.0)) - 30.0) + g_tx  # dBW


def cell_snr_db(eirp_tx_dbw: float, rx_peak_gain_dbi: float,
                range_tx_m: float, range_rx_m: float,
                cell_area_m2: float, sigma0_db: float,
                freq_hz: float, lb: dict) -> float:
    """
    Bistatic radar SNR in dB for a distributed ocean surface cell.

    Uses the Rx antenna peak gain as an optimistic approximation
    (exact for cell centres near the beam axis, up to 3 dB high at the -3 dB edge).

    lb keys used:
      system_losses_db        : implementation + polarisation losses
      polarization_losses_db  : (added to system_losses_db)
      rx_system_noise_temp_k  : system noise temperature [K]
      rx_bandwidth_mhz        : receiver noise bandwidth [MHz]
    """
    wavelength_m = SPEED_OF_LIGHT_M_S / freq_hz
    sigma_db     = sigma0_db + 10.0 * np.log10(max(cell_area_m2, 1.0))
    losses_db    = (float(lb.get("system_losses_db", 2.0))
                    + float(lb.get("polarization_losses_db", 0.5)))
    pr_dbw       = (eirp_tx_dbw
                    + rx_peak_gain_dbi
                    + sigma_db
                    + 20.0 * np.log10(wavelength_m)
                    - 30.0 * np.log10(4.0 * np.pi)
                    - 20.0 * np.log10(range_tx_m)
                    - 20.0 * np.log10(range_rx_m)
                    - losses_db)
    t_sys        = float(lb.get("rx_system_noise_temp_k", 290.0))
    bw_hz        = float(lb.get("rx_bandwidth_mhz", 10.0)) * 1e6
    pn_dbw       = (10.0 * np.log10(BOLTZMANN_K)
                    + 10.0 * np.log10(t_sys)
                    + 10.0 * np.log10(bw_hz))
    return pr_dbw - pn_dbw
