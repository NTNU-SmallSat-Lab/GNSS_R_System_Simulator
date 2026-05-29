"""Deterministic in-memory RxAntenna fixture for golden-file regression tests.

Bypasses RxAntenna.__init__ and populates the attributes that get_gain_dbi_for_direction
depends on: a tuple-keyed (azimuth, elevation, frequency) lookup of steered patterns plus the
steering grid. Each synthetic steered pattern peaks at its steering direction, so the bilinear
interpolation across the four bracketing steering patterns is genuinely exercised.
"""

import numpy as np

from object_definitions.Rx_antenna_def import RxAntenna

FREQUENCY_MHZ = 1575.42
N_THETA = 37   # 5 deg steps over 0..180
N_PHI = 72     # 5 deg steps over 0..360 (endpoint-exclusive)

STEERING_AZIMUTHS_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
STEERING_ELEVATIONS_DEG = [0.0, 5.0, 10.0, 15.0, 20.0]

BORESIGHT_DIRECTIVITY_DB = 10.0
PATTERN_FALLOFF_DB_PER_DEG = 0.2


def _steered_pattern_db(az_steer_deg: float, el_steer_deg: float) -> np.ndarray:
    """Synthetic steered power pattern (dB), peak 0 dB at the steering direction.

    Gain falls off linearly with angular separation from the steering axis, so different steering
    directions produce different samples at a fixed (theta, phi) look angle.
    """
    theta = np.linspace(0.0, 180.0, N_THETA) * np.pi / 180.0
    phi = np.linspace(0.0, 360.0, N_PHI, endpoint=False) * np.pi / 180.0
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")

    d_x = np.sin(theta_mesh) * np.cos(phi_mesh)
    d_y = np.sin(theta_mesh) * np.sin(phi_mesh)
    d_z = np.cos(theta_mesh)

    el_s = np.deg2rad(el_steer_deg)
    az_s = np.deg2rad(az_steer_deg)
    axis = np.array([np.sin(el_s) * np.cos(az_s), np.sin(el_s) * np.sin(az_s), np.cos(el_s)])

    cos_sep = np.clip(d_x * axis[0] + d_y * axis[1] + d_z * axis[2], -1.0, 1.0)
    separation_deg = np.degrees(np.arccos(cos_sep))
    return -PATTERN_FALLOFF_DB_PER_DEG * separation_deg


def make_test_antenna() -> RxAntenna:
    """Build a deterministic steerable RxAntenna with a known steered-pattern sweep."""
    antenna = RxAntenna.__new__(RxAntenna)
    antenna.R_antenna_to_body = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    antenna.max_steering_off_boresight_deg = 30.0
    antenna.frequencies = [FREQUENCY_MHZ]
    antenna.bands = ["L1"]
    antenna.polarizations = ["RHCP"]
    antenna.footprint_dB_contour = -3.0
    antenna.footprint_resolution = 64

    antenna.antenna_lookuptable = {}
    for az in STEERING_AZIMUTHS_DEG:
        for el in STEERING_ELEVATIONS_DEG:
            antenna.antenna_lookuptable[(az, el, FREQUENCY_MHZ)] = {
                "radiation_pattern_dB": _steered_pattern_db(az, el),
                "directivity_dB": BORESIGHT_DIRECTIVITY_DB,
                "directivity": 10.0 ** (BORESIGHT_DIRECTIVITY_DB / 10.0),
                "vector_bundle": np.zeros((64, 3)),
                "frequency": FREQUENCY_MHZ,
                "elevation": el,
                "azimuth": az,
            }
    antenna._build_steering_grid()
    return antenna


def gain_sampling_grid() -> tuple:
    """Deterministic input grid: look directions (theta, phi) x steering commands (az, el)."""
    theta_deg = np.array([5.0, 15.0, 30.0, 60.0])
    phi_deg = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.0])
    az_steer = np.array([0.0, 45.0, 90.0, 135.0, 200.0, 290.0, 350.0])
    el_steer = np.array([0.0, 5.0, 10.0, 15.0, 25.0])
    return theta_deg, phi_deg, az_steer, el_steer


def compute_gain_grid(antenna: RxAntenna) -> np.ndarray:
    """Compute gain across the full sampling grid; returns a 4D ndarray of shape
    (n_theta, n_phi, n_az_steer, n_el_steer)."""
    theta_deg, phi_deg, az_steer, el_steer = gain_sampling_grid()

    gains = np.full((len(theta_deg), len(phi_deg), len(az_steer), len(el_steer)), np.nan)
    for i, th in enumerate(theta_deg):
        th_rad = np.radians(th)
        for j, ph in enumerate(phi_deg):
            ph_rad = np.radians(ph)
            direction = np.array([
                np.sin(th_rad) * np.cos(ph_rad),
                np.sin(th_rad) * np.sin(ph_rad),
                np.cos(th_rad),
            ])
            for k, az in enumerate(az_steer):
                for m, el in enumerate(el_steer):
                    gains[i, j, k, m] = antenna.get_gain_dbi_for_direction(
                        direction, FREQUENCY_MHZ, azimuth_deg=az, elevation_deg=el
                    )
    return gains
