"""Deterministic in-memory RxAntenna fixture for golden-file regression tests.

Bypasses RxAntenna.__init__ and populates the attributes that get_gain_dbi_for_direction
depends on. Gain is computed analytically from the array geometry; only one unsteered
pattern per frequency is needed to supply the boresight directivity used in the on-demand
gain scaling.
"""

import numpy as np

from object_definitions.Rx_antenna_def import RxAntenna


FREQUENCY_MHZ = 1575.42
N_THETA = 36   # 5° steps
N_PHI = 72     # 5° steps

# Array geometry chosen so the analytic AF has visible directional structure within the
# test sampling grid: 4x4 half-wavelength array at L1 yields a main beam ~26° wide and
# clear sidelobes/nulls.
WAVELENGTH_M = 299792458.0 / (FREQUENCY_MHZ * 1e6)
ELEMENTS_X = 4
ELEMENTS_Y = 4
D_X = WAVELENGTH_M / 2.0
D_Y = WAVELENGTH_M / 2.0
BORESIGHT_DIRECTIVITY_DB = 10.0


def make_test_antenna() -> RxAntenna:
    """Build a deterministic RxAntenna with a known array geometry and boresight directivity."""
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
    antenna._array_geometry = {
        "elements_along_x": ELEMENTS_X,
        "elements_along_y": ELEMENTS_Y,
        "d_x": D_X,
        "d_y": D_Y,
    }

    # Single unsteered entry per frequency — supplies the boresight directivity that the on-demand
    # gain path scales the analytic AF by. The radiation_pattern_dB grid is retained for the
    # footprint-contour pipeline (not exercised by the golden gain test).
    pattern_db = (-0.5 * np.arange(N_THETA).reshape(-1, 1)).repeat(N_PHI, axis=1).astype(float)
    antenna.antenna_lookuptable = {
        FREQUENCY_MHZ: {
            "radiation_pattern_dB": pattern_db,
            "directivity_dB": BORESIGHT_DIRECTIVITY_DB,
            "directivity": 10.0 ** (BORESIGHT_DIRECTIVITY_DB / 10.0),
            "vector_bundle": np.zeros((64, 3)),
            "frequency": FREQUENCY_MHZ,
        }
    }
    return antenna


def gain_sampling_grid() -> tuple:
    """Deterministic input grid for the gain measurements."""
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
