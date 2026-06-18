"""Delay-Doppler resolution-cell geometry for GNSS-R.

    A GNSS-R receiver correlates the reflected signal against a local PRN-code replica (the delay
    axis) and Fourier-transforms over the coherent integration time T_coh (the Doppler axis). One
    Delay-Doppler Map (DDM) pixel is the surface region that maps into one code chip of delay AND
    one Doppler bin. That region -- not the antenna footprint -- is the area of distributed ocean
    clutter that competes with a point target inside the same cell, and the area a target occults
    in forward-scatter mode.

    The cell is sized from the Jacobian of the (delay, Doppler) -> surface map. At a surface point
    with outward normal n_surface and the two scalar surface fields

        L(r)    = R_t(r) + R_r(r)                                 total path length [m]
        f_D(r)  = (1 / lambda) ( n_tx2s . v_tx - n_s2rx . v_rx )  Doppler [Hz]

    the tangent-plane gradients are

        grad L   = n_tx2s - n_s2rx                                (then projected onto the tangent plane)
        grad f_D = (1 / lambda) ( v_tx_perp / R_t + v_rx_perp / R_r )   (v_perp = component perpendicular to the line of sight)

    One chip maps to a path-length step  delta_L = c / R_chip  and one Doppler bin is  delta_fD = 1 / T_coh.
    Because grad L and grad f_D both lie in the tangent plane, the magnitude of the (L, f_D) -> (x, y)
    Jacobian determinant is basis-free,

        |det J| = | grad L x grad f_D |          [Hz/m]

    and the surface area mapping into one chip-by-bin box is

        A_cell = delta_L * delta_fD / |det J|    [m^2].

    Degenerate geometry: at the specular point grad L -> 0 (it is the stationary point of L), so
    |det J| -> 0 and A_cell diverges -- the iso-delay rings are physically far apart at the glint.
    The same happens where grad f_D -> 0 (momentary zero range-rate gradient). Both are bounded by
    capping A_cell at the antenna footprint area, the largest surface the receiver can integrate.
    The exact near-specular cell (first iso-delay ellipse from the curvature of L) is a future
    refinement; the footprint cap is the conservative bound used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry


def delay_doppler_cell_area(
    geometry: BistaticGeometry,
    tx_velocities_eci: np.ndarray,
    rx_velocities_eci: np.ndarray,
    wavelength_m: float,
    chip_length_m: float,
    coherent_integration_time_s: float,
    footprint_area_m2: np.ndarray | float | None = None,
) -> dict:
    """
    =========================================================================================================
    Delay-Doppler resolution-cell area per sample from the (delay, Doppler) -> surface Jacobian.

    INPUTS:
        geometry:                    [BistaticGeometry]            Per-sample directions/ranges (n samples)
        tx_velocities_eci:           [np.ndarray, shape (3, n)]    Transmitter ECI velocities [m/s]
        rx_velocities_eci:           [np.ndarray, shape (3, n)]    Receiver ECI velocities [m/s]
        wavelength_m:                [float]                       Carrier wavelength [m]
        chip_length_m:               [float]                       One code-chip path length delta_L = c / R_chip [m]
        coherent_integration_time_s: [float]                       Coherent integration time T_coh [s]; Doppler bin = 1 / T_coh
        footprint_area_m2:           [np.ndarray | float | None]   Upper-bound cap (antenna footprint area) [m^2]; None -> no cap

    OUTPUT:
        result: [dict]   Per-sample (n,) arrays:
            cell_area_m2           DDM cell area, footprint-capped [m^2]
            uncapped_cell_area_m2  cell area before the footprint cap [m^2]
            delay_width_m          one-chip width across the surface, delta_L / |grad L| [m]
            doppler_width_m        one-bin width across the surface, delta_fD / |grad f_D| [m]
            delay_gradient         |grad L| (tangent), dimensionless [m/m]
            doppler_gradient_hz_m  |grad f_D| (tangent) [Hz/m]
            abs_jacobian_det_hz_m  |det J| = |grad L x grad f_D| [Hz/m]
            footprint_capped       True where the cap (or a near-specular degeneracy) set the area to the footprint
    =========================================================================================================
    """
    if wavelength_m <= 0.0 or not np.isfinite(wavelength_m):
        raise ValueError(f"wavelength_m must be finite and > 0; got {wavelength_m}")
    if chip_length_m <= 0.0 or not np.isfinite(chip_length_m):
        raise ValueError(f"chip_length_m must be finite and > 0; got {chip_length_m}")
    if coherent_integration_time_s <= 0.0 or not np.isfinite(coherent_integration_time_s):
        raise ValueError(f"coherent_integration_time_s must be finite and > 0; got {coherent_integration_time_s}")

    n_surface = np.asarray(geometry.n_surface, dtype=float)   # (n, 3) outward unit normal
    n_tx2s = np.asarray(geometry.n_tx2s, dtype=float)         # (n, 3) Tx -> surface unit vector
    n_s2rx = np.asarray(geometry.n_s2rx, dtype=float)         # (n, 3) surface -> Rx unit vector
    range_tx = np.asarray(geometry.range_tx_m, dtype=float)   # (n,)
    range_rx = np.asarray(geometry.range_rx_m, dtype=float)   # (n,)
    n = n_surface.shape[0]

    v_tx = np.asarray(tx_velocities_eci, dtype=float)
    v_rx = np.asarray(rx_velocities_eci, dtype=float)
    if v_tx.shape != (3, n) or v_rx.shape != (3, n):
        raise ValueError(f"tx/rx velocities must have shape (3, {n}); got {v_tx.shape} and {v_rx.shape}")
    v_tx = v_tx.T                                             # (n, 3)
    v_rx = v_rx.T                                             # (n, 3)

    delta_L = float(chip_length_m)                            # one chip in path length [m]
    delta_fD = 1.0 / float(coherent_integration_time_s)       # one Doppler bin [Hz]

    def _project_tangent(vectors: np.ndarray) -> np.ndarray:
        normal_component = np.einsum("ij,ij->i", vectors, n_surface)   # vectors . n_surface
        return vectors - normal_component[:, None] * n_surface         # remove the normal part

    # Delay-field surface gradient: grad L = n_tx2s - n_s2rx, projected onto the tangent plane.
    grad_L = _project_tangent(n_tx2s - n_s2rx)

    # Doppler-field surface gradient: line-of-sight-perpendicular velocity over range, per platform.
    with np.errstate(invalid="ignore", divide="ignore"):
        v_tx_perp = v_tx - np.einsum("ij,ij->i", v_tx, n_tx2s)[:, None] * n_tx2s   # v_tx perpendicular to Tx LOS
        v_rx_perp = v_rx - np.einsum("ij,ij->i", v_rx, n_s2rx)[:, None] * n_s2rx   # v_rx perpendicular to Rx LOS
        grad_fD = _project_tangent((v_tx_perp / range_tx[:, None] + v_rx_perp / range_rx[:, None]) / wavelength_m)

    # |det J| = | grad L x grad f_D | because both gradients lie in the tangent plane.
    abs_det_J = np.linalg.norm(np.cross(grad_L, grad_fD), axis=1)     # [Hz/m]
    delay_gradient = np.linalg.norm(grad_L, axis=1)                   # |grad L| [m/m]
    doppler_gradient = np.linalg.norm(grad_fD, axis=1)               # |grad f_D| [Hz/m]

    geometry_valid = (
        np.isfinite(range_tx) & (range_tx > 0.0)
        & np.isfinite(range_rx) & (range_rx > 0.0)
        & np.isfinite(grad_L).all(axis=1)
        & np.isfinite(grad_fD).all(axis=1)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        uncapped = (delta_L * delta_fD) / abs_det_J                  # diverges to +inf where |det J| -> 0 (near specular)
        delay_width = delta_L / delay_gradient
        doppler_width = delta_fD / doppler_gradient

    uncapped = np.where(geometry_valid, uncapped, np.nan)

    if footprint_area_m2 is None:
        cell_area = np.where(geometry_valid & np.isfinite(uncapped) & (uncapped > 0.0), uncapped, np.nan)
        footprint_capped = geometry_valid & ~(np.isfinite(uncapped) & (uncapped > 0.0))
    else:
        footprint = np.broadcast_to(np.asarray(footprint_area_m2, dtype=float), (n,)).copy()
        footprint = np.where(np.isfinite(footprint) & (footprint > 0.0), footprint, np.nan)
        # np.fmin ignores NaN, so a near-specular divergence (uncapped = inf/NaN) collapses to the footprint.
        capped_area = np.fmin(uncapped, footprint)
        cell_area = np.where(geometry_valid, capped_area, np.nan)
        footprint_capped = geometry_valid & np.isfinite(footprint) & ~(uncapped < footprint)

    return {
        "cell_area_m2": cell_area,
        "uncapped_cell_area_m2": uncapped,
        "delay_width_m": np.where(geometry_valid, delay_width, np.nan),
        "doppler_width_m": np.where(geometry_valid, doppler_width, np.nan),
        "delay_gradient": np.where(geometry_valid, delay_gradient, np.nan),
        "doppler_gradient_hz_m": np.where(geometry_valid, doppler_gradient, np.nan),
        "abs_jacobian_det_hz_m": np.where(geometry_valid, abs_det_J, np.nan),
        "delay_chip_length_m": delta_L,
        "doppler_bin_hz": delta_fD,
        "footprint_capped": footprint_capped,
    }
