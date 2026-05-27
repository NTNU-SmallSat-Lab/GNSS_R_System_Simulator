"""Bistatic RCS helpers for ship-as-target modelling.

flat_plate_rcs:  Physical-optics RCS of a rectangular PEC plate with back-face culling.
dihedral_rcs:    Bistatic RCS of a 90-deg dihedral corner reflector with wedge visibility.

Both functions are vectorised over the per-sample axis (n samples per call) and operate on
direction unit vectors / range scalars; they have no dependence on the simulator's frame
choices beyond a consistent reference frame for all input vectors.
"""
from __future__ import annotations

import numpy as np


def flat_plate_rcs(
    n_plate: np.ndarray,
    edge_1: np.ndarray,
    edge_2: np.ndarray,
    length_edge_1_m: float,
    length_edge_2_m: float,
    n_tx2s: np.ndarray,
    n_s2rx: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """
    =========================================================================================================
    Bistatic RCS of a rectangular PEC flat plate via physical optics.

        sigma = (4 pi A^2 / lambda^2) * cos^2(alpha) * sinc^2(u_1) * sinc^2(u_2) * visibility

    where:
        - A = length_edge_1_m * length_edge_2_m  (plate area)
        - alpha = angle between plate normal and the bistatic bisector hat(n_s2rx - n_tx2s)
        - u_i = length_edge_i_m / lambda * (n_tx2s - n_s2rx) . edge_i
        - visibility = 1 if plate normal points toward both Tx and Rx (back-face culling), 0 otherwise

    INPUTS:
        n_plate:           [np.ndarray, shape (n, 3)]   Outward plate normal (unit)
        edge_1, edge_2:    [np.ndarray, shape (n, 3)]   Two perpendicular edge directions (unit) in plate plane
        length_edge_1_m:   [float]                      Plate length along edge_1 [m]
        length_edge_2_m:   [float]                      Plate length along edge_2 [m]
        n_tx2s:            [np.ndarray, shape (n, 3)]   Tx -> surface unit direction
        n_s2rx:            [np.ndarray, shape (n, 3)]   Surface -> Rx unit direction
        wavelength_m:      [float]                      Carrier wavelength [m]

    OUTPUT:
        rcs_m2: [np.ndarray, shape (n,)]   Per-sample plate RCS [m^2]; 0 when not visible
    =========================================================================================================
    """
    n_samples = n_plate.shape[0]
    rcs = np.zeros(n_samples, dtype=float)

    # Back-face culling: plate normal must point toward both Tx and Rx.
    cos_to_tx = np.einsum("ij,ij->i", n_plate, -n_tx2s)    # plate normal . (surface -> Tx) > 0 => Tx side
    cos_to_rx = np.einsum("ij,ij->i", n_plate, n_s2rx)     # plate normal . (surface -> Rx) > 0 => Rx side
    visible = (cos_to_tx > 0.0) & (cos_to_rx > 0.0)
    if not visible.any():
        return rcs

    # Bistatic bisector direction (specular "mirror normal" for this geometry).
    bisector = n_s2rx - n_tx2s                              # (n, 3)
    bisector_mag = np.linalg.norm(bisector, axis=1)
    have_bisector = visible & (bisector_mag > 1e-12)
    if not have_bisector.any():
        return rcs

    bisector_hat = np.zeros_like(bisector)
    bisector_hat[have_bisector] = bisector[have_bisector] / bisector_mag[have_bisector, None]
    cos_alpha = np.einsum("ij,ij->i", n_plate, bisector_hat)   # plate normal . bisector direction

    # Form factor: bistatic path mismatch projected onto plate edges.
    delta = n_tx2s - n_s2rx                                 # (n, 3); -bisector*|...|
    u_1 = length_edge_1_m / wavelength_m * np.einsum("ij,ij->i", delta, edge_1)
    u_2 = length_edge_2_m / wavelength_m * np.einsum("ij,ij->i", delta, edge_2)

    area = length_edge_1_m * length_edge_2_m
    sigma_peak = 4.0 * np.pi * area**2 / wavelength_m**2

    rcs[have_bisector] = (
        sigma_peak
        * cos_alpha[have_bisector] ** 2
        * np.sinc(u_1[have_bisector]) ** 2     # np.sinc(x) = sin(pi x)/(pi x), matches PO form factor
        * np.sinc(u_2[have_bisector]) ** 2
    )
    return rcs


def dihedral_rcs(
    edge_hat: np.ndarray,
    leg_1_normal: np.ndarray,
    leg_2_normal: np.ndarray,
    length_edge_m: float,
    leg_dimension_m: float,
    n_tx2s: np.ndarray,
    n_s2rx: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """
    =========================================================================================================
    Bistatic RCS of a 90-deg dihedral corner reflector.

        sigma = (8 pi (L H)^2 / lambda^2) * sinc^2(u_edge) * wedge_visibility

    where:
        - L = length_edge_m            (length along the dihedral edge)
        - H = leg_dimension_m          (perpendicular leg height)
        - u_edge = L / lambda * (n_s2rx + n_tx2s) . edge_hat
                   At ideal retro (n_s2rx == -n_tx2s) this is zero, giving the peak.
        - wedge_visibility = 1 only when both -n_tx2s (towards Tx) and n_s2rx (towards Rx) have
          positive components along BOTH leg normals — i.e., both endpoints are inside the open
          quadrant of the corner. Outside this quadrant the dihedral is shadowed by itself.

    INPUTS:
        edge_hat:         [np.ndarray, shape (n, 3)]   Unit vector along dihedral edge
        leg_1_normal:     [np.ndarray, shape (n, 3)]   Outward normal of leg 1 (unit, perpendicular to edge)
        leg_2_normal:     [np.ndarray, shape (n, 3)]   Outward normal of leg 2 (unit, perpendicular to both)
        length_edge_m:    [float]                      Length along the edge [m]
        leg_dimension_m:  [float]                      Leg height perpendicular to edge [m]
        n_tx2s:           [np.ndarray, shape (n, 3)]   Tx -> surface unit direction
        n_s2rx:           [np.ndarray, shape (n, 3)]   Surface -> Rx unit direction
        wavelength_m:     [float]                      Carrier wavelength [m]

    OUTPUT:
        rcs_m2: [np.ndarray, shape (n,)]   Per-sample dihedral RCS [m^2]; 0 outside wedge
    =========================================================================================================
    """
    n_samples = edge_hat.shape[0]
    rcs = np.zeros(n_samples, dtype=float)

    incoming_outward = -n_tx2s                                          # surface -> Tx
    outgoing_outward = n_s2rx                                           # surface -> Rx

    vis = (
        (np.einsum("ij,ij->i", incoming_outward, leg_1_normal) > 0.0)
        & (np.einsum("ij,ij->i", incoming_outward, leg_2_normal) > 0.0)
        & (np.einsum("ij,ij->i", outgoing_outward, leg_1_normal) > 0.0)
        & (np.einsum("ij,ij->i", outgoing_outward, leg_2_normal) > 0.0)
    )
    if not vis.any():
        return rcs

    edge_mismatch = np.einsum("ij,ij->i", n_s2rx + n_tx2s, edge_hat)    # 0 at ideal retro
    u_edge = length_edge_m / wavelength_m * edge_mismatch

    sigma_peak = 8.0 * np.pi * (length_edge_m * leg_dimension_m) ** 2 / wavelength_m**2
    rcs[vis] = sigma_peak * np.sinc(u_edge[vis]) ** 2
    return rcs
