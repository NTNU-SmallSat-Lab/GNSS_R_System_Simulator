"""Bistatic RCS helpers for ship-as-target modelling.

flat_plate_rcs:  Scalar physical-optics RCS of a rectangular PEC plate with back-face culling.
dihedral_rcs:    Two-path double-bounce physical-optics RCS of a 90-deg dihedral with wedge visibility.

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
    """Bistatic RCS of a rectangular PEC flat plate via physical optics.

    The cross-section is ``sigma = (4 pi A^2 / lambda^2) * ((cos theta_i + cos theta_s) / 2)^2 * sinc^2(u_1) *
    sinc^2(u_2) * visibility``, with plate area ``A = length_edge_1_m * length_edge_2_m``, ``theta_i`` / ``theta_s``
    the angles of the directions to Tx / Rx from the plate normal (the obliquity equals ``cos(alpha) cos(beta / 2)``,
    ``alpha`` the angle between the plate normal and the bistatic bisector, ``beta`` the bistatic angle),
    ``u_i = length_edge_i_m / lambda * (n_tx2s - n_s2rx) . edge_i``, and a back-face visibility that is 1 only when
    the plate normal faces both Tx and Rx.

    Parameters
    ----------
    n_plate : numpy.ndarray
        Outward plate normal (unit), shape ``(n, 3)``.
    edge_1, edge_2 : numpy.ndarray
        Two perpendicular in-plane edge directions (unit), shape ``(n, 3)``.
    length_edge_1_m, length_edge_2_m : float
        Plate lengths along ``edge_1`` / ``edge_2`` [m].
    n_tx2s, n_s2rx : numpy.ndarray
        Tx-to-surface and surface-to-Rx unit directions, shape ``(n, 3)``.
    wavelength_m : float
        Carrier wavelength [m].

    Returns
    -------
    numpy.ndarray
        Per-sample plate RCS [m^2], shape ``(n,)``; 0 when not visible.
    """
    n_samples = n_plate.shape[0]
    rcs = np.zeros(n_samples, dtype=float)

    # Back-face culling: plate normal must point toward both Tx and Rx.
    cos_to_tx = np.einsum("ij,ij->i", n_plate, -n_tx2s)    # plate normal . (surface -> Tx) > 0 => Tx side
    cos_to_rx = np.einsum("ij,ij->i", n_plate, n_s2rx)     # plate normal . (surface -> Rx) > 0 => Rx side
    visible = (cos_to_tx > 0.0) & (cos_to_rx > 0.0)
    if not visible.any():
        return rcs

    # Physical-optics obliquity: projected aperture (cos theta_i + cos theta_s) / 2 = cos(alpha) cos(beta / 2).
    obliquity = 0.5 * (cos_to_tx + cos_to_rx)

    # Form factor: bistatic path mismatch projected onto plate edges.
    delta = n_tx2s - n_s2rx                                 # (n, 3); -bisector*|...|
    u_1 = length_edge_1_m / wavelength_m * np.einsum("ij,ij->i", delta, edge_1)
    u_2 = length_edge_2_m / wavelength_m * np.einsum("ij,ij->i", delta, edge_2)

    area = length_edge_1_m * length_edge_2_m
    sigma_peak = 4.0 * np.pi * area**2 / wavelength_m**2

    rcs[visible] = (
        sigma_peak
        * obliquity[visible] ** 2
        * np.sinc(u_1[visible]) ** 2           # np.sinc(x) = sin(pi x)/(pi x), matches PO form factor
        * np.sinc(u_2[visible]) ** 2
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
    """Bistatic double-bounce RCS of a 90-degree dihedral corner reflector with two equal legs.

    Each double-bounce path (leg 2 then leg 1, and leg 1 then leg 2) is a geometric-optics first bounce
    followed by a physical-optics last bounce, and the two paths add coherently:
    ``sigma = (4 pi L^2 / lambda^2) * sinc^2(u_edge) * |S_1 + S_2|^2 * wedge_visibility``, with
    ``S_1 = (a_1 + b_1) / 2 * s_1 * exp(i pi v_1) * sinc(v_1)``, ``s_1 = H * min(1, a_2 / a_1)``,
    ``v_1 = s_1 / lambda * (a_2 - b_2)`` (``S_2`` with indices 1 and 2 swapped), and ``u_edge = L / lambda *
    (n_tx2s - n_s2rx) . edge_hat``. Here ``L = length_edge_m``, ``H = leg_dimension_m``, and ``a_i`` / ``b_i``
    are the components of the directions to Tx / Rx along leg normal ``i``. The double bounce keeps the
    edge-parallel direction component and reverses both leg-normal components, so the return peaks only
    toward the Tx direction mirrored about the plane perpendicular to the edge, with lobe widths of order
    ``lambda / L`` along and ``lambda / (H min(a_1, a_2))`` across the edge. The peak is
    ``16 pi (L H min(a_1, a_2))^2 / lambda^2`` (``8 pi (L H)^2 / lambda^2`` at 45 deg). The wedge visibility
    is 1 only when the directions to both Tx and Rx lie inside the open corner quadrant (positive along both
    leg normals); outside it the dihedral shadows itself. The legs are flat, smooth and perfectly
    conducting, so this is not a calibrated absolute ship RCS.

    Parameters
    ----------
    edge_hat : numpy.ndarray
        Unit vector along the dihedral edge, shape ``(n, 3)``.
    leg_1_normal, leg_2_normal : numpy.ndarray
        Outward unit normals of the two legs, shape ``(n, 3)``.
    length_edge_m : float
        Length along the edge [m].
    leg_dimension_m : float
        Leg height perpendicular to the edge [m].
    n_tx2s, n_s2rx : numpy.ndarray
        Tx-to-surface and surface-to-Rx unit directions, shape ``(n, 3)``.
    wavelength_m : float
        Carrier wavelength [m].

    Returns
    -------
    numpy.ndarray
        Per-sample dihedral RCS [m^2], shape ``(n,)``; 0 outside the wedge.
    """
    n_samples = edge_hat.shape[0]
    rcs = np.zeros(n_samples, dtype=float)

    incoming_outward = -n_tx2s                                          # surface -> Tx
    outgoing_outward = n_s2rx                                           # surface -> Rx
    a_1 = np.einsum("ij,ij->i", incoming_outward, leg_1_normal)
    a_2 = np.einsum("ij,ij->i", incoming_outward, leg_2_normal)
    b_1 = np.einsum("ij,ij->i", outgoing_outward, leg_1_normal)
    b_2 = np.einsum("ij,ij->i", outgoing_outward, leg_2_normal)

    vis = (a_1 > 0.0) & (a_2 > 0.0) & (b_1 > 0.0) & (b_2 > 0.0)
    if not vis.any():
        return rcs
    a_1, a_2, b_1, b_2 = a_1[vis], a_2[vis], b_1[vis], b_2[vis]

    edge_mismatch = np.einsum("ij,ij->i", n_tx2s[vis] - n_s2rx[vis], edge_hat[vis])   # 0 when the edge component is preserved
    u_edge = length_edge_m / wavelength_m * edge_mismatch

    # Last bounce on leg 1 (after leg 2) and on leg 2 (after leg 1)
    s_1 = leg_dimension_m * np.minimum(1.0, a_2 / a_1)                  # lit extent of leg 1 across the edge
    s_2 = leg_dimension_m * np.minimum(1.0, a_1 / a_2)                  # lit extent of leg 2 across the edge
    v_1 = s_1 / wavelength_m * (a_2 - b_2)                              # 0 when the leg-2 component is reversed back
    v_2 = s_2 / wavelength_m * (a_1 - b_1)                              # 0 when the leg-1 component is reversed back
    path_1 = 0.5 * (a_1 + b_1) * s_1 * np.exp(1j * np.pi * v_1) * np.sinc(v_1)   # obliquity * aperture * form factor
    path_2 = 0.5 * (a_2 + b_2) * s_2 * np.exp(1j * np.pi * v_2) * np.sinc(v_2)

    rcs[vis] = 4.0 * np.pi * length_edge_m**2 / wavelength_m**2 * np.sinc(u_edge) ** 2 * np.abs(path_1 + path_2) ** 2
    return rcs
