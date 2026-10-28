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
        f_D(r)  = (1 / lambda) [
                      n_tx2s . (v_tx - v_s) - n_s2rx . (v_rx - v_s)
                  ]                                               Doppler [Hz]

    the tangent-plane gradients are

        grad L   = n_tx2s - n_s2rx                                (then projected onto the tangent plane)
        grad f_D = (1 / lambda) [
                       v_tx,rel,perp / R_t + v_rx,rel,perp / R_r
                       + omega_E x (n_tx2s - n_s2rx)
                   ]

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

    delay_doppler_cell_samples samples the cell that contains a given surface point (a ship) directly on
    the ellipsoid. It stays exact at the specular point and where grad L || grad f_D, and it includes the
    ambiguity twin (the second part of the surface with the same delay and Doppler).

    Production calls supply the surface position, so ``v_s = omega_E x r`` is included. Omitting
    ``surface_positions_eci`` retains the earlier stationary-in-ECI convention for direct API callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymap3d

from object_definitions.Constants_def import Constants

if TYPE_CHECKING:
    from functions.link_budget import BistaticGeometry

_COARSE_RAYS = 360                 # first-pass rays that size the azimuth sampling of the cell integration
_RADIUS_TOLERANCE_M = 1.0e-3       # bisection end width for the band radii [m]
_FD_STEP_M = 1.0                   # radial finite-difference step for the area element [m]
_FD_STEP_RAD = 1.0e-7              # azimuthal finite-difference step for the area element [rad]


def delay_doppler_cell_area(
    geometry: BistaticGeometry,
    tx_velocities_eci: np.ndarray,
    rx_velocities_eci: np.ndarray,
    wavelength_m: float,
    chip_length_m: float,
    coherent_integration_time_s: float,
    footprint_area_m2: np.ndarray | float | None = None,
    surface_positions_eci: np.ndarray | None = None,
) -> dict:
    """Delay-Doppler resolution-cell area per sample from the (delay, Doppler) -> surface Jacobian.

    Parameters
    ----------
    geometry : BistaticGeometry
        Per-sample directions and ranges (n samples).
    tx_velocities_eci : numpy.ndarray
        Transmitter ECI velocities [m/s], shape ``(3, n)``.
    rx_velocities_eci : numpy.ndarray
        Receiver ECI velocities [m/s], shape ``(3, n)``.
    wavelength_m : float
        Carrier wavelength [m].
    chip_length_m : float
        One code-chip path length ``delta_L = c / R_chip`` [m].
    coherent_integration_time_s : float
        Coherent integration time ``T_coh`` [s]; the Doppler bin is ``1 / T_coh``.
    footprint_area_m2 : numpy.ndarray or float or None
        Upper-bound cap (antenna footprint area) [m^2]; ``None`` means no cap.
    surface_positions_eci : numpy.ndarray or None
        Surface positions [m], shape ``(n, 3)``. When supplied, rigid Earth rotation is included in
        the Doppler field and its spatial gradient. ``None`` retains a stationary-in-ECI surface.

    Returns
    -------
    dict
        Per-sample ``(n,)`` arrays: ``cell_area_m2`` (footprint-capped cell area),
        ``uncapped_cell_area_m2``, ``delay_width_m`` (``delta_L / |grad L|``),
        ``doppler_width_m`` (``delta_fD / |grad f_D|``), ``delay_gradient`` (``|grad L|``),
        ``doppler_gradient_hz_m`` (``|grad f_D|``), ``abs_jacobian_det_hz_m``
        (``|grad L x grad f_D|``), and ``footprint_capped`` (True where the cap or a near-specular
        degeneracy set the area to the footprint).
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

    include_surface_rotation = surface_positions_eci is not None
    if include_surface_rotation:
        surface_positions = np.asarray(surface_positions_eci, dtype=float)
        if surface_positions.shape != (n, 3):
            raise ValueError(f"surface_positions_eci must have shape ({n}, 3); got {surface_positions.shape}")
        omega_eci = np.array([0.0, 0.0, Constants.EARTH_ROTATION_RATE_RAD_S])
        v_surface = np.cross(np.broadcast_to(omega_eci, surface_positions.shape), surface_positions)
    else:
        omega_eci = np.zeros(3, dtype=float)
        v_surface = np.zeros((n, 3), dtype=float)

    delta_L = float(chip_length_m)                            # one chip in path length [m]
    delta_fD = 1.0 / float(coherent_integration_time_s)       # one Doppler bin [Hz]

    def _project_tangent(vectors: np.ndarray) -> np.ndarray:
        normal_component = np.einsum("ij,ij->i", vectors, n_surface)   # vectors . n_surface
        return vectors - normal_component[:, None] * n_surface         # remove the normal part

    # Delay-field surface gradient: grad L = n_tx2s - n_s2rx, projected onto the tangent plane.
    grad_L = _project_tangent(n_tx2s - n_s2rx)

    # Doppler-field surface gradient. With a rotating surface,
    # f_D = [n_i.(v_tx-v_s) - n_o.(v_rx-v_s)] / lambda and grad(v_s) contributes
    # omega x (n_i-n_o) in addition to the two LOS-derivative terms.
    with np.errstate(invalid="ignore", divide="ignore"):
        v_tx_relative = v_tx - v_surface
        v_rx_relative = v_rx - v_surface
        v_tx_perp = v_tx_relative - np.einsum("ij,ij->i", v_tx_relative, n_tx2s)[:, None] * n_tx2s
        v_rx_perp = v_rx_relative - np.einsum("ij,ij->i", v_rx_relative, n_s2rx)[:, None] * n_s2rx
        rotation_term = np.cross(np.broadcast_to(omega_eci, n_tx2s.shape), n_tx2s - n_s2rx)
        grad_fD = _project_tangent(
            (v_tx_perp / range_tx[:, None] + v_rx_perp / range_rx[:, None] + rotation_term) / wavelength_m
        )

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
        "surface_rotation_included": include_surface_rotation,
    }


def _surface_from_specular(lat_sp: float, lon_sp: float, radius_m: np.ndarray, azimuth_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS-84 surface points and outward normals (ECEF) at a spherical distance and azimuth from the specular point."""
    central = radius_m / Constants.R_EARTH_MEAN                                   # central angle [rad]
    lat = np.arcsin(np.sin(lat_sp) * np.cos(central) + np.cos(lat_sp) * np.sin(central) * np.cos(azimuth_rad))
    lon = lon_sp + np.arctan2(np.sin(azimuth_rad) * np.sin(central) * np.cos(lat_sp), np.cos(central) - np.sin(lat_sp) * np.sin(lat))
    x, y, z = pymap3d.geodetic2ecef(lat, lon, 0.0, deg=False)
    normal = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)   # geodetic normal
    return np.stack((x, y, z), axis=-1), normal


def delay_doppler_cell_samples(
    tx_pos_ecef: np.ndarray,
    tx_vel_ecef: np.ndarray,
    rx_pos_ecef: np.ndarray,
    rx_vel_ecef: np.ndarray,
    specular_ecef: np.ndarray,
    point_ecef: np.ndarray,
    wavelength_m: float,
    chip_length_m: float,
    coherent_integration_time_s: float,
    arc_step_m: float,
    radial_step_m: float,
) -> dict:
    """Surface samples of the delay-Doppler cell that contains a surface point, on the WGS-84 ellipsoid.

    The cell is the visible surface where the path length ``L`` and the Doppler ``f_D`` lie within one chip
    and one Doppler bin centred on the values at ``point_ecef``. Each ray from the specular point crosses an
    iso-delay contour once, so per ray the delay band runs between the radii where ``L = L0 -/+ delta_L / 2``
    (a disk when the band contains the specular point). Band samples are kept where ``f_D`` lies in the bin,
    so the contour curvature and the ambiguity twin are part of the cell. Vectors are Earth-fixed (surface
    at rest); velocities are relative to the rotating Earth.

    Parameters
    ----------
    tx_pos_ecef, tx_vel_ecef, rx_pos_ecef, rx_vel_ecef : numpy.ndarray
        Transmitter / receiver ECEF position [m] and Earth-relative velocity [m/s], shape ``(3,)``.
    specular_ecef : numpy.ndarray
        Specular point of the Tx/Rx pair on the ellipsoid (minimum of ``L``), shape ``(3,)``.
    point_ecef : numpy.ndarray
        Surface point that defines the cell (the ship), shape ``(3,)``.
    wavelength_m : float
        Carrier wavelength [m].
    chip_length_m : float
        One code-chip path length ``delta_L = c / R_chip`` [m].
    coherent_integration_time_s : float
        Coherent integration time ``T_coh`` [s]; the Doppler bin is ``1 / T_coh``.
    arc_step_m : float
        Target sample spacing along the iso-delay contour [m].
    radial_step_m : float
        Maximum sample spacing across the delay band [m].

    Returns
    -------
    dict
        ``positions_ecef`` / ``normals_ecef`` ``(m, 3)`` and ``area_m2`` ``(m,)`` of the cell samples,
        ``cell_area_m2`` (total), ``segment_areas_m2`` (one entry per disjoint part of the cell along the
        contour), ``point_segment_area_m2`` (the part that contains ``point_ecef``), ``specular_in_cell``,
        ``n_azimuth`` and ``n_radial`` (sample counts).
    """
    tx = np.asarray(tx_pos_ecef, dtype=float)
    v_tx = np.asarray(tx_vel_ecef, dtype=float)
    rx = np.asarray(rx_pos_ecef, dtype=float)
    v_rx = np.asarray(rx_vel_ecef, dtype=float)
    sp = np.asarray(specular_ecef, dtype=float)
    p0 = np.asarray(point_ecef, dtype=float)

    def path_length(p):
        return np.linalg.norm(p - tx, axis=-1) + np.linalg.norm(rx - p, axis=-1)                  # L = R_t + R_r

    def doppler(p):
        n_in = (p - tx) / np.linalg.norm(p - tx, axis=-1)[..., None]                               # Tx -> surface
        n_out = (rx - p) / np.linalg.norm(rx - p, axis=-1)[..., None]                              # surface -> Rx
        return (n_in @ v_tx - n_out @ v_rx) / wavelength_m                                         # f_D, surface at rest

    def visible(p, n):
        return (np.sum(n * (tx - p), axis=-1) > 0.0) & (np.sum(n * (rx - p), axis=-1) > 0.0)

    L0 = float(path_length(p0))
    f0 = float(doppler(p0))
    L_sp = float(path_length(sp))
    L_lo, L_hi = L0 - 0.5 * chip_length_m, L0 + 0.5 * chip_length_m
    half_bin_hz = 0.5 / coherent_integration_time_s
    lat_sp, lon_sp, _ = pymap3d.ecef2geodetic(sp[0], sp[1], sp[2], deg=False)

    # Any surface point the receiver sees lies within the Rx horizon, so this radius bounds the rays.
    polar_radius_m = Constants.RADIUS_EARTH_EQUAT * (1.0 - Constants.EARTH_FLATTENING)
    rx_norm = np.linalg.norm(rx)
    psi_sp = np.arccos(np.clip(np.dot(rx, sp) / (rx_norm * np.linalg.norm(sp)), -1.0, 1.0))        # SP to Rx sub-point
    psi_horizon = np.arccos(np.clip(polar_radius_m / rx_norm, -1.0, 1.0))                          # Rx sub-point to horizon
    r_max = Constants.R_EARTH_MEAN * (psi_sp + psi_horizon)
    n_iter = int(np.ceil(np.log2(r_max / _RADIUS_TOLERANCE_M)))

    def band_radius(target, azimuth):
        """Smallest ray radius with L >= target (bisection; L grows along each ray), NaN if not reached."""
        lo = np.zeros_like(azimuth)
        hi = np.full_like(azimuth, r_max)
        reachable = path_length(_surface_from_specular(lat_sp, lon_sp, hi, azimuth)[0]) >= target
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            below = path_length(_surface_from_specular(lat_sp, lon_sp, mid, azimuth)[0]) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return np.where(reachable, hi, np.nan)

    # Size the azimuth sampling from the outer contour length.
    az_coarse = 2.0 * np.pi * (np.arange(_COARSE_RAYS) + 0.5) / _COARSE_RAYS
    contour = _surface_from_specular(lat_sp, lon_sp, band_radius(L_hi, az_coarse), az_coarse)[0]
    perimeter_m = np.nansum(np.linalg.norm(contour - np.roll(contour, 1, axis=0), axis=1))
    n_az = max(_COARSE_RAYS, int(np.ceil(perimeter_m / arc_step_m)))
    d_az = 2.0 * np.pi / n_az
    azimuth = d_az * (np.arange(n_az) + 0.5)

    r_out = band_radius(L_hi, azimuth)
    r_in = band_radius(L_lo, azimuth) if L_lo > L_sp else np.zeros(n_az)                          # disk around the SP
    thickness = r_out - r_in
    n_rad = max(2, int(np.ceil(np.nanmax(thickness) / radial_step_m)))
    frac = (np.arange(n_rad) + 0.5) / n_rad
    radius = r_in[:, None] + frac[None, :] * thickness[:, None]                                    # (n_az, n_rad) midpoints
    az_grid = np.broadcast_to(azimuth[:, None], radius.shape)

    points, normals = _surface_from_specular(lat_sp, lon_sp, radius, az_grid)
    dp_dr = (_surface_from_specular(lat_sp, lon_sp, radius + _FD_STEP_M, az_grid)[0]
             - _surface_from_specular(lat_sp, lon_sp, radius - _FD_STEP_M, az_grid)[0]) / (2.0 * _FD_STEP_M)
    dp_daz = (_surface_from_specular(lat_sp, lon_sp, radius, az_grid + _FD_STEP_RAD)[0]
              - _surface_from_specular(lat_sp, lon_sp, radius, az_grid - _FD_STEP_RAD)[0]) / (2.0 * _FD_STEP_RAD)
    area = np.linalg.norm(np.cross(dp_dr, dp_daz), axis=-1) * (thickness / n_rad)[:, None] * d_az  # |dp/dr x dp/daz| dr daz

    with np.errstate(invalid="ignore"):
        keep = (
            np.isfinite(points).all(axis=-1) & np.isfinite(area)
            & visible(points, normals)
            & (np.abs(doppler(points) - f0) <= half_bin_hz)
        )

    # Disjoint parts of the cell along the contour (the point's own part and its ambiguity twin).
    ray_in = keep.any(axis=1)
    ray_area = np.where(keep, area, 0.0).sum(axis=1)
    labels = np.full(n_az, -1)
    if ray_in.all():
        labels[:] = 0
    elif ray_in.any():
        start = int(np.argmin(ray_in))                                                             # a ray outside the cell
        rolled = np.roll(ray_in, -start)
        run_start = rolled & ~np.concatenate(([False], rolled[:-1]))
        labels = np.roll(np.where(rolled, np.cumsum(run_start) - 1, -1), start)
    segment_areas = np.bincount(labels[labels >= 0], weights=ray_area[labels >= 0]) if ray_in.any() else np.zeros(0)

    lat_p, lon_p, _ = pymap3d.ecef2geodetic(p0[0], p0[1], p0[2], deg=False)
    az_point = np.arctan2(np.sin(lon_p - lon_sp) * np.cos(lat_p),
                          np.cos(lat_sp) * np.sin(lat_p) - np.sin(lat_sp) * np.cos(lat_p) * np.cos(lon_p - lon_sp))
    k_point = int(np.floor((az_point % (2.0 * np.pi)) / d_az)) % n_az
    near = labels[(k_point + np.arange(-2, 3)) % n_az]                                            # tolerate an edge ray
    point_label = int(near[near >= 0][0]) if (near >= 0).any() else -1

    sp_normal = _surface_from_specular(lat_sp, lon_sp, np.array(0.0), np.array(0.0))[1]
    specular_in_cell = bool(L_sp >= L_lo and abs(float(doppler(sp)) - f0) <= half_bin_hz and visible(sp, sp_normal))

    return {
        "positions_ecef": points[keep],
        "normals_ecef": normals[keep],
        "area_m2": area[keep],
        "cell_area_m2": float(area[keep].sum()),
        "segment_areas_m2": segment_areas,
        "point_segment_area_m2": float(segment_areas[point_label]) if point_label >= 0 else float("nan"),
        "specular_in_cell": specular_in_cell,
        "n_azimuth": n_az,
        "n_radial": n_rad,
    }
