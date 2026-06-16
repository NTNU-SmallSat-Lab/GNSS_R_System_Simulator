from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pymap3d

from functions.rotations import eci_to_antenna_rotation_from_state
from object_definitions.Constants_def import Constants

if TYPE_CHECKING:
    from object_definitions.Rx_antenna_def import RxAntenna
    from object_definitions.Target_def import Target


@dataclass(slots=True)
class BistaticLinkBudgetSettings:
    """
    Configuration values for the bistatic GNSS-R link budget.

    Field-by-field source of truth for the `[LINK BUDGET]` + `[OCEAN]` INI sections;
    field names and units defined here mirror the INI keys. The physical ocean/loss fields are
    mandatory so a missing value raises instead of silently falling back.
    The INI loader is strict (missing keys raise); the thermal-noise + detection-threshold fields
    are consumed only by the moving-target detector and default to None (they are omitted from the
    area-coverage config, which is geometric). Patch-integration controls have conservative defaults
    and can be overridden from `[LINK BUDGET]`. Direct construction with `settings=None` is not accepted.

    ATTRIBUTES:
        system_losses_db:          [float]  Lumped implementation losses [dB]
        polarization_losses_db:    [float]  Receiver-side polarization mismatch losses [dB]
        ocean_wave_height_m:       [float]  RMS wave height [m]  (Rayleigh roughness diagnostic)
        ocean_wind_speed_m_s:      [float]  Wind speed at 10 m [m/s]  (drives Cox-Munk slope variances)
        ocean_wind_direction_deg:  [float]  Wind direction (meteorological FROM bearing) [deg]
        ocean_sea_temperature_c:   [float]  Sea surface temperature [degC]
        ocean_salinity_psu:        [float]  Salinity (practical salinity units) [psu]
        ocean_inverse_wave_age:    [float]  Inverse wave age Omega_c [-] (0.84 fully developed; drives the Elfouhaily Bragg spectrum)
        noise_bandwidth_hz:        [float | None]  Thermal-noise bandwidth for SNR [Hz] (moving-target only; None for area coverage)
        system_noise_temperature_k:[float | None]  Equivalent input/system noise temperature [K] (moving-target only)
        detection_sinr_threshold_db:[float | None] Ship detectability threshold for P_ship / (P_ocean + P_noise) [dB] (moving-target only)
        ocean_patch_integration_enabled:[bool] Enable footprint patch integration for antenna-limited ocean clutter.
        ocean_patch_grid_size:     [int]    Number of cells per tangent-plane footprint axis for patch integration.

    NOTE: GNSS transmitter EIRP is per-(constellation, band) and lives in
    `Constants.GNSS_EIRP_DBW` — it is not a link-budget config knob.
    """

    system_losses_db: float
    polarization_losses_db: float
    ocean_wave_height_m: float
    ocean_wind_speed_m_s: float
    ocean_wind_direction_deg: float
    ocean_sea_temperature_c: float
    ocean_salinity_psu: float
    ocean_inverse_wave_age: float
    # Thermal-noise + detection-threshold settings: consumed only by the moving-target detector.
    # Area coverage is geometric and ignores them, so they are optional (omitted from that INI).
    noise_bandwidth_hz: float | None = None
    system_noise_temperature_k: float | None = None
    detection_sinr_threshold_db: float | None = None
    # LB-2b patch integration: when enabled and an Rx antenna is present, ocean clutter is summed
    # over finite surface patches with per-patch geometry and off-axis antenna gain. The scalar
    # glistening-area cap remains the fallback for antenna=None, custom targets, or invalid footprints.
    ocean_patch_integration_enabled: bool = False
    ocean_patch_grid_size: int = 9

    def __post_init__(self) -> None:
        if not isinstance(self.ocean_patch_integration_enabled, bool):
            raise TypeError("ocean_patch_integration_enabled must be a bool")
        if not isinstance(self.ocean_patch_grid_size, int):
            raise TypeError("ocean_patch_grid_size must be an int")
        if self.ocean_patch_grid_size < 3:
            raise ValueError(f"ocean_patch_grid_size must be >= 3; got {self.ocean_patch_grid_size}")
        if self.ocean_patch_grid_size > 101:
            raise ValueError(f"ocean_patch_grid_size must be <= 101 to avoid accidental runaway runtimes; got {self.ocean_patch_grid_size}")
        # Validate the noise/detection settings only when supplied (moving-target mode).
        if self.noise_bandwidth_hz is not None and self.noise_bandwidth_hz <= 0.0:
            raise ValueError(f"noise_bandwidth_hz must be > 0; got {self.noise_bandwidth_hz}")
        if self.system_noise_temperature_k is not None and self.system_noise_temperature_k <= 0.0:
            raise ValueError(f"system_noise_temperature_k must be > 0; got {self.system_noise_temperature_k}")
        if self.detection_sinr_threshold_db is not None and not np.isfinite(self.detection_sinr_threshold_db):
            raise ValueError(f"detection_sinr_threshold_db must be finite; got {self.detection_sinr_threshold_db}")


def coerce_link_budget_settings(settings: dict | BistaticLinkBudgetSettings) -> BistaticLinkBudgetSettings:
    if isinstance(settings, BistaticLinkBudgetSettings):
        return settings
    if isinstance(settings, dict):
        return BistaticLinkBudgetSettings(**settings)
    if settings is None:
        raise ValueError(
            "Link-budget/ocean settings are required but none were provided. Supply a "
            "BistaticLinkBudgetSettings (or the [LINK BUDGET] + [OCEAN] INI values) with every "
            "field set; there are no default fallbacks."
        )
    raise TypeError("settings must be a dict or BistaticLinkBudgetSettings")


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


def ellipsoid_surface_normals_ecef(surface_positions_ecef: np.ndarray) -> np.ndarray:
    """
    =========================================================================================================
    Compute outward WGS-84 ellipsoid surface normals in ECEF coordinates.

    INPUTS:
        surface_positions_ecef: [np.ndarray, shape (n, 3)]   Surface positions on/near the WGS-84 ellipsoid [m]

    OUTPUT:
        normals_ecef: [np.ndarray, shape (n, 3)]   Unit outward normal vectors in ECEF; invalid rows are NaN
    =========================================================================================================
    """
    surface_positions_ecef = np.asarray(surface_positions_ecef, dtype=float)
    if surface_positions_ecef.ndim != 2 or surface_positions_ecef.shape[1] != 3:
        raise ValueError(f"surface_positions_ecef must have shape (n, 3); got {surface_positions_ecef.shape}")

    a = Constants.RADIUS_EARTH_EQUAT
    b = Constants.RADIUS_EARTH_EQUAT * (1.0 - Constants.EARTH_FLATTENING)

    normals = np.full_like(surface_positions_ecef, np.nan, dtype=float)
    valid_positions = np.isfinite(surface_positions_ecef).all(axis=1)
    if not valid_positions.any():
        return normals

    normals_raw = np.column_stack(
        (
            surface_positions_ecef[valid_positions, 0] / a**2,
            surface_positions_ecef[valid_positions, 1] / a**2,
            surface_positions_ecef[valid_positions, 2] / b**2,
        )
    )
    normal_mag = np.linalg.norm(normals_raw, axis=1)
    valid_normals = normal_mag > 0.0

    valid_indices = np.where(valid_positions)[0][valid_normals]
    normals[valid_indices] = normals_raw[valid_normals] / normal_mag[valid_normals, None]
    return normals


def ellipsoid_surface_normals_eci(surface_positions_ecef: np.ndarray, times) -> np.ndarray:
    """
    =========================================================================================================
    Compute outward WGS-84 ellipsoid surface normals and rotate them from ECEF to ECI.

    INPUTS:
        surface_positions_ecef: [np.ndarray, shape (n, 3)]   Surface positions on/near the WGS-84 ellipsoid [m]
        times:                  [sequence[datetime]]         Timestamps matching the surface-position rows

    OUTPUT:
        normals_eci: [np.ndarray, shape (n, 3)]   Unit outward normal vectors in ECI; invalid rows are NaN
    =========================================================================================================
    """
    surface_positions_ecef = np.asarray(surface_positions_ecef, dtype=float)
    normals_ecef = ellipsoid_surface_normals_ecef(surface_positions_ecef)
    normals_eci = np.full_like(normals_ecef, np.nan, dtype=float)

    times_array = np.asarray(times, dtype=object)
    if times_array.shape[0] != surface_positions_ecef.shape[0]:
        raise ValueError(f"times length must match surface_positions_ecef rows; got {times_array.shape[0]} vs {surface_positions_ecef.shape[0]}")

    valid = np.isfinite(normals_ecef).all(axis=1)
    if not valid.any():
        return normals_eci

    normals_eci_valid = np.column_stack(
        pymap3d.ecef2eci(
            normals_ecef[valid, 0],
            normals_ecef[valid, 1],
            normals_ecef[valid, 2],
            list(times_array[valid]),
        )
    )
    normal_mag = np.linalg.norm(normals_eci_valid, axis=1)
    valid_normals = normal_mag > 0.0
    valid_indices = np.where(valid)[0][valid_normals]
    normals_eci[valid_indices] = normals_eci_valid[valid_normals] / normal_mag[valid_normals, None]
    return normals_eci


def _surface_tangent_basis(surface_normal: np.ndarray, reference_vector: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    normal = np.asarray(surface_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return None
    normal_mag = np.linalg.norm(normal)
    if normal_mag <= 0.0:
        return None
    n_hat = normal / normal_mag

    if reference_vector is not None and np.asarray(reference_vector).shape == (3,) and np.isfinite(reference_vector).all():
        ref = np.asarray(reference_vector, dtype=float)
        e1 = ref - np.dot(ref, n_hat) * n_hat
    else:
        e1 = np.zeros(3, dtype=float)

    if np.linalg.norm(e1) <= 1e-9:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fallback, n_hat)) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0])
        e1 = fallback - np.dot(fallback, n_hat) * n_hat

    e1_mag = np.linalg.norm(e1)
    if e1_mag <= 0.0:
        return None
    e1 /= e1_mag

    e2 = np.cross(n_hat, e1)
    e2_mag = np.linalg.norm(e2)
    if e2_mag <= 0.0:
        return None
    e2 /= e2_mag
    return e1, e2


def _intersect_sphere_rays(origin_eci: np.ndarray, directions_eci: np.ndarray, radius_m: float) -> np.ndarray:
    origin = np.asarray(origin_eci, dtype=float)
    directions = np.asarray(directions_eci, dtype=float)
    intersections = np.full(directions.shape, np.nan, dtype=float)

    if origin.shape != (3,) or directions.ndim != 2 or directions.shape[1] != 3:
        return intersections
    if not (np.isfinite(origin).all() and np.isfinite(radius_m) and radius_m > 0.0):
        return intersections

    direction_norms = np.linalg.norm(directions, axis=1)
    valid_dir = np.isfinite(direction_norms) & (direction_norms > 0.0)
    dirs = np.full_like(directions, np.nan, dtype=float)
    dirs[valid_dir] = directions[valid_dir] / direction_norms[valid_dir, None]

    b_quad = 2.0 * np.einsum("ij,j->i", dirs, origin)
    c_quad = np.dot(origin, origin) - radius_m**2
    discriminant = b_quad**2 - 4.0 * c_quad
    valid_disc = valid_dir & np.isfinite(discriminant) & (discriminant >= 0.0)
    if not valid_disc.any():
        return intersections

    sqrt_disc = np.sqrt(discriminant[valid_disc])
    roots_1 = (-b_quad[valid_disc] - sqrt_disc) / 2.0
    roots_2 = (-b_quad[valid_disc] + sqrt_disc) / 2.0
    positive_roots = np.column_stack((roots_1, roots_2))
    positive_roots = np.where(positive_roots > 0.0, positive_roots, np.nan)
    has_root = np.isfinite(positive_roots).any(axis=1)
    if not has_root.any():
        return intersections

    valid_indices = np.where(valid_disc)[0][has_root]
    t_vals = np.nanmin(positive_roots[has_root], axis=1)
    intersections[valid_indices] = origin + t_vals[:, None] * dirs[valid_indices]
    return intersections


def _polygon_area_xy(polygon_xy: np.ndarray) -> float:
    xy = np.asarray(polygon_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] != 2:
        return float("nan")
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points_in_polygon_xy(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    polygon = np.asarray(polygon_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return np.zeros(points.shape[0], dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        yi = poly_y[i]
        yj = poly_y[j]
        xi = poly_x[i]
        xj = poly_x[j]
        crosses = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def _project_tangent_offsets_to_local_sphere(center_eci: np.ndarray, offsets_eci: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    candidates = center_eci[np.newaxis, :] + offsets_eci
    norms = np.linalg.norm(candidates, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0) & np.isfinite(radius_m) & (radius_m > 0.0)

    positions = np.full_like(candidates, np.nan, dtype=float)
    normals = np.full_like(candidates, np.nan, dtype=float)
    positions[valid] = radius_m * candidates[valid] / norms[valid, None]
    normals[valid] = positions[valid] / radius_m
    return positions, normals


def _sample_antenna_footprint_patches(
    *,
    rx_pos_eci: np.ndarray,
    rx_vel_eci: np.ndarray,
    center_eci: np.ndarray,
    center_normal_eci: np.ndarray,
    antenna: RxAntenna,
    frequency_mhz: float,
    q_sat: np.ndarray,
    grid_size: int,
) -> dict | None:
    """Return tangent-plane patch samples inside the steered receiver footprint."""
    look_center_a = compute_rx_look_direction_antenna_frame(rx_pos_eci, rx_vel_eci, center_eci, antenna, q_sat=q_sat)
    steering_angles = antenna.resolve_steering_angles_for_direction(look_center_a)
    if steering_angles is None:
        return None

    try:
        contour_dirs_a = antenna._get_antenna_footprint(look_center_a, frequency_mhz)
    except (AttributeError, ValueError):
        return None
    if contour_dirs_a.ndim != 2 or contour_dirs_a.shape[0] < 4 or contour_dirs_a.shape[1] != 3:
        return None

    r_eci_to_antenna = eci_to_antenna_rotation_from_state(rx_pos_eci, rx_vel_eci, antenna, q_sat)
    if r_eci_to_antenna is None:
        return None
    contour_dirs_eci = (r_eci_to_antenna.T @ contour_dirs_a.T).T

    radius_m = np.linalg.norm(center_eci)
    contour_eci = _intersect_sphere_rays(rx_pos_eci, contour_dirs_eci, radius_m)
    valid_contour = np.isfinite(contour_eci).all(axis=1)
    contour_eci = contour_eci[valid_contour]
    if contour_eci.shape[0] < 4:
        return None

    basis = _surface_tangent_basis(center_normal_eci, rx_vel_eci)
    if basis is None:
        return None
    e1, e2 = basis
    contour_offsets = contour_eci - center_eci[np.newaxis, :]
    contour_xy = np.column_stack((contour_offsets @ e1, contour_offsets @ e2))
    if not np.isfinite(contour_xy).all():
        return None

    footprint_area_m2 = _polygon_area_xy(contour_xy)
    if not np.isfinite(footprint_area_m2) or footprint_area_m2 <= 0.0:
        return None

    x_min, y_min = np.min(contour_xy, axis=0)
    x_max, y_max = np.max(contour_xy, axis=0)
    if not (np.isfinite([x_min, x_max, y_min, y_max]).all() and x_max > x_min and y_max > y_min):
        return None

    x_edges = np.linspace(x_min, x_max, int(grid_size) + 1)
    y_edges = np.linspace(y_min, y_max, int(grid_size) + 1)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    if dx <= 0.0 or dy <= 0.0:
        return None

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="xy")
    candidate_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inside = _points_in_polygon_xy(candidate_xy, contour_xy)
    if not inside.any():
        return None

    patch_xy = candidate_xy[inside]
    offsets_eci = patch_xy[:, 0, None] * e1[np.newaxis, :] + patch_xy[:, 1, None] * e2[np.newaxis, :]
    patch_positions_eci, patch_normals_eci = _project_tangent_offsets_to_local_sphere(center_eci, offsets_eci, radius_m)
    valid_patch = np.isfinite(patch_positions_eci).all(axis=1) & np.isfinite(patch_normals_eci).all(axis=1)
    if not valid_patch.any():
        return None

    cell_area_m2 = dx * dy
    patch_positions_eci = patch_positions_eci[valid_patch]
    patch_normals_eci = patch_normals_eci[valid_patch]
    patch_areas_m2 = np.full(patch_positions_eci.shape[0], cell_area_m2, dtype=float)

    return {
        "patch_positions_eci": patch_positions_eci,
        "patch_normals_eci": patch_normals_eci,
        "patch_areas_m2": patch_areas_m2,
        "steering_azimuth_deg": float(steering_angles[0]),
        "steering_elevation_deg": float(steering_angles[1]),
        "footprint_area_m2": float(footprint_area_m2),
        "sampled_area_m2": float(patch_areas_m2.sum()),
    }


def _equivalent_rcs_from_received_power(
    *,
    received_power_w: float,
    range_tx_m: float,
    range_rx_m: float,
    rx_gain_dbi: float,
    wavelength_m: float,
    tx_eirp_dbw: float,
    settings: BistaticLinkBudgetSettings,
) -> float:
    if not (
        np.isfinite(received_power_w)
        and received_power_w > 0.0
        and np.isfinite(range_tx_m)
        and range_tx_m > 0.0
        and np.isfinite(range_rx_m)
        and range_rx_m > 0.0
        and np.isfinite(rx_gain_dbi)
        and np.isfinite(wavelength_m)
        and wavelength_m > 0.0
    ):
        return float("nan")

    tx_eirp_w = _linear_from_db(tx_eirp_dbw)
    rx_gain_lin = _linear_from_db(rx_gain_dbi)
    total_losses_lin = _linear_from_db(settings.system_losses_db + settings.polarization_losses_db)
    denom = tx_eirp_w * rx_gain_lin * wavelength_m**2
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")

    return float(received_power_w * ((4.0 * np.pi) ** 3) * range_tx_m**2 * range_rx_m**2 * total_losses_lin / denom)


def _center_rx_gain_dbi(
    rx_pos_eci: np.ndarray,
    rx_vel_eci: np.ndarray,
    surface_pos_eci: np.ndarray,
    antenna: RxAntenna,
    frequency_mhz: float,
    q_sat: np.ndarray,
) -> float:
    look_center_a = compute_rx_look_direction_antenna_frame(rx_pos_eci, rx_vel_eci, surface_pos_eci, antenna, q_sat=q_sat)
    return antenna.get_gain_dbi_for_direction(look_center_a, frequency_mhz)


def _compute_patch_integrated_ocean_rcs(
    *,
    target,
    center_geometry: BistaticGeometry,
    center_target_rcs,
    surface_positions_eci: np.ndarray,
    tx_positions_eci: np.ndarray,
    rx_positions_eci: np.ndarray,
    rx_velocities_eci: np.ndarray,
    antenna: RxAntenna,
    frequency_hz: float,
    tx_eirp_dbw: float,
    settings: BistaticLinkBudgetSettings,
    q_sat: np.ndarray,
) -> object:
    """Replace scalar ocean-area RCS with equivalent RCS from patch-summed received power."""
    from object_definitions.Target_def import TargetRcs

    num_positions = center_geometry.range_tx_m.shape[0]
    equivalent_rcs_m2 = np.array(center_target_rcs.rcs_m2, dtype=float, copy=True)
    info = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in center_target_rcs.info.items()}

    patch_used = np.zeros(num_positions, dtype=bool)
    patch_count = np.zeros(num_positions, dtype=int)
    patch_footprint_area = np.full(num_positions, np.nan, dtype=float)
    patch_sampled_area = np.full(num_positions, np.nan, dtype=float)
    patch_diffuse_rcs_sum = np.full(num_positions, np.nan, dtype=float)
    patch_bragg_rcs_sum = np.full(num_positions, np.nan, dtype=float)
    patch_diffuse_power = np.full(num_positions, np.nan, dtype=float)
    patch_coherent_power = np.full(num_positions, np.nan, dtype=float)
    patch_total_power = np.full(num_positions, np.nan, dtype=float)
    patch_equiv_diffuse_rcs = np.full(num_positions, np.nan, dtype=float)
    patch_equiv_total_rcs = np.full(num_positions, np.nan, dtype=float)

    frequency_mhz = float(frequency_hz) / 1e6
    wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz

    for i in range(num_positions):
        center_pos = np.asarray(surface_positions_eci[i], dtype=float)
        if not (
            np.isfinite(center_geometry.range_tx_m[i])
            and np.isfinite(center_geometry.range_rx_m[i])
            and np.isfinite(center_geometry.n_surface[i]).all()
            and np.isfinite(center_pos).all()
            and np.isfinite(tx_positions_eci[:, i]).all()
            and np.isfinite(rx_positions_eci[:, i]).all()
            and np.isfinite(rx_velocities_eci[:, i]).all()
            and np.isfinite(equivalent_rcs_m2[i])
        ):
            continue

        samples = _sample_antenna_footprint_patches(
            rx_pos_eci=rx_positions_eci[:, i],
            rx_vel_eci=rx_velocities_eci[:, i],
            center_eci=center_pos,
            center_normal_eci=center_geometry.n_surface[i],
            antenna=antenna,
            frequency_mhz=frequency_mhz,
            q_sat=q_sat,
            grid_size=settings.ocean_patch_grid_size,
        )
        if samples is None:
            continue

        patch_positions = samples["patch_positions_eci"]
        patch_normals = samples["patch_normals_eci"]
        patch_count_i = patch_positions.shape[0]
        tx_patch = np.repeat(tx_positions_eci[:, i:i + 1], patch_count_i, axis=1)
        rx_patch = np.repeat(rx_positions_eci[:, i:i + 1], patch_count_i, axis=1)

        patch_geometry = build_bistatic_geometry(tx_patch, rx_patch, patch_positions, patch_normals)
        patch_rcs = target.compute_rcs(
            patch_geometry,
            wavelength_m,
            illuminated_area_m2=samples["patch_areas_m2"],
        )

        diffuse_rcs = np.asarray(patch_rcs.info.get("rcs_incoherent_m2", patch_rcs.rcs_m2), dtype=float)
        diffuse_rcs = np.where(np.isfinite(diffuse_rcs) & (diffuse_rcs > 0.0), diffuse_rcs, np.nan)

        patch_gain_dbi = np.full(patch_count_i, np.nan, dtype=float)
        for patch_idx, patch_pos in enumerate(patch_positions):
            look_patch_a = compute_rx_look_direction_antenna_frame(
                rx_positions_eci[:, i],
                rx_velocities_eci[:, i],
                patch_pos,
                antenna,
                q_sat=q_sat,
            )
            patch_gain_dbi[patch_idx] = antenna.get_gain_dbi_for_direction(
                look_patch_a,
                frequency_mhz,
                azimuth_deg=samples["steering_azimuth_deg"],
                elevation_deg=samples["steering_elevation_deg"],
            )

        patch_budget = compute_bistatic_link_budget(
            range_tx_m=patch_geometry.range_tx_m,
            range_rx_m=patch_geometry.range_rx_m,
            rx_gain_dbi=patch_gain_dbi,
            rcs_m2=diffuse_rcs,
            frequency_hz=frequency_hz,
            tx_eirp_dbw=tx_eirp_dbw,
            settings=settings,
        )
        diffuse_power_i = float(np.nansum(patch_budget["received_power_w"]))
        if not np.isfinite(diffuse_power_i) or diffuse_power_i <= 0.0:
            continue

        coherent_rcs_i = float(center_target_rcs.info.get("rcs_coherent_m2", np.full(num_positions, 0.0))[i])
        coherent_rcs_i = coherent_rcs_i if np.isfinite(coherent_rcs_i) and coherent_rcs_i > 0.0 else 0.0
        if coherent_rcs_i > 0.0:
            coherent_budget = compute_bistatic_link_budget(
                range_tx_m=np.array([center_geometry.range_tx_m[i]]),
                range_rx_m=np.array([center_geometry.range_rx_m[i]]),
                rx_gain_dbi=np.array([_center_rx_gain_dbi(rx_positions_eci[:, i], rx_velocities_eci[:, i], center_pos, antenna, frequency_mhz, q_sat)]),
                rcs_m2=np.array([coherent_rcs_i]),
                frequency_hz=frequency_hz,
                tx_eirp_dbw=tx_eirp_dbw,
                settings=settings,
            )
            coherent_power_i = float(coherent_budget["received_power_w"][0])
            if not np.isfinite(coherent_power_i):
                coherent_power_i = 0.0
        else:
            coherent_power_i = 0.0

        total_power_i = diffuse_power_i + coherent_power_i
        diffuse_equiv_rcs_i = _equivalent_rcs_from_received_power(
            received_power_w=diffuse_power_i,
            range_tx_m=float(center_geometry.range_tx_m[i]),
            range_rx_m=float(center_geometry.range_rx_m[i]),
            rx_gain_dbi=_center_rx_gain_dbi(rx_positions_eci[:, i], rx_velocities_eci[:, i], center_pos, antenna, frequency_mhz, q_sat),
            wavelength_m=wavelength_m,
            tx_eirp_dbw=tx_eirp_dbw,
            settings=settings,
        )
        total_equiv_rcs_i = diffuse_equiv_rcs_i + coherent_rcs_i
        if not np.isfinite(total_equiv_rcs_i) or total_equiv_rcs_i <= 0.0:
            continue

        equivalent_rcs_m2[i] = total_equiv_rcs_i
        patch_used[i] = True
        patch_count[i] = patch_count_i
        patch_footprint_area[i] = samples["footprint_area_m2"]
        patch_sampled_area[i] = samples["sampled_area_m2"]
        patch_diffuse_rcs_sum[i] = float(np.nansum(diffuse_rcs))
        patch_bragg_rcs_sum[i] = float(np.nansum(patch_rcs.info.get("rcs_bragg_m2", np.full(patch_count_i, np.nan))))
        patch_diffuse_power[i] = diffuse_power_i
        patch_coherent_power[i] = coherent_power_i
        patch_total_power[i] = total_power_i
        patch_equiv_diffuse_rcs[i] = diffuse_equiv_rcs_i
        patch_equiv_total_rcs[i] = total_equiv_rcs_i

    info["patch_integration_used"] = patch_used
    info["patch_count"] = patch_count
    info["patch_footprint_area_m2"] = patch_footprint_area
    info["patch_sampled_area_m2"] = patch_sampled_area
    info["patch_diffuse_rcs_sum_m2"] = patch_diffuse_rcs_sum
    info["patch_bragg_rcs_sum_m2"] = patch_bragg_rcs_sum
    info["patch_diffuse_received_power_w"] = patch_diffuse_power
    info["patch_coherent_received_power_w"] = patch_coherent_power
    info["patch_total_received_power_w"] = patch_total_power
    info["patch_equivalent_diffuse_rcs_m2"] = patch_equiv_diffuse_rcs
    info["patch_equivalent_total_rcs_m2"] = patch_equiv_total_rcs
    if "rcs_incoherent_m2" in info:
        info["rcs_incoherent_m2"] = np.where(patch_used, patch_equiv_diffuse_rcs, info["rcs_incoherent_m2"])
    if "rcs_bragg_m2" in info:
        info["rcs_bragg_m2"] = np.where(patch_used, patch_bragg_rcs_sum, info["rcs_bragg_m2"])
    if "effective_glistening_area_m2" in info:
        info["effective_glistening_area_m2"] = np.where(patch_used, patch_sampled_area, info["effective_glistening_area_m2"])
    if "illuminated_area_m2" in info:
        info["illuminated_area_m2"] = np.where(patch_used, patch_sampled_area, info["illuminated_area_m2"])

    return TargetRcs(rcs_m2=equivalent_rcs_m2, info=info)


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
    target: Target | None = None,
    surface_normals_eci: np.ndarray | None = None,
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
        surface_normals_eci:      [np.ndarray, shape (n, 3) | None]  Outward surface normals in ECI.
                                  Defaults to geocentric normals, which are exact only for a sphere.

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

    if surface_normals_eci is None:
        surface_normals_eci = reflection_positions_eci
    else:
        surface_normals_eci = np.asarray(surface_normals_eci, dtype=float)

    geometry = build_bistatic_geometry(
        tx_positions_eci,
        rx_positions_eci,
        reflection_positions_eci,
        surface_normals_eci,
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
            rx_gain_dbi[i] = antenna.get_gain_dbi_for_direction(look_dir_a, freq_mhz)

    wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz

    # Antenna-beam-limited illuminated area for distributed (area-extensive) returns. The receiver
    # auto-steers to the reflection, so rx_gain_dbi is the steered-beam peak directivity and the beam
    # solid angle 4 pi / G projects to a ground footprint (4 pi / G) * R_r^2 / cos(theta_inc). With no
    # antenna the area is left to the target's geometric fallback.
    if antenna is None:
        illuminated_area_m2 = None
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            g_rx_lin = np.power(10.0, rx_gain_dbi / 10.0)
            cos_inc = np.clip(np.cos(np.radians(incidence_deg)), 0.05, 1.0)
            illuminated_area_m2 = (4.0 * np.pi / g_rx_lin) * range_rx_m ** 2 / cos_inc

    target_rcs = target.compute_rcs(geometry, wavelength_m, illuminated_area_m2=illuminated_area_m2)

    if settings_obj.ocean_patch_integration_enabled and antenna is not None:
        from object_definitions.OceanTarget_def import OceanTarget

        if isinstance(target, OceanTarget):
            target_rcs = _compute_patch_integrated_ocean_rcs(
                target=target,
                center_geometry=geometry,
                center_target_rcs=target_rcs,
                surface_positions_eci=reflection_positions_eci,
                tx_positions_eci=tx_positions_eci,
                rx_positions_eci=rx_positions_eci,
                rx_velocities_eci=rx_velocities_eci,
                antenna=antenna,
                frequency_hz=frequency_hz,
                tx_eirp_dbw=tx_eirp_dbw,
                settings=settings_obj,
                q_sat=q_sat,
            )

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
    return np.asarray(n_ar_a / n_ar_a_norm, dtype=float)


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

    # Thermal noise (moving-target mode only). Area coverage supplies no noise settings; emit NaN so
    # the geometric path never depends on a noise floor it does not model.
    if settings_obj.noise_bandwidth_hz is not None and settings_obj.system_noise_temperature_k is not None:
        noise_density_w_hz = Constants.BOLTZMANN_CONSTANT * settings_obj.system_noise_temperature_k   # N0 = k T [W/Hz]
        noise_power_w = noise_density_w_hz * settings_obj.noise_bandwidth_hz                           # N = k T B [W]
        noise_density_dbw_hz = float(10.0 * np.log10(noise_density_w_hz))
        noise_power_dbw = float(10.0 * np.log10(noise_power_w))
        noise_bandwidth_hz_out = float(settings_obj.noise_bandwidth_hz)
        system_noise_temperature_k_out = float(settings_obj.system_noise_temperature_k)
    else:
        noise_density_w_hz = noise_power_w = noise_density_dbw_hz = noise_power_dbw = float("nan")
        noise_bandwidth_hz_out = system_noise_temperature_k_out = float("nan")
    noise_power_dbm = noise_power_dbw + 30.0
    cn0_db_hz = pr_dbw - noise_density_dbw_hz       # C/N0 = Pr / (k T), bandwidth-independent [dB-Hz]
    snr_rf_db = pr_dbw - noise_power_dbw            # SNR in noise_bandwidth_hz; pre-correlation, excludes despreading/integration gain

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
        "noise_bandwidth_hz": noise_bandwidth_hz_out,
        "system_noise_temperature_k": system_noise_temperature_k_out,
        "noise_density_w_hz": float(noise_density_w_hz),
        "noise_density_dbw_hz": noise_density_dbw_hz,
        "noise_power_w": float(noise_power_w),
        "noise_power_dbw": noise_power_dbw,
        "noise_power_dbm": noise_power_dbm,
        "cn0_db_hz": cn0_db_hz,
        "snr_rf_db": snr_rf_db,
    }
