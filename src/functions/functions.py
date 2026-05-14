from collections import defaultdict
import datetime as dt
import warnings

import numpy as np
import pymap3d
from shapely.geometry import Point, Polygon
from skyfield.api import load, wgs84

from functions.rotations import eci_to_antenna_rotation_from_state
from object_definitions.Constants_def import Constants
from object_definitions.SatConstellation_def import SatelliteConstellation

"""
def ecef_to_lla(x, y, z):
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    # Transform ECEF (x, y, z) to LLA (lat, lon, alt)
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt

# Example usage:
x, y, z = 1113194.907, 1113194.907, 0  # example ECEF coordinates in meters
vec = [ 5179877.56825321, -3709226.89116256,   -12051.87884988]
x, y, z = vec[0], vec[1], vec[2]

lat, lon, alt = ecef_to_lla(x, y, z)
print(f"Latitude: {lat}, Longitude: {lon}, Altitude: {alt}")
"""


def ensure_utc_aware(dt_obj):
    """
    =========================================================================================================
    Convert datetime to UTC-aware if it's naive.

    INPUTS:
        dt_obj: [datetime]   Datetime object (can be timezone-aware or naive)

    OUTPUT:
        result: [datetime]   Datetime guaranteed to be UTC-aware
    =========================================================================================================
    """
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=dt.UTC)
    return dt_obj


def to_utc_datetime_array(times):
    """
    =========================================================================================================
    Coerce datetime-like input to an object-dtype ndarray of UTC-aware Python datetimes.

    pymap3d (>=3.2) requires its time argument to be Python ``datetime`` instances, not
    ``numpy.datetime64``, so we keep the array as ``dtype=object``. Timezone-aware inputs are
    converted to UTC; naive inputs are assumed to already represent UTC and are tagged accordingly.
    Preserving tzinfo matters because downstream consumers (e.g. MovingTargetDetection's
    reflection_dict_timeSorted) key by datetime, and mixing naive/aware datetimes will never match
    in a dict lookup.

    INPUTS:
        times: [datetime | list[datetime] | np.ndarray]   Input times in any datetime-like form

    OUTPUT:
        result: [np.ndarray, dtype=object]   Array of UTC-aware Python datetimes
    =========================================================================================================
    """
    if isinstance(times, np.ndarray) and np.issubdtype(times.dtype, np.datetime64):
        times = times.astype("datetime64[us]").tolist()

    if isinstance(times, dt.datetime):
        times = [times]

    normalized_times = []
    for time_value in times:
        if isinstance(time_value, dt.datetime):
            if time_value.tzinfo is None:
                normalized_times.append(time_value.replace(tzinfo=dt.UTC))
            else:
                normalized_times.append(time_value.astimezone(dt.UTC))
        else:
            normalized_times.append(time_value)
    return np.array(normalized_times, dtype=object)


def angle_between_vectors(v1, v2, unit="degrees"):
    """
    =========================================================================================================
    Calculate the angle between two vectors in degrees

    INPUTS:
        v1:         [np.array]               First vector
        v2:         [np.array]               Second vector

    OUTPUT:
        angle between vectors: [np.float]    Angle between the vectors in degrees
    =========================================================================================================
    """
    dot_product = np.dot(v1, v2)
    magnitude_a = np.linalg.norm(v1)
    magnitude_b = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    theta_radians = np.arccos(cos_theta)
    if unit == "radians":
        return theta_radians
    elif unit == "degrees":
        return np.degrees(theta_radians)
    else:
        raise ValueError("Unit must be either 'radians' or 'degrees'.")


def skew_matrix(v):
    """
    =========================================================================================================
    Returns the skew symmetric matrix of the vector v

    INPUTS:
        v:           [np.array]            Vector for which the skew symmetric matrix is calculated

    OUTPUT:
        skew_matrix: [np.array]            Skew symmetric matrix of the vector v
    =========================================================================================================
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def load_gnss_constellation_dict(gnss_costl_configs: dict) -> dict:
    """
    =========================================================================================================
    Load GNSS constellation configurations.

    INPUTS:
        gnss_costl_configs: [dict]   GNSS constellation configurations read from the configuration file

    OUTPUT:
        gnss_constl_dict: [dict]   {name: SatelliteConstellation} for each GNSS constellation
    =========================================================================================================
    """
    gnss_constl_dict = {}
    for name, tle_file_dict in gnss_costl_configs.items():
        print(f'GNSS constellation: "{name}" loaded from: {tle_file_dict["tle_file"]}.')
        gnss_constl_dict[name] = SatelliteConstellation(tle_file=tle_file_dict["tle_file"], satConfigName=name)
    return gnss_constl_dict


def propagate_all_orbits(sat_const, gnss_constl_dict: dict) -> None:
    """
    =========================================================================================================
    Propagate orbits for satellite constellation and GNSS constellations.

    INPUTS:
        sat_const:       [SatelliteConstellation]  Satellite constellation object to propagate
        gnss_constl_dict: [dict]                    Dictionary of GNSS constellation objects
        step_size:       [float]                   Step size for propagation in seconds TODO replace with stepsize from sat_const

    OUTPUT:
        None:                                       Propagation is done in-place, no return value.
    =========================================================================================================
    """
    # Robustness checks
    if not isinstance(sat_const, SatelliteConstellation):
        raise ValueError("sat_const must be an instance of SatelliteConstellation class.")
    if not isinstance(gnss_constl_dict, dict):
        raise ValueError("gnss_constl_dict must be a dictionary of GNSS constellations.")
    if not all(isinstance(gnss_const, SatelliteConstellation) for gnss_const in gnss_constl_dict.values()):
        raise ValueError("All values in gnss_constl_dict must be instances of SatelliteConstellation class.")

    # Get the start time, duration and step size from the GNSS-R / LEO satellite constellation
    start_time_skf = sat_const.times_skyfield[0] if sat_const.times_skyfield is not None else None
    if start_time_skf is None:
        raise ValueError("Start time for the satellite constellation is not set.")
    start_time = dt.datetime(
        int(start_time_skf.utc.year),
        int(start_time_skf.utc.month),
        int(start_time_skf.utc.day),
        int(start_time_skf.utc.hour),
        int(start_time_skf.utc.minute),
        int(start_time_skf.utc.second),
    )
    start_time = ensure_utc_aware(start_time)  # Ensure the start time is UTC-aware
    duration = sat_const.duration_sec
    step_size = sat_const.propagation_step_size

    # Propagate the satellite constellation (GNSS-R / LEO)
    print(f'Propagating satellite constellation: "{sat_const.satConfigName}"')
    sat_const.propagate_constellation(start_time, duration, step_size)

    # Propagate all GNSS constellations
    for name, gnss_const in gnss_constl_dict.items():
        print(f'Propagating GNSS constellation: "{name}"')
        gnss_const.propagate_constellation(start_time, duration, step_size)
        if gnss_const.times_skyfield is None:
            print(f'WARNING: GNSS constellation "{name}" has not been propagated.')


def color_gnss_const(gnss_const: str) -> str:
    """
    =========================================================================================================
    Assign a color to the GNSS constellation based on its name.

    INPUTS:
        gnss_const: [str]   Name of the GNSS constellation (e.g., 'GPS', 'GLONASS', etc.)

    OUTPUT:
        color: [str | None]   Matplotlib color name for the constellation, or None if unknown
    =========================================================================================================
    """
    if gnss_const == "GPS":
        color = "red"
    elif gnss_const == "GLONASS":
        color = "orange"
    elif gnss_const == "GALILEO":
        color = "blue"
    elif gnss_const == "BEIDOU":
        color = "green"
    elif gnss_const == "QZSS":
        color = "purple"
    elif gnss_const == "IRNSS":
        color = "pink"
    else:
        color = None
    return color


def normalize_gnss_constellation_arg(arg, available_names) -> list:
    """Coerce a constellation filter (str/list/tuple/None) to list[str]; None → all available."""
    if arg is None:
        return list(available_names)
    if isinstance(arg, str):
        return [item.strip() for item in arg.strip("[]").split(",")]
    if isinstance(arg, (list, tuple)):
        return [str(item).strip() for item in arg]
    raise TypeError(f"gnss_constellation must be str, list, tuple, or None; got {type(arg).__name__}")


def delta_lon_from_distance(lat: float, x_dist: float) -> float:
    """
    =========================================================================================================
    Calculate the change in longitude corresponding to a distance in meters.
    This function assumes a elliptical Earth model.

    INPUTS:
      lat:         [float]                 Latitude in degrees
      x_distance : [float]                 Distance in meters in the east-west direction

    OUTPUT:
      np.float:     [degrees]               delta_lon
    =========================================================================================================
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)

    # Calculate the radius of the Earth at the given latitude
    e2 = 2 * (Constants.EARTH_FLATTENING) - (Constants.EARTH_FLATTENING) ** 2  # Earth eccentricity squared
    a = Constants.RADIUS_EARTH_EQUAT  # Semi-major axis of the Earth in meters

    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Calculate changes in lon
    delta_lon_rad = x_dist / (N * np.cos(lat_rad))

    # Convert to degrees
    delta_lon = np.degrees(delta_lon_rad)

    return delta_lon


def delta_lat_from_distance(lat: float, y_dist: float) -> float:
    """
    =========================================================================================================
    Calculate the change in latitude corresponding to a distance in meters.
    This function assumes a elliptical Earth model.

    INPUTS:
      lat:         [float]                 Latitude in degrees
      y_distance : [float]                 Distance in meters in the north-south direction

    OUTPUT:
      np.float:     [degrees]               delta_lat
    =========================================================================================================
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)

    # Calculate the Earth's eccentricity squared
    e2 = 2 * Constants.EARTH_FLATTENING - Constants.EARTH_FLATTENING**2

    # Semi-major axis of the Earth (equatorial radius)
    a = Constants.RADIUS_EARTH_EQUAT  # in meters

    # Calculate the meridional radius of curvature
    # This is the radius of curvature along a north-south line
    M = a * (1 - e2) / np.power(1 - e2 * np.sin(lat_rad) ** 2, 1.5)

    # Calculate changes in latitude (radians)
    delta_lat_rad = y_dist / M

    # Convert to degrees
    delta_lat = np.degrees(delta_lat_rad)

    return delta_lat


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float = 0) -> tuple:
    """
    =========================================================================================================
    Convert geodetic coordinates (latitude, longitude, height) to ECEF (Earth-Centered, Earth-Fixed)
    coordinates.

    INPUTS:
        lat_deg: [float]   Latitude in degrees
        lon_deg: [float]   Longitude in degrees
        h:       [float]   Height above the ellipsoid in meters (default is 0)

    OUTPUT:
        result: [tuple]   (X, Y, Z) ECEF coordinates in meters
    =========================================================================================================
    """
    # Convert latitude and longitude from degrees to radians
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    e2 = Constants.EARTH_FLATTENING * (2 - Constants.EARTH_FLATTENING)  # Earth eccentricity squared
    N = Constants.RADIUS_EARTH_EQUAT / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = ((1 - e2) * N + h) * np.sin(lat)

    return X, Y, Z


def rotation_matrix_from_vectors(v1, v2):
    """
    =========================================================================================================
    Calculate the rotation matrix that rotates vector v1 to vector v2.

    INPUTS:
        v1: [np.array]   First vector
        v2: [np.array]   Second vector

    OUTPUT:
        R: [np.array, shape (3, 3)]   Rotation matrix that rotates v1 to v2
    =========================================================================================================
    """
    def rodrigues_rotation_matrix(axis, angle):
        """
        =========================================================================================================
        Create rotation matrix using Rodrigues' formula.

        INPUTS:
            axis:  [np.ndarray, shape (3,)]   Unit rotation axis
            angle: [float]                    Rotation angle [rad]

        OUTPUT:
            R: [np.ndarray, shape (3, 3)]   Rotation matrix
        =========================================================================================================
        """
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Check if vectors are already aligned
    dot_product = np.dot(v1, v2)
    if np.allclose(dot_product, 1.0):
        return np.eye(3)  # No rotation needed

    # If vectors are opposite, we need a 180° rotation
    if np.allclose(dot_product, -1.0):
        # Find any perpendicular vector
        perp = np.array([1, 0, 0])
        if abs(np.dot(v1, perp)) > 0.9:
            perp = np.array([0, 1, 0])

        # Create perpendicular vector using cross product
        axis = np.cross(v1, perp)
        axis = axis / np.linalg.norm(axis)

        # 180° rotation around this axis
        return rodrigues_rotation_matrix(axis, np.pi)

    # General case: use cross product for rotation axis
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return rodrigues_rotation_matrix(axis, angle)




def compute_visible_times_and_resample_ship(moving_target_detection, sat_const, elev_min=40):
    """
    =========================================================================================================
    Compute times when at least one satellite is visible from the moving ship and resample the ship
    trajectory at exactly those timestamps.

    INPUTS:
        moving_target_detection: [MovingTargetDetection]   Object containing the ship trajectory
        sat_const:               [SatelliteConstellation]  Satellite constellation (Skyfield objects)
        elev_min:                [float]                   Minimum elevation angle for visibility [deg]

    OUTPUT:
        result: [tuple]   (resampled_ship_positions, sat_visible_times)
                          - resampled_ship_positions: list of (lat, lon) at the visible timestamps
                          - sat_visible_times:         dict {satellite_name: list of visible datetimes}
    =========================================================================================================
    """
    ts = load.timescale()

    ship_times = moving_target_detection.moving_target_object.timestamps
    ship_positions = [(float(lat), float(lon)) for lat, lon in moving_target_detection.moving_target_object.trajectory_points]

    sat_times_dt = [t.utc_datetime() for t in sat_const.times_skyfield]

    # Convert to seconds for interpolation
    ship_times_sec = np.array([(t - ship_times[0]).total_seconds() for t in ship_times])
    sat_times_sec = np.array([(t - ship_times[0]).total_seconds() for t in sat_times_dt])
    lats = np.array([pos[0] for pos in ship_positions])
    lons = np.array([pos[1] for pos in ship_positions])

    # Interpolate ship positions
    interp_lats = np.interp(sat_times_sec, ship_times_sec, lats)
    interp_lons = np.interp(sat_times_sec, ship_times_sec, lons)
    ship_positions_all = list(zip(interp_lats, interp_lons, strict=False))

    # Compute visibility
    sat_visible_times = defaultdict(list)
    for pos, t in zip(ship_positions_all, sat_times_dt, strict=False):
        lat, lon = pos
        observer = wgs84.latlon(lat, lon)
        for sat in sat_const.satellites:
            topocentric = (sat.skf - observer).at(ts.from_datetime(t))
            alt, az, distance = topocentric.altaz()
            if alt.degrees >= elev_min:
                sat_visible_times[sat.skf.name].append(t)

    # Collect all timestamps where at least one satellite is visible
    all_visible_times = sorted({t for times in sat_visible_times.values() for t in times})
    sat_time_to_idx = {t: i for i, t in enumerate(sat_times_dt)}
    indices = [sat_time_to_idx[t] for t in all_visible_times]
    resampled_ship_positions = [ship_positions_all[i] for i in indices]

    # Save resampled ship positions in the object
    moving_target_detection.ship_positions_resampled = [
        Point(lon, lat)  # Note: Shapely expects Point(x, y) = Point(lon, lat)
        for lat, lon in resampled_ship_positions
    ]
    moving_target_detection.timestamps_resampled = all_visible_times

    return resampled_ship_positions, sat_visible_times


def propagate_only_visible_satellites(sat_const, sat_visible_times, gnss_constl_dict=None):
    """
    =========================================================================================================
    Propagate satellites only at timestamps where they were visible from the moving ship
    (or GNSS satellites at all visible times if provided). Much faster and memory-efficient for
    sparse timestamp sets.

    INPUTS:
        sat_const:         [SatelliteConstellation]    Main satellite constellation (LEO or others)
        sat_visible_times: [dict]                      {satellite_name : list of datetimes}
        gnss_constl_dict:  [dict | None]               Optional {name : GNSS SatelliteConstellation}

    OUTPUT:
        visible_sat_objs: [list[Satellite]]   Satellites propagated only at relevant (visible) timestamps
    =========================================================================================================
    """
    ts = load.timescale()

    all_visible_times = sorted({t for times in sat_visible_times.values() for t in times})

    visible_times_skyfield = ts.from_datetimes(all_visible_times)
    visible_sat_objs = []

    # --- LEO satellites ---
    for sat in sat_const.satellites:
        if sat.skf.name not in sat_visible_times:
            continue

        times_for_sat = sat_visible_times[sat.skf.name]
        if not times_for_sat:
            continue

        # Propagate only at visible timestamps
        sat.propagate_orbit_for_times(ts.from_datetimes(times_for_sat))
        sat_times_dt = [t.utc_datetime() for t in sat.time_prop_vec]

        # Synchronize to all_visible_times, filling missing with NaN
        sat_positions_sync = []
        sat_velocities_sync = []
        sat_time_to_idx = {t: i for i, t in enumerate(sat_times_dt)}

        for t in all_visible_times:
            idx = sat_time_to_idx.get(t)
            if idx is not None:
                sat_positions_sync.append(sat.positions_eci[:, idx])
                sat_velocities_sync.append(sat.velocities_eci[:, idx])
            else:
                sat_positions_sync.append(np.full(3, np.nan))
                sat_velocities_sync.append(np.full(3, np.nan))

        sat.positions_eci = np.array(sat_positions_sync).T
        sat.velocities_eci = np.array(sat_velocities_sync).T
        sat.time_prop_vec = visible_times_skyfield
        visible_sat_objs.append(sat)

    # --- GNSS satellites propagated at all visible times ---
    if gnss_constl_dict is not None:
        for gnss_const in gnss_constl_dict.values():
            for sat in gnss_const.satellites:
                sat.propagate_orbit_for_times(visible_times_skyfield)
                sat.time_prop_vec = visible_times_skyfield
                visible_sat_objs.append(sat)

    return visible_sat_objs


def compute_footprint(sat_pos, sat_vel, refl_pos, time_i, antenna, frequency, q_sat=None, earth_model="spherical"):
    """
    =========================================================================================================
    Compute the antenna footprint on the Earth's surface as a Shapely polygon in lon/lat coordinates,
    given the satellite state and the reflection point at one time step.

    Extracted from Reflection._compute_footprint / BRP._compute_footprint so both classes can share
    one implementation. Behaviour is identical to those methods.

    INPUTS:
        sat_pos:     [np.array, shape (3,)]   Satellite position vector in ECI [m]
        sat_vel:     [np.array, shape (3,)]   Satellite velocity vector in ECI [m/s]
        refl_pos:    [np.array, shape (3,)]   Reflection point in ECI [m]
        time_i:      [Skyfield Time]          Time at this step
        antenna:     [RxAntenna]              Receiver antenna object
        frequency:   [float]                  Carrier frequency [MHz]
        q_sat:       [np.array, shape (4,) | None]   Satellite attitude quaternion [w, x, y, z]
                                                     (default: identity [1, 0, 0, 0])
        earth_model: [str]                    "spherical" or "elliptical"

    OUTPUT:
        result: [tuple]   (polygon_footprint, center_point)
                          - polygon_footprint: Shapely Polygon of the footprint (lon, lat)
                          - center_point:      np.array([lat, lon]) of the reflection point
    =========================================================================================================
    """
    # Check if the satellite position is valid
    if np.any(np.isnan(sat_pos)) or np.any(np.isnan(refl_pos)):
        print("Invalid satellite or reflection position. Skipping footprint calculation.")
        return Polygon(), np.array([])

    l_sr = np.linalg.norm(refl_pos - sat_pos)  # Beam length (slant range)
    n_ar_eci = (refl_pos - sat_pos) / l_sr  # Satellite-to-reflection unit vector in ECI coordinates

    # ========== ECI → ANTENNA ROTATION (via VNC; degeneracy returns None) ==========
    R_eci_to_antenna = eci_to_antenna_rotation_from_state(sat_pos, sat_vel, antenna, q_sat)
    if R_eci_to_antenna is None:
        warnings.warn("Invalid satellite state for VNC frame construction")
        return Polygon(), np.array([])

    # ========== BEAM / RAY COORDINATE SYSTEM {r} ==========
    # Create perpendicular axes for the beam cone (orthogonal to the beam direction)
    v_s_proj_i = sat_vel - np.dot(sat_vel, n_ar_eci) * n_ar_eci
    # Handle edge case where velocity is parallel to the beam direction (should never happen in practice)
    if np.linalg.norm(v_s_proj_i) < 1e-6:
        if abs(n_ar_eci[2]) < 0.9:
            temp_vec = np.array([0, 0, 1])
        else:
            temp_vec = np.array([1, 0, 0])
        v_s_proj_i = temp_vec - np.dot(temp_vec, n_ar_eci) * n_ar_eci

    x_r_eci = v_s_proj_i / np.linalg.norm(v_s_proj_i)
    y_r_eci = np.cross(n_ar_eci, x_r_eci)
    y_r_eci = y_r_eci / np.linalg.norm(y_r_eci)
    z_r_eci = n_ar_eci
    # (kept as local variables for parity with the pre-refactor bodies; not currently used downstream)
    _R_eci_to_reflection = np.vstack([x_r_eci, y_r_eci, z_r_eci])

    # ========== REFLECTION DIRECTION IN ANTENNA FRAME ==========
    n_ar_a = R_eci_to_antenna @ n_ar_eci
    if n_ar_a[2] < 0:
        # Normal geometry (reflection in antenna back hemisphere); empty polygon propagates downstream
        # and is counted via MovingTargetDetection.footprint_stats.
        return Polygon(), np.array([])

    # ========== ROTATE NADIR-POINTING FOOTPRINT TO REFLECTION-BEAM POINTING ==========
    beam_dir_a = antenna._get_antenna_footprint(n_ar_a, frequency)

    # ========== VECTORIZED BEAM DIRECTIONS ==========
    beam_dirs_i = R_eci_to_antenna.T @ beam_dir_a.T
    no_footprint_points = beam_dirs_i.shape[1]

    # ========== VECTORIZED INTERSECTION ==========
    if earth_model == "spherical":
        earth_radius = Constants.R_EARTH_MEAN
    else:
        # Use radius at reflection point for all (faster than per-point calculation)
        latitude_rad = np.arcsin(np.clip(refl_pos[2] / np.linalg.norm(refl_pos), -1, 1))
        a = Constants.RADIUS_EARTH_EQUAT
        f = Constants.EARTH_FLATTENING
        b = a * (1 - f)
        # Geocentric radius: R(φ) = √(a²cos²φ + b²sin²φ)
        earth_radius = np.sqrt((a**2 * np.cos(latitude_rad) ** 2) + (b**2 * np.sin(latitude_rad) ** 2))

    a = 1.0
    b = 2.0 * np.dot(sat_pos, beam_dirs_i)
    c = np.dot(sat_pos, sat_pos) - earth_radius**2

    discriminant = b**2 - 4 * a * c
    valid_discriminant = discriminant >= 0

    t_vals = np.full(no_footprint_points, np.nan)
    t_vals[valid_discriminant] = (-b[valid_discriminant] - np.sqrt(discriminant[valid_discriminant])) / 2.0

    positive_t_mask = t_vals > 0
    valid_mask = valid_discriminant & positive_t_mask

    intersections_eci = sat_pos[np.newaxis, :] + t_vals[:, np.newaxis] * beam_dirs_i.T

    # ========== VECTORIZED LAT/LON CONVERSION ==========
    times = np.array([time_i] * no_footprint_points)
    lat_lon_points = np.full((no_footprint_points, 2), np.nan)
    if valid_mask.size > 0:
        try:
            lat, lon, _alt = pymap3d.eci2geodetic(
                intersections_eci[valid_mask, 0],
                intersections_eci[valid_mask, 1],
                intersections_eci[valid_mask, 2],
                times[valid_mask],
            )
            lat_lon_points[valid_mask] = np.column_stack((lat, lon))
        except Exception as e:
            warnings.warn(f"Failed to convert some points to geodetic coordinates: {e}")
            lat_lon_points[valid_mask] = np.nan

    # ========== OUTPUTS ==========
    ellipse_points = np.column_stack((lat_lon_points[:, 0], lat_lon_points[:, 1]))
    lat_center, lon_center, _ = pymap3d.eci2geodetic(refl_pos[0], refl_pos[1], refl_pos[2], time_i)
    center_point = np.array([lat_center, lon_center])

    footprint_points = ellipse_points[:, [1, 0]]
    if np.isnan(footprint_points).any():
        # Footprint partly tangenting earth — drop NaN rows
        # TODO: handle this more gracefully than dropping points
        footprint_points = footprint_points[~np.isnan(footprint_points).any(axis=1)]

    polygon_footprint = Polygon(footprint_points)

    return polygon_footprint, center_point


