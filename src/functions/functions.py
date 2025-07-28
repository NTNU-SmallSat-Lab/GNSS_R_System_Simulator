import datetime as dt

import numpy as np

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

    INPUT:
        dt_obj: datetime object (can be timezone-aware or naive)

    OUTPUT:
        datetime object that is guaranteed to be UTC-aware
    =========================================================================================================
    """
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=dt.UTC)
    return dt_obj


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

    INPUT:
        gnss_costl_configs: Dictionary containing GNSS constellation configurations
                            read from the configuration file.

    OUTPUT:
        gnss_constl_dict: Dictionary containing GNSS constellation objects.
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
        gnss_const: Name of the GNSS constellation (e.g., 'GPS', 'GLONASS', etc.).
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
    Convert geodetic coordinates (latitude, longitude, height) to ECEF (Earth-Centered, Earth-Fixed) coordinates.
    INPUTS:
        lat_deg: [float] Latitude in degrees
        lon_deg: [float] Longitude in degrees
        h:       [float] Height above the ellipsoid in meters (default is 0)
    OUTPUTS:
        X:       [float] ECEF X coordinate in meters
        Y:       [float] ECEF Y coordinate in meters
        Z:       [float] ECEF Z coordinate in meters
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


def quat_to_rotation_matrix(q: list) -> np.array:
    """
    =========================================================================================================
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    INPUTS:
        q: [list or np.array] Quaternion in the form [w, x, y, z]
    OUTPUTS:
        R: [np.array] 3x3 Rotation matrix corresponding to the quaternion
    =========================================================================================================
    """
    w, x, y, z = q
    # Normalize quaternion
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )

def rotation_matrix_from_vectors(v1, v2):
    """
    =========================================================================================================
    Calculate the rotation matrix that rotates vector v1 to vector v2.

    INPUTS:
        v1: [np.array] First vector
        v2: [np.array] Second vector

    OUTPUTS:
        R: [np.array] 3x3 Rotation matrix that rotates v1 to v2
    =========================================================================================================
    """
    def rodrigues_rotation_matrix(axis, angle):
        """
        Create rotation matrix using Rodrigues' formula.
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
