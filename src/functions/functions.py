import numpy    as np
import pymap3d

from pyproj                                  import Transformer
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Constants_def        import Constants


'''
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
'''
def angle_between_vectors(v1, v2):
    '''
    Calculate the angle between two vectors in degrees

    INPUTS:
        v1:         [np.array]               First vector
        v2:         [np.array]               Second vector

    OUTPUT:
        angle between vectors: [np.float]    Angle between the vectors in degrees
    '''
    dot_product   = np.dot(v1, v2)
    magnitude_a   = np.linalg.norm(v1)
    magnitude_b   = np.linalg.norm(v2)
    cos_theta     = dot_product / (magnitude_a * magnitude_b)
    theta_radians = np.arccos(cos_theta)
    return np.degrees(theta_radians)

def skew_matrix(v):
    '''
    Returns the skew symmetric matrix of the vector v
    '''
    return np.array([[0,    -v[2], v[1]],
                     [v[2],  0,   -v[0]],
                     [-v[1], v[0], 0]])

def load_gnss_constellation_dict(gnss_costl_configs):
    '''
    Load GNSS constellation configurations.

    INPUT:
        gnss_costl_configs: Dictionary containing GNSS constellation configurations
                            read from the configuration file.
    OUTPUT:
        gnss_constl_dict: Dictionary containing GNSS constellation objects.
    '''
    gnss_constl_dict = {}
    for name, tle_file_dict in gnss_costl_configs.items():
        print(f"GNSS constellation: \"{name}\" loaded from: {tle_file_dict['tle_file']}.")
        gnss_constl_dict[name] = SatelliteConstellation(
            tle_file      = tle_file_dict['tle_file'], 
            satConfigName = name
        )
    return gnss_constl_dict

def propagate_all_orbits(sat_const, gnss_constl_dict, start_time, duration_h, step_size):
    '''
    Propagate orbits for satellite constellation and GNSS constellations.
    '''
    # TODO extract start time from the satellite constellation

    duration = int(duration_h * Constants.SEC_PER_HOUR)
    # Propagate the satellite constellation (GNSS-R / LEO)
    print(f"Propagating satellite constellation: \"{sat_const.satConfigName}\"")
    sat_const.propagate_constellation(start_time, duration, step_size)

    # Propagate all GNSS constellations
    for name, gnss_const in gnss_constl_dict.items():
        print(f"Propagating GNSS constellation: \"{name}\"")
        gnss_const.propagate_constellation(start_time, duration, step_size)
        if gnss_const.times_skyfield is None:
            print(f"WARNING: GNSS constellation \"{name}\" has not been propagated.")

def color_gnss_const(gnss_const):
    '''
    Assign a color to the GNSS constellation based on its name.

    INPUTS:
        gnss_const: Name of the GNSS constellation (e.g., 'GPS', 'GLONASS', etc.).
    '''
    if gnss_const == 'GPS':
            color = 'red'
    elif gnss_const == 'GLONASS':
        color = 'orange'
    elif gnss_const == 'GALILEO':
        color = 'blue'
    elif gnss_const == 'BEIDOU':
        color = 'green'
    elif gnss_const == 'QZSS':
        color = 'purple'
    elif gnss_const == 'IRNSS':
        color = 'pink'
    else:
        color = None
    return color

def delta_lon_from_distance(lat, x_dist) -> float:
    """
    Calculate the change in longitude corresponding to a distance in meters.
    This function assumes a elliptical Earth model.

    INPUTS:
      lat:         [float]                 Latitude in degrees
      x_distance : [float]                 Distance in meters in the east-west direction

    OUTPUT:
      np.float:     [degrees]               delta_lon
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)

    # Calculate the radius of the Earth at the given latitude
    e2 = 2*(Constants.EARTH_FLATTENING) - (Constants.EARTH_FLATTENING)**2  # Earth eccentricity squared
    a  = Constants.RADIUS_EARTH_EQUAT                                      # Semi-major axis of the Earth in meters

    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Calculate changes in lon
    delta_lon_rad = x_dist / (N * np.cos(lat_rad))

    # Convert to degrees
    delta_lon = np.degrees(delta_lon_rad)

    return delta_lon

def delta_lat_from_distance(lat, y_dist) -> float:
    """
    Calculate the change in latitude corresponding to a distance in meters.
    This function assumes a elliptical Earth model.

    INPUTS:
      lat:         [float]                 Latitude in degrees
      y_distance : [float]                 Distance in meters in the north-south direction

    OUTPUT:
      np.float:     [degrees]               delta_lat
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)
    
    # Calculate the Earth's eccentricity squared
    e2 = 2*Constants.EARTH_FLATTENING - Constants.EARTH_FLATTENING**2
    
    # Semi-major axis of the Earth (equatorial radius)
    a = Constants.RADIUS_EARTH_EQUAT  # in meters
    
    # Calculate the meridional radius of curvature
    # This is the radius of curvature along a north-south line
    M = a * (1 - e2) / np.power(1 - e2 * np.sin(lat_rad)**2, 1.5)

    # Calculate changes in latitude (radians)
    delta_lat_rad = y_dist / M

    # Convert to degrees
    delta_lat = np.degrees(delta_lat_rad)

    return delta_lat