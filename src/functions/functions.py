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
#TODO clean up this file

def compute_reflection(finalPositionGPS, finalPositionSat, ellipse, angle, times):
    # Preallocate outputs for efficiency
    theta_vec = np.full((len(finalPositionSat[0]), 3), np.nan)
    specular_point_vec = np.full((len(finalPositionSat[0]), 3), np.nan)
    # Calculate the angles (theta) between GPS and satellite positions
    dot_product = np.einsum('ij,ij->j', finalPositionGPS, finalPositionSat)
    magnitude_a = np.linalg.norm(finalPositionGPS, axis=0)
    magnitude_b = np.linalg.norm(finalPositionSat, axis=0)
    cos_theta   = dot_product / (magnitude_a * magnitude_b)
    theta       = np.degrees(np.arccos(cos_theta))
    # Loop through each satellite position to compute reflection points
    for i in range(finalPositionSat.shape[1]):
        has_los = check_los_satellites(finalPositionSat[:, i] / 1e3, finalPositionGPS[:, i] / 1e3, Constants.Radius_earth / 1e3)
        if theta[i] < angle and has_los:
            # Depending on 'ellipse' flag, call appropriate reflection calculation
            '''
            if ellipse == 1:
                specular_point_ecef, theta_degrees2, theta_degrees3 = find_specular_point_ellipse(finalPositionSat[:, i], finalPositionGPS[:, i])
                theta_vec[:, i] = [theta_degrees2, theta_degrees3, theta[i]]
                specular_point_vec[:, i] = specular_point_ecef
            '''
            if ellipse == 0:
                reflection_point, theta_2_circ, theta_3_circ = get_reflection_point_FF(finalPositionGPS[:, i], finalPositionSat[:, i])
                theta_vec[i, :] = [theta_2_circ, theta_3_circ, theta[i]]
                specular_point_vec[i, :] = reflection_point
    #print(theta_vec)
    # Convert ECEF specular points to latitude and longitude
        #print(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])
    #latlon_ref = np.full((len(finalPositionSat[0]), 2), np.nan)
    latlon_ref = np.full((finalPositionSat.shape[1], 2), np.nan)
    for i in range(finalPositionSat.shape[1]):
        if not np.isnan(specular_point_vec[i, 0]):

            lat, lon, alt = pymap3d.eci2geodetic(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2], times[i])
            #lat, lon, alt =  ecef_to_lla(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])
            latlon_ref[i, :] = [lat, lon]
            #latlon_ref[i, :] = ecef_to_lla(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])  # Ensure `ecef_to_lla` is defined or available
    return specular_point_vec, theta_vec, latlon_ref

def check_los_satellites(r1, r2, R_earth):
    """
    Checks if there is a clear line of sight (LOS) between two points r1 and r2,
    given the Earth's radius in kilometers.
    """
    # Vector between the two points
    d = r2 - r1

    # Parametric line between points, sampled at 100 points
    t_values = np.linspace(0, 1, 100)

    # Check if any point along the line passes through Earth
    for t in t_values:
        r_t = r1 + t * d  # Parametric equation of the line
        distance_to_earth_center = np.linalg.norm(r_t)

        if distance_to_earth_center < R_earth:
            return False  # No line of sight, obstructed by Earth
    return True  # LOS is clear

def get_reflection_point_FF(satPos, gpsPos):
    # Convert positions to column vectors
    satCoord = np.array(satPos).reshape(-1, 1)
    GPSCoord = np.array(gpsPos).reshape(-1, 1)

    # Radius of Earth in meters
    R_earth = 6378.15 * 1e3
    
    # Check for line of sight (LOS)
    d = satCoord - GPSCoord
    t_values = np.linspace(0, 1, 200)  # Sample points along the line

    # Verify LOS by checking if any point along the line passes through Earth
    for t in t_values:
        r_t = GPSCoord + t * d
        distance_to_earth_center = np.linalg.norm(r_t)
        if distance_to_earth_center < R_earth:
            # No line of sight, obstructed by Earth
            return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan

    # Get rotation matrix
    Rot = getRotationEcefToPlaneCoords(satCoord, GPSCoord)

    # Rotate satellite and GPS coordinates
    satCoord1 = Rot @ satCoord
    GPSCoord1 = Rot @ GPSCoord

    # Find 2D reflection point
    N, A, B = get_reflection_point2D(satCoord1.flatten(), GPSCoord1.flatten())

    # Transform reflection point to 3D (z = 0)
    N = np.array([N[0], N[1], 0])

    # Convert reflection point back to ECEF coordinates
    ReflectionPoint = Rot.T @ N

    # Calculate angles between vectors
    vec_1 = ReflectionPoint - np.array([0, 0, 0])
    vec_2 = satPos - ReflectionPoint
    vec_3 = gpsPos - ReflectionPoint

    theta_degrees2 = angle_between_vectors(vec_1, vec_2)
    theta_degrees3 = angle_between_vectors(vec_1, vec_3)

    return ReflectionPoint, theta_degrees2, theta_degrees3

def angle_between_vectors(v1, v2):
    # Calculate the angle between two vectors in degrees
    dot_product   = np.dot(v1, v2)
    magnitude_a   = np.linalg.norm(v1)
    magnitude_b   = np.linalg.norm(v2)
    cos_theta     = dot_product / (magnitude_a * magnitude_b)
    theta_radians = np.arccos(cos_theta)
    return np.degrees(theta_radians)

def skew_matrix(v):
    # Returns the skew symmetric matrix of the vector v
    return np.array([[0,    -v[2], v[1]],
                     [v[2],  0,   -v[0]],
                     [-v[1], v[0], 0]])

def getRotationEcefToPlaneCoords(satCoord, GPSCoord):
    # Calculate the rotation matrix to transform from ECEF to plane coordinates
    plane = np.cross(satCoord.flatten(), GPSCoord.flatten())
    plane = plane / np.linalg.norm(plane)
    z     = np.array([0, 0, 1])

    vec = np.cross(plane, z)
    s   = np.linalg.norm(vec)
    c   = np.dot(plane, z)

    vec2  = vec / np.linalg.norm(vec)
    theta = np.arccos(np.dot(plane, z) / np.linalg.norm(plane))
    c2    = np.cos(theta)
    s2    = np.sin(theta)

    Sv  = skew_matrix(vec2)
    Rot = np.eye(3) + s2 * Sv + Sv @ Sv * (1 - c2)
    return Rot

def get_reflection_point2D(satCoord1, GPSCoord1):
    # Calculates the reflection point in 2D
    Re = 6371000  # Approximate Earth radius in meters

    A = satCoord1[:2] / Re
    B = GPSCoord1[:2] / Re

    u = np.dot(A, A)
    v = np.dot(A, B)
    w = np.dot(B, B)

    r0 = w - 1
    r1 = 2 * (w - v)
    r2 = u + 2 * v + w - 4 * u * w
    r3 = 4 * (v**2 - u * w)
    r4 = 4 * u * (u * w - v**2)

    coefvct = [r4, r3, r2, r1, r0]  # Coefficients for polynomial
    roots = np.roots(coefvct)  # Find roots of the polynomial
    roots = roots.real  # Assuming real roots for simplicity

    for x in roots:
        if x > 0:
            y = (1 + x - 2 * u * x**2) / (1 + 2 * v * x)
            if y > 0:
                N = x * A + y * B
                return Re * N, A, B
    return None, None, None

## NOTE: CLEANED UP AFTER THIS POINT
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
            tle_file=tle_file_dict['tle_file'], 
            satConfigName=name
        )
    return gnss_constl_dict

def propagate_all_orbits(sat_const, gnss_constl_dict, start_time, duration_h, step_size): # TODO move to functions.py
    '''
    Propagate orbits for satellite constellation and GNSS constellations.
    '''
    duration = int(duration_h * Constants.Sec_per_hour)
    # Propagate the satellite constellation (GNSS-R / LEO)
    print(f"Propagating satellite constellation: \"{sat_const.satConfigName}\"")
    sat_const.propagate_constellation(start_time, duration, step_size)

    # Propagate all GNSS constellations
    for name, gnss_const in gnss_constl_dict.items():
        print(f"Propagating GNSS constellation: \"{name}\"")
        gnss_const.propagate_constellation(start_time, duration, step_size)

def color_gnss_const(gnss_const): # TODO move to functions_def.py
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