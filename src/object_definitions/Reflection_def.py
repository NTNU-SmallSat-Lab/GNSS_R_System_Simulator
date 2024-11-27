import numpy as np
import pymap3d
from object_definitions.Constants_def import Constants

class Reflection:
    def __init__(self, tx_satID=None, rx_satID=None, earth_model=None, angle=None, gnss_const=None):
        """
        =========================================================================================================
        Reflection class:
            - Defines a reflection object: Initialized with the following parameters:
                INPUTS:
                - tx_satID: Transmitting satellite ID (GNSS satellite IS)
                - rx_satID: Receiving satellite ID    (LEO satellite ID)
                - Earth model                         (spherical or elliptical)
                - angle                               (Cutoff angle) (default: 45 degrees)
                - gnss_const: GNSS constellation      (GPS, GLONASS, GALILEO, BEIDOU) (optional)
                
                CLASS VARIABLES:
                - time_vec: Time vector               (vector with UTC datetime objects, when reflection occurs)
                - refl_ECI_vec: Reflection points    (vector containing ECEF coordinates of point reflections)
                - refl_LL_vec: Reflection points      (vector containing Lat-Lon coordinates of point reflections)
        =========================================================================================================
        """
        # ENTRY point to the Object (Initialization)
        if earth_model is None:
            # If not specified, use the 'spherical' earth model
            print("Earth model not specified - Using default ('spherical')")
            earth_model = 'spherical'
        else:
            if not Reflection.__is_valid_earth_model(earth_model):
                # Raise a warning and return an empty object if a wrong earth model is chosen
                print("Wrong earth model specified. Use either 'spherical' or 'elliptical'.")
                return None
            else:
                # - Robustness - : If the earth model is correct, initialize the object
                pass

        if angle is None:
            # If not specified, use the default cutoff angle of 45 degrees
            print("Cutoff angle not specified - Using default (45 degrees)")
            angle = 45

        if tx_satID is None or rx_satID is None:
            Warning("Transmitting or Receiving satellite IDs are not specified    \n WARNING: Reflection object not initialized")
            return None
        self.initialization(earth_model, angle, tx_satID, rx_satID, gnss_const)   

    def initialization(self, earth_model, angle, tx_satID, rx_satID, gnss_const):
        self.earth_model       = earth_model       # 'spherical' or 'circular'
        self.gnss_const        = gnss_const        # Could be [GPS, GLONASS, GALILEO, BEIDOU]
        self.cutoff_angle      = angle
        self.tx_satID          = tx_satID
        self.rx_satID          = rx_satID
        self.time_vec          = []
        self.refl_ECI_vec     = []
        self.refl_LL_vec       = []
        self.gnss_pos_vec      = []
        self.gnss_vel_vec      = []
        self.sat_pos_vec       = []
        self.sat_vel_vec       = []
        self.refl_ang_vec      = []

    def compute_reflection_coordinates(self, GNSS_state, Sat_state, times):
        times = np.array(times)

        # Preallocate outputs for efficiency
        num_positions      = Sat_state.positions_eci.shape[1]
        theta_vec          = np.full((num_positions, 3), np.nan)
        specular_point_vec = np.full((num_positions, 3), np.nan)
        
        # Calculate the angles (theta) between GPS and satellite positions
        dot_product        = np.einsum('ij,ij->j', GNSS_state.positions_eci, Sat_state.positions_eci)
        magnitude_a        = np.linalg.norm(GNSS_state.positions_eci, axis=0)
        magnitude_b        = np.linalg.norm(Sat_state.positions_eci,  axis=0)
        cos_theta          = dot_product / (magnitude_a * magnitude_b)
        theta              = np.degrees(np.arccos(cos_theta))

        # Precompute line-of-sight (LOS) and valid theta indices for efficiency
        los_mask = np.array([
            self.__check_los_satellites(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i])
            for i in range(num_positions)
        ])
        valid_indices = (theta < self.cutoff_angle) & los_mask
    
        # Loop through satellite positions with valid LOS to compute reflection points
        for i in np.where(valid_indices)[0]:
            if self.earth_model == 'elliptical':
                specular_point_ecef, theta_degrees2, theta_degrees3 = self.find_specular_point_ellipse(Sat_state.positions_eci[:, i], GNSS_state.positions_eci[:, i])
                theta_vec[i, :]                                     = [theta_degrees2, theta_degrees3, theta[i]]
                specular_point_vec[i, :]                            = specular_point_ecef
            elif self.earth_model == 'spherical':
                reflection_point, theta_2_circ, theta_3_circ = self.get_reflection_point_FF(GNSS_state.positions_eci[:, i], Sat_state.positions_eci[:, i])
                theta_vec[i, :]                              = [theta_2_circ, theta_3_circ, theta[i]]
                specular_point_vec[i, :]                     = reflection_point

        # Compute geodetic coordinates for valid specular points
        valid_specular_points = ~np.isnan(specular_point_vec[:, 0])
        latlon_ref            = np.full((num_positions, 2), np.nan)
        if valid_specular_points.any():
            lat, lon, _ = pymap3d.eci2geodetic(
                specular_point_vec[valid_specular_points, 0],
                specular_point_vec[valid_specular_points, 1],
                specular_point_vec[valid_specular_points, 2],
                times[valid_specular_points]
            )
            latlon_ref[valid_specular_points, :] = np.column_stack((lat, lon))
            lons, lats                           = self.__prevent_longitude_wraparound(latlon_ref[:, 1], latlon_ref[:, 0])

        distance_R = np.linalg.norm((specular_point_vec - np.transpose(Sat_state.positions_eci))*1e-3, axis=1)
        distance_T = np.linalg.norm((specular_point_vec - np.transpose(GNSS_state.positions_eci))*1e-3, axis=1)

        power = 1/(distance_R**2*distance_T**2)*1e15

        # Store results in the object
        self.time_vec      = times                              # Time vector numpy array
        self.refl_ECI_vec  = specular_point_vec                 # Reflection points         ECI coordinates 
        self.refl_LL_vec   = np.column_stack((lats, lons))      # Reflection points         Lat-Lon coordinates
        self.gnss_pos_vec  = GNSS_state.positions_eci           # GNSS satellite positions  ECI coordinates
        self.gnss_vel_vec  = GNSS_state.velocities_eci          # GNSS satellite velocities ECI coordinates
        self.sat_pos_vec   = Sat_state.positions_eci            # LEO satellite positions   ECI coordinates
        self.sat_vel_vec   = Sat_state.velocities_eci           # LEO satellite velocities  ECI coordinates
        self.refl_ang_vec  = theta_vec                          # Reflection angles         ECI coordinates [deg]
        self.power         = power                              # Power of the reflection   []

    def get_reflection_point_FF(self, satPos, gpsPos):
        # Convert positions to column vectors
        satCoord = np.array(satPos).reshape(-1, 1)
        GPSCoord = np.array(gpsPos).reshape(-1, 1)

        # Check for line of sight (LOS)
        d        = satCoord - GPSCoord
        t_values = np.linspace(0, 1, 200)  # Sample points along the line
    
        # Verify LOS by checking if any point along the line passes through Earth
        for t in t_values:
            r_t                      = GPSCoord + t * d
            distance_to_earth_center = np.linalg.norm(r_t)
            if distance_to_earth_center < Constants.Radius_earth:
                # No line of sight, obstructed by Earth
                return np.array([np.nan, np.nan, np.nan]), np.nan, np.nan
    
        # Get rotation matrix
        Rot = self.getRotationEcefToPlaneCoords(satCoord, GPSCoord)
    
        # Rotate satellite and GPS coordinates
        satCoord1 = Rot @ satCoord
        GPSCoord1 = Rot @ GPSCoord
    
        # Find 2D reflection point
        N, A, B = self.get_reflection_point2D(satCoord1.flatten(), GPSCoord1.flatten())
    
        # Transform reflection point to 3D (z = 0)
        N = np.array([N[0], N[1], 0])
    
        # Convert reflection point back to ECEF coordinates
        ReflectionPoint = Rot.T @ N
    
        # Calculate angles between vectors
        vec_1 = ReflectionPoint - np.array([0, 0, 0])
        vec_2 = satPos - ReflectionPoint
        vec_3 = gpsPos - ReflectionPoint
    
        theta_degrees2 = self.__angle_between_vectors(vec_1, vec_2)
        theta_degrees3 = self.__angle_between_vectors(vec_1, vec_3)
    
        return ReflectionPoint, theta_degrees2, theta_degrees3
    
    def find_specular_point_ellipse(self, satCoord, GPSCoord):
        #TODO: Implement the elliptical model
        pass

    def getRotationEcefToPlaneCoords(self, satCoord, GPSCoord):
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
    
        Sv  = self.__skew_matrix(vec2)
        Rot = np.eye(3) + s2 * Sv + Sv @ Sv * (1 - c2)
        return Rot

    @staticmethod
    def get_reflection_point2D(satCoord1, GPSCoord1):
        # Calculates the reflection point in 2D
        A = satCoord1[:2] / Constants.Radius_earth
        B = GPSCoord1[:2] / Constants.Radius_earth

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
                    return Constants.Radius_earth * N, A, B
        return None, None, None

    @staticmethod
    def __angle_between_vectors(v1, v2):
        # Calculate the angle between two vectors in degrees
        dot_product   = np.dot(v1, v2)
        magnitude_a   = np.linalg.norm(v1)
        magnitude_b   = np.linalg.norm(v2)
        cos_theta     = dot_product / (magnitude_a * magnitude_b)
        theta_radians = np.arccos(cos_theta)
        return np.degrees(theta_radians)

    @staticmethod
    def __check_los_satellites(r1, r2)                                        -> bool:
        """
        Checks if there is a clear Line Of Sight (LOS) between two points r1 and r2,
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

            if distance_to_earth_center < Constants.Radius_earth:
                return False  # No line of sight, obstructed by Earth
        return True  # LOS is clear

    @staticmethod
    def __is_valid_earth_model(earth_model) -> bool:
        # Check if earth_model is either 'spherical' or 'elliptical'
        valid_models = ['spherical', 'elliptical']
        return earth_model in valid_models

    @staticmethod
    def __skew_matrix(v):
        # Returns the skew symmetric matrix of the vector v
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @staticmethod
    def __prevent_longitude_wraparound(lons, lats):
        # Find where the difference between consecutive longitudes exceeds 180 degrees -> avoid plotting lines from -179 to 179 deg longitude
        wraparound = np.abs(np.diff(lons)) > 180
        # Replace the longitude and latitude at the wraparound point with NaN
        lons[1:][wraparound] = np.nan
        lats[1:][wraparound] = np.nan
        return lons, lats
