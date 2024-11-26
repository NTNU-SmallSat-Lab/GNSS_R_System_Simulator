import numpy as np
import pymap3d
from object_definitions.Constants_def import Constants

class Reflection:
    def __init__(self, earth_model=None, angle=None, tx_satID=None, rx_satID=None, gnss_const=None):
        """
        =========================================================================================================
        Reflection class:
            - Defines a reflection object: Initialized with the following parameters:
                - Earth model (spherical or elliptical)
                - Reflection points (time, reflECEF, reflLL, gnssPos, gnssVel, satPos, satVel)
                - Transmitting satellite ID
                - Receiving satellite ID
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

        self.earth_model       = earth_model       # 'spherical' or 'circular'
        self.gnss_const        = gnss_const        # Could be [GPS, GLONASS, GALILEO, BEIDOU]
        self.cutoff_angle      = angle
        self.tx_satID          = tx_satID
        self.rx_satID          = rx_satID
        self.time_vec          = []
        self.refl_ECEF_vec     = []
        self.refl_LL_vec       = []
        self.gnss_pos_vec      = []
        self.gnss_vel_vec      = []
        self.sat_pos_vec       = []
        self.sat_vel_vec       = []
        self.refl_ang_vec      = []


    def calculate_reflection(self, finalPositionGNSS, finalPositionSat, times):
        # Calculate Specular Reflections

        # Preallocate outputs for efficiency
        theta_vec          = np.full((len(finalPositionSat.positions_eci[0]), 3), np.nan)
        specular_point_vec = np.full((len(finalPositionSat.positions_eci[0]), 3), np.nan)
        # Calculate the angles (theta) between GPS and satellite positions
        dot_product = np.einsum('ij,ij->j', finalPositionGNSS.positions_eci, finalPositionSat.positions_eci)
        magnitude_a = np.linalg.norm(finalPositionGNSS.positions_eci, axis=0)
        magnitude_b = np.linalg.norm(finalPositionSat.positions_eci, axis=0)
        cos_theta   = dot_product / (magnitude_a * magnitude_b)
        theta       = np.degrees(np.arccos(cos_theta))

        # Loop through each satellite position to compute reflection points
        for i in range(finalPositionSat.positions_eci.shape[1]):
            has_los = self.__check_los_satellites(finalPositionSat.positions_eci[:, i], finalPositionGNSS.positions_eci[:, i])
            if theta[i] < self.cutoff_angle and has_los:
                # Depending on 'ellipse' flag, call appropriate reflection calculation
                if self.earth_model == 'elliptical':
                    specular_point_ecef, theta_degrees2, theta_degrees3 = self.find_specular_point_ellipse(finalPositionSat.positions_eci[:, i], finalPositionGNSS.positions_eci[:, i])
                    theta_vec[:, i]                                     = [theta_degrees2, theta_degrees3, theta[i]]
                    specular_point_vec[:, i]                            = specular_point_ecef
                elif self.earth_model == 'spherical':
                    reflection_point, theta_2_circ, theta_3_circ = self.get_reflection_point_FF(finalPositionGNSS.positions_eci[:, i], finalPositionSat.positions_eci[:, i])
                    theta_vec[i, :]                              = [theta_2_circ, theta_3_circ, theta[i]]
                    specular_point_vec[i, :]                     = reflection_point
        #print(theta_vec)
        # Convert ECEF specular points to latitude and longitude
            #print(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])
        #latlon_ref = np.full((len(finalPositionSat.positions_eci[0]), 2), np.nan)
        latlon_ref = np.full((finalPositionSat.positions_eci.shape[1], 2), np.nan)
        for i in range(finalPositionSat.positions_eci.shape[1]):
            if not np.isnan(specular_point_vec[i, 0]):

                lat, lon, alt = pymap3d.eci2geodetic(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2], times[i])
                #lat, lon, alt =  ecef_to_lla(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])
                latlon_ref[i, :] = [lat, lon]
                #latlon_ref[i, :] = ecef_to_lla(specular_point_vec[i, 0], specular_point_vec[i, 1], specular_point_vec[i, 2])  # Ensure `ecef_to_lla` is defined or available
        # Store the reflection points in the object
        self.time_vec          = times
        self.refl_ECEF_vec     = specular_point_vec
        self.refl_LL_vec       = latlon_ref
        self.gnss_pos_vec      = finalPositionGNSS.positions_eci
        self.gnss_vel_vec      = finalPositionGNSS.velocities_eci
        self.sat_pos_vec       = finalPositionSat.positions_eci
        self.sat_vel_vec       = finalPositionSat.velocities_eci
        self.refl_ang_vec      = theta_vec
        
        return specular_point_vec, theta_vec, latlon_ref
    
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
