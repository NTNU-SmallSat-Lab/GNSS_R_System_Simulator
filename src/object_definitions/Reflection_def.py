import numpy as np

class Reflection:
    def __init__(self, earth_model=None):
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

        self.earth_model       = earth_model      # 'spherical' or 'circular'
        self.sat_const         = 'GPS'            # For now hardcoded to 'GPS', could be [GPS, GLONASS, GALILEO, BEIDOU]
        self.reflection_points = {
                                'time':     [], #TBD
                                'reflECEF': [], #TBD
                                'reflLL':   [], #TBD
                                'gnssPos':  [], #TBD
                                'gnssVel':  [], #TBD
                                'satPos':   [], #TBD
                                'satVel':   []  #TBD
                                
                                }
        self.tx_satID          = None
        self.rx_satID          = None

    def calculate_reflection(self, finalPositionGPS, finalPositionSat, ellipse, angle, times):
        # Calculate Specular Reflections
        radius = 6378137  # Earth radius in meters
        # Preallocate outputs for efficiency
        theta_vec          = np.full((len(finalPositionSat[0]), 3), np.nan)
        specular_point_vec = np.full((len(finalPositionSat[0]), 3), np.nan)
        # Calculate the angles (theta) between GPS and satellite positions
        dot_product = np.einsum('ij,ij->j', finalPositionGPS, finalPositionSat)
        magnitude_a = np.linalg.norm(finalPositionGPS, axis=0)
        magnitude_b = np.linalg.norm(finalPositionSat, axis=0)
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        theta = np.degrees(np.arccos(cos_theta))
        # Loop through each satellite position to compute reflection points
        for i in range(finalPositionSat.shape[1]):
            has_los = check_los_satellites(finalPositionSat[:, i] / 1e3, finalPositionGPS[:, i] / 1e3, radius / 1e3)
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
    
    @staticmethod
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
    

    @staticmethod
    def __is_valid_earth_model(earth_model) -> bool:
        # Check if earth_model is either 'spherical' or 'elliptical'
        valid_models = ['spherical', 'elliptical']
        return earth_model in valid_models
