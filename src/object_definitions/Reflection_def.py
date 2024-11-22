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

    # TBD
    def calculate_reflection(self):
        # Calculate Specular Reflections
        # Stub: Write function implementation here
        pass

    @staticmethod
    def __is_valid_earth_model(earth_model) -> bool:
        # Check if earth_model is either 'spherical' or 'elliptical'
        valid_models = ['spherical', 'elliptical']
        return earth_model in valid_models
