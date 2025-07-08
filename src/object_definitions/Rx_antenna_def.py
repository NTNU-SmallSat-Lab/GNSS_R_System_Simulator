class RxAntenna:
    def __init__(self, antenna_gain=None):
        """
        Initialize the RxAntenna object.

        INPUTS:
            antenna_id: [str] Unique identifier for the antenna
            position:   [tuple] (latitude, longitude) position of the antenna in degrees
            orientation:[tuple] (azimuth, elevation) orientation angles in degrees
            gain_pattern:[np.array] Optional gain pattern of the antenna
        """
        self.antenna_gain_x = antenna_gain[0] if antenna_gain is not None else 0.0  # Antenna gain in dB
        self.antenna_gain_y = antenna_gain[1] if antenna_gain is not None else 0.0  # Antenna gain in dB
        # - Position and orientation on the satellite (relative to the satellite coordinate frame)
        # -
