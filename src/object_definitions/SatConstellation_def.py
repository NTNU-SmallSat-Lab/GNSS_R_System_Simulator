class SatelliteConfiguration:
    """
    =========================================================================================================
    SatelliteConfiguration class:
        - Initialized with an array of Satellite objects
          or 
        - a TLE file containing the satellite configuration
    =========================================================================================================
    """
    def __init__(self, satArray=None, tle_file=None):
        # Check if the TLE file is provided
        if tle_file is not None:
            self.satellites = self.load_tle_file(tle_file)
        else:
            self.satellites = satArray


        # check if the constellation name is valid
        self.satellites     = satArray                              # An array of Satellite objects
        self.no_satellites  = len(satArray)                         # Number of satellites in the constellation
        self.orbits_array   = self.extract_orbits()                 # Array of Orbit objects TODO: write a function to extract the orbits from the satellites

    def extract_orbits(self) -> list:
        # Extract orbits from the satellite objects
        for sat in self.satellites:
            self.orbits_array.append(sat.orbit)
        return self.orbits_array

    def add_satellite(self, satellite) -> bool:
        # Add a satellite to the constellation
        self.satellites.append(satellite)
        self.no_satellites += 1
        self.orbits_array.append(satellite.orbit)
        return True

    def remove_satellite(self, satellite_id) -> bool:
        # Remove a satellite from the constellation
        for sat in self.satellites:
            if sat.sat_id == satellite_id:
                self.satellites.remove(sat)
                self.no_satellites -= 1
                self.orbits_array.remove(sat.orbit)
                return True
        return False

    def generate_constellation_TLE(self) -> bool:
        # TODO: Generate TLE data for the entire satellite constellation
        pass
