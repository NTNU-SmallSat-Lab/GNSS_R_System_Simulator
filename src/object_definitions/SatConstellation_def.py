import os, math
from object_definitions.Satellite_def import Satellite
from object_definitions.Constants_def import Constants
from object_definitions.ReadWriteSatData import ReadWriteSatData as rw

class SatelliteConfiguration:
    """
    =========================================================================================================
    SatelliteConfiguration class:
        - Initialized with an array of Satellite objects
          or 
        - a TLE file containing the satellite configuration
        ! NOTE: This class should be initialized with either a list of Satellite objects or a TLE file
                But not both !
    =========================================================================================================
    """
    def __init__(self, satArray=None, tle_file=None, satConfigName=None):
        # Check if the TLE file is provided / Robustness checks
        if tle_file is not None and satArray is not None:
            raise ValueError("Cannot initialize SatelliteConfiguration with both TLE file and Satellite array")
        elif satArray is None and tle_file is None:
            raise ValueError("SatelliteConfiguration must be initialized with either a TLE file or a Satellite array")
        elif satArray is None and tle_file is not None:
            if os.path.exists(tle_file):
                # Read the TLE data from the file
                print(f"Initializing Satellite Configuration with TLE file: {tle_file}")
                satList = rw.read_tle_data(tle_file)
                satArray = []
                for sat_data in satList:
                    # calculate semi major axis
                    semi_major_axis = (Constants.GM_earth_tt / (sat_data.meanMotion * 2 * math.pi / Constants.Sec_per_day) ** 2) ** (1/3)
                    orbit_array     = [sat_data.inclination, sat_data.eccentricity, sat_data.raan, sat_data.argOfPerigee, sat_data.meanAnomaly, semi_major_axis, sat_data.epoch]
                    sat             = Satellite(sat_data.satCatalogNumber, orbit_array, sat_data.satName)
                    satArray.append(sat)
            else:
                raise FileNotFoundError("TLE file not found")
        elif satArray is not None and tle_file is None:
            if isinstance(satArray, list):
                # Use the provided satellite array
                print(f"Initializing Satellite Configuration with satellite array")
                pass
            else:
                raise ValueError("Invalid input for Satellite array")
        else:
            raise ValueError("Invalid input")

        # check if the constellation name is valid
        self.satConfigName  = satConfigName
        self.satellites     = satArray                                           # An array of Satellite objects
        self.no_satellites  = len(satArray)                                      # Number of satellites in the constellation
        self.orbits_array   = [sat_i.get_orbit() for sat_i in self.satellites]   # Array of Orbit objects

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


