import os, math
from object_definitions.Satellite_def    import Satellite
from object_definitions.Constants_def    import Constants
from object_definitions.ReadWriteSatData import ReadWriteSatData as rw

class SatelliteConstellation:
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
                satList  = rw.read_tle_data(tle_file)
                satArray = []
                satNo   = 0
                for sat_data in satList:
                    sat = Satellite(sat_data.satCatalogNumber, satName=sat_data.satName, tle_file_path=tle_file, satNo_in_tle=satNo)
                    satArray.append(sat)
                    satNo += 1
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

    def add_satellite(self, satellite)       -> bool:
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

    def generate_constellation_TLE(self)     -> str:
        tle_str = ""
        for sat in self.satellites:
            tle_str += rw.generate_tle_data_string(sat)
            tle_str += "\n"

        filePath = os.path.join(Constants.tle_eph_folder_path, f"{self.satConfigName}.tle")
        if tle_str is None:
            print("TLE DATA CANNOT BE GENERATED FOR SATELLITE CONSTELLATION: ", self.satConfigName)
            return None

        # Make sure the directory exists
        if not os.path.exists(Constants.tle_eph_folder_path):
            print(f"Creating directory: {Constants.tle_eph_folder_path}")
            os.makedirs(Constants.tle_eph_folder_path)

        # Check if the file already exists
#        if os.path.exists(filePath):
#            print(f"Warning: TLE file already exists for {self.satConfigName } \n-- Overwriting previous TLE file --")
#            # delete the file
#            os.remove(filePath)
        
        
        base_name = 'Constellation'  # Keep the original name
        filePath = os.path.join(Constants.tle_eph_folder_path, f"{base_name}.tle")
        incrementing_number = 0

        # Check for existing files and generate a unique name
        while os.path.exists(filePath):
            incrementing_number += 1
            filePath = os.path.join(Constants.tle_eph_folder_path, f"{base_name}_{incrementing_number}.tle")
            
        '''
        incrementing_number = 0
        while os.path.exists(filePath):
            # HACK - Append a number to the file name
            self.satConfigName += f"_{incrementing_number}"
            incrementing_number += 1
            filePath = os.path.join(Constants.tle_eph_folder_path, f"{self.satConfigName}.tle")
        '''

        # Write the TLE to the file
        try:
            with open(filePath, 'w') as file:
                file.write(tle_str)
            self.tle_file_path = filePath
            print(f"TLE written to {filePath}")
        except IOError:
            print(f"Warning: TLE file could not be written for {self.satID}")
            return None

        return filePath
    
    def get_constellation_name(self)        -> str:
        return self.satConfigName
    
    def get_constellation_size(self)        -> int:
        return self.no_satellites
    
    def get_constellation_orbits(self)      -> list:
        return self.orbits_array
    
    def get_constellation_TLE(self)         -> str:
        return self.tle_file_path

    def propagate_constellation(self, start_time, duration, step_size) -> dict:
        # Propagate the entire constellation
        for sat in self.satellites:
            sat.propagate_orbit(start_time, duration, step_size)
        pass

#TODO - Add the following functions:
# - get_satellite_by_id
# - get_satellite_by_name

# - propagate_constellation

# - get_satellite_positions

