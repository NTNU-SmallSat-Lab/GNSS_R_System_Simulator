#from datetime import datetime
import re, os
from datetime import datetime, timezone, timedelta
from math import pi
from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit
from object_definitions.ReadWriteSatData import ReadWriteSatData

class Satellite:
    def __init__(self, satID, orbit=None, satName=None):
        """
        =========================================================================================================
        Satellite class:
            - A satellite object that contains satellite data.
              -> Satellite ID: Unique identifier for the satellite (int)
                -> Orbit:         Orbit object containing the satellite's orbital data (Orbit) (optional)
                -> sensors:       List of sensors on the satellite (list) TODO
                -> battery:       Battery object containing the satellite's battery data (Battery) TODO
                -> solarArray:    Solar array object containing the satellite's solar array data (SolarArray) TODO
                -> tle_file_path: Path to the TLE file (str) (optional)
                   [Only available if Satellite was initialized with an Orbit object]

        =========================================================================================================
        """
        self.sensors        = []
        self.battery        = []
        self.solarArray     = []
        self.satID          = satID
        self.satName        = satName if satName is not None else f"Sat_{satID}"
        self.classification = 'U'
        self.tle_file_path  = None  # Pointer to the TLE file

        # Initialize Satellite Orbit
        if not isinstance(orbit, Orbit) and isinstance(orbit, list):
            if len(orbit) >= 6:
                # Orbit is initialized with new orbital parameters
                self.orbit = Orbit(*orbit)
            elif len(orbit) < 6:
                print("Insuffizient data to specify Orbit of satellite: ", self.satName)
        elif orbit is None:
            print("Orbit of satellite: ", self.satName, " is not specified")
            self.orbit = None
        elif isinstance(orbit, Orbit):
            # Orbit object is passed
            self.orbit = orbit
        else:
            raise ValueError("Invalid Orbit object provided for satellite: ", self.satName)
    
    """
    =========================================================================================================
    Setter functions
    =========================================================================================================
    """
    def set_classification(self, classification) -> None:
        """Set the classification of the satellite."""
        if classification not in ['U', 'C', 'S']:
            raise ValueError("Invalid classification")
        self.classification = classification

    def set_orbit(self, orbit) -> None:
        """Set the orbit of the satellite."""
        if isinstance(orbit, Orbit):
            self.orbit = orbit
        elif isinstance(orbit, list):
            if len(orbit) == 6:
                # Orbit is initialized with new orbital parameters
                self.orbit = Orbit(*orbit)
            elif len(orbit) < 6:
                print("Insuffizient data to specify Orbit of satellite: ", self.satName)
        else:
            raise ValueError("Invalid Orbit object provided for satellite: ", self.satName)
    
    def set_epoch(self, epoch) -> None:
        """Set the epoch of the satellite."""
        if self.orbit is None:
            print("Warning: Orbit is not specified for this satellite. Cannot set epoch.")
            return
        self.orbit.set_start_time(epoch)

    """
    =========================================================================================================
    Getter functions
    =========================================================================================================
    """
    def get_TLE(self)                            -> str:
        """Get the TLE path if it exists."""
        if self.tle_exists():
            return self.tle_file_path
        else:
            print("Warning: TLE has not yet been generated")
            return None

    def tle_exists(self):
        """Check if the TLE file exists."""
        return self.tle_file_path is not None

    def write_tle_file(self):
        """
        =========================================================================
        Write the TLE object passed in "tle_str" to a file
            and return the path to the file in path_TLE
        =========================================================================
        """
        tle_str = ReadWriteSatData.generate_tle_data_string(self)
        if tle_str is None:
            print("TLE DATA CANNOT BE GENERATED FOR SATELLITE: ", self.satID)
            return None

        file_name = f"gnssr_satID_{self.satID}.tle"
#        filePath  = f"{Constants.tle_folderpath}/autogenerated/{file_name}"
        filePath = os.path.join(Constants.tle_eph_folder_path, file_name)

        # Make sure the directory exists
        if not os.path.exists(Constants.tle_eph_folder_path):
            print(f"Creating directory: {Constants.tle_eph_folder_path}")
            os.makedirs(Constants.tle_eph_folder_path)

        # Check if the file already exists
        if os.path.exists(filePath):
            print(f"Warning: TLE file already exists for {self.satID} \n-- Overwriting previous TLE file --")
            # delete the file
            os.remove(filePath)

        # Write the TLE to the file
        try:
            with open(filePath, 'w') as file:
                file.write(tle_str)
            self.tle_file_path = filePath
            print(f"TLE written to {filePath}")
        except IOError:
            print(f"Warning: TLE file could not be written for {self.satID}")
