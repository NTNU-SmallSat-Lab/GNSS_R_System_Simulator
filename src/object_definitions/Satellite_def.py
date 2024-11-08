#from datetime import datetime
import re, os
from math import pi
from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit

class Satellite:
    def __init__(self, satID, *args):
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
        self.sensors       = []
        self.battery       = []
        self.solarArray    = []
        self.satID         = satID
        self.orbit         = None
        self.tle_file_path = None  # Pointer to the TLE file

        # Initialize satellite based on the number of arguments
        if len(args) == 0:
            # No Orbit passed
            self.__initialize_noOrbit()
        elif len(args) == 1 and isinstance(args[0], Orbit):
            # Orbit object passed
            self.__initialize_Orbit(args[0])
        elif len(args) == 7:
            # Orbital parameters passed
            i, e, raan, p, M, a, epochTime = args
            self.__initialize_Orbit(Orbit(i, e, raan, p, M, a, epochTime))
        else:
            raise ValueError("Wrong number of inputs \n-- Satellite initialization failed --  \n    - A Satellite must be initialized with either: \n        1. No arguments \n        2. An Orbit object \n        3. Orbital parameters (i, e, raan, p, M, a, epochTime)")

    def __initialize_noOrbit(self):
        """Initialize satellite with no orbit."""
        pass

    def __initialize_Orbit(self, orbit):
        """Initialize satellite with an Orbit object and generate TLE."""
        self.orbit = orbit
        self.generate_TLE()

    def get_TLE(self):
        """Get the TLE path if it exists."""
        if self.tle_exists():
            return self.tle_file_path
        else:
            print("Warning: TLE has not yet been generated")
            return None

    def generate_TLE(self):
        """
        =========================================================================
        Generate TLE based on satellite orbit data.
        =========================================================================
        """
        if self.orbit.epoch is None:
            print("Warning: Epoch was not specified when this Orbit was created.\n-- Insufficient information to generate a TLE --")
            return

        balistic_coeff = -0.00000001 # TODO calculate balistic_coeff in a separate function
        meanMotion     = 0           # TODO calculate meanMotion in a separate function
        B_star         = -0.00000001 # TODO calculate B_star in a separate function

        # Line 0
        line0_str = f"GNSS-R {self.satID}"

        # Line 1
        line_no_1_str   = "1"                                                                               # Line number "1"
        satIDStr        = f"{self.satID:05d}"                                                               # Sat_<n> n is satellite, resulting in "Sat_1.tle"               
        classStr        = "U"                                                                               # The satellite is unclassified [Can be U: Unclassified, C: Classified, S: Secret]
        IntD_LYear      = f"{self.orbit.dt_start.year % 100:02d}"                                           # International Designator (last two digits of launch year) COSPAR ID
        IntD_LNo        = "001"                                                                             # International Designator (launch no. of the year)         COSPAR ID
        IntD_POL        = "A"                                                                               # International Designator (piece of the launch)            COSPAR ID [hardcoded to 'A']
        epochStr        = self.orbit.epoch                                                                  # Epoch in TLE format [YYDDD.DDDDDDDD]
        balisticCoefStr  = re.sub(r'-0', '-', f"{balistic_coeff:10.8f}")                                    # Balistic coeffizient / First derivative of the mean motion in TLE format
        meanMotionStr   = self.__str2tle_format(meanMotion)                                                 # Second derivative of Mean Motion (decimal point assumed) HARDCODED FOR NOW /TODO 
        b_star_str      = self.__str2tle_format(B_star)                                                     # B* the drag term, or radiation pressure coefficient (decimal point assumed) e.g. 1.234e-8 => 12345-8 [https://en.wikipedia.org/wiki/BSTAR] TODO
        ephemeris_str   = "0"                                                                               # Ephemeris type (always zero; only used in undistributed TLE data) (0: unclassified, 1: classified) [hardcoded to '0']               
        elem_st_no_str  = "0001"                                                                            # Element set number. Incremented when a new TLE is generated for this object TODO
        checksum1       = self.__calc_checksum([line_no_1_str, satIDStr, IntD_LYear, IntD_LNo, epochStr, balisticCoefStr, meanMotionStr, b_star_str, ephemeris_str, elem_st_no_str])
        checksum1       = ""                                                                                # Checksum for line 1 (modulo 10) [hardcoded to '0']

        line1_str = f"{line_no_1_str} {satIDStr}{classStr} {IntD_LYear}{IntD_LNo}{IntD_POL}   {epochStr} {balisticCoefStr} {meanMotionStr} {b_star_str} {ephemeris_str} {elem_st_no_str}{checksum1}"

        # Line 2
        line_no_2_str   = "2"                                                                               # Line number "2"
        satcat_str      = "99999"                                                                           # Satellite Catalog Number (5 digits) [hardcoded to '99999']
        i_str           = f"{self.orbit.i:8.4f}"                                                            # Inclination in degrees
        raan_str        = f"{self.orbit.RAAN:8.4f}"                                                         # Right Ascension of Ascending Node in degrees
        eccen_str       = f"{int(self.orbit.e * 1e7):07d}"                                                  # Eccentricity  * 10^7 as decimal value / decimal point assumed
        per_str         = f"{self.orbit.p:8.4f}"                                                            # Argument of perigee in degrees
        mean_anom_str   = f"{self.orbit.M:8.4f}"                                                            # Mean anomaly
        rev_per_day     = (self.orbit.n / (2 * pi)) * Constants.Sec_per_day
        mean_Mot_str    = f"{rev_per_day:11.8f}"                                                            # Revolutions per day
        rev_str         = f"{0:05d}"                                                                        # Revolution number at epoch [hardcoded to '00000']
        checksum2       = self.__calc_checksum([line_no_2_str, satcat_str, i_str, raan_str, eccen_str, per_str, mean_anom_str, mean_Mot_str, rev_str]) # Checksum for line 2 (modulo 10)

        line2_str = f"{line_no_2_str} {satcat_str} {i_str} {raan_str} {eccen_str} {per_str} {mean_anom_str} {mean_Mot_str}{rev_str}{checksum2}"

        # Combine the lines to the full TLE
        tle_str = f"{line0_str}\n{line1_str}\n{line2_str}"

        # Write to file
        self.__write_new_tle_file(tle_str)

    def tle_exists(self):
        """Check if the TLE file exists."""
        return self.tle_file_path is not None

    def __str2tle_format(self, value):
        """
        =========================================================================
        str2tle_format:
            - Convert one of the following numbers:
               "Second derivative of mean motion"
               "B*, the drag term"
              from its numeric value to the format:
              <sign>XXXXX<sign>X e.g. 1.2345e-4 => ' 12345-4
        =========================================================================
        """
        # Sign of the value
        if value >= 0:
            sign = " "
        else:
            sign = "-"

        value        = abs(value)                                     # Absolute value
        sci_notation = f"{value:.4e}"
        #sci_notation = f"{value:.4e}".split("e")                      # Scientific notation

        # Split the scientific notation into mantissa and exponent parts
        parts        = sci_notation.split("e")
        matissa      = parts[0].replace(".", "")[:5]
        exponentStr  = parts[1].lstrip("+")
        exponent     = int(exponentStr)
        if exponent >= 0:
            exp_sign = "+"
        else:
            exp_sign = "-"

        # Format the string according to TLE format
        exponent_str = f"{exp_sign}{abs(exponent):01d}"

        tle_str = f"{sign}{matissa}{exponent_str}"

        return tle_str

    def __write_new_tle_file(self, tle_str):
        """
        =========================================================================
        Write the TLE object passed in "tle_str" to a file
            and return the path to the file in path_TLE
        =========================================================================
        """
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

    @staticmethod
    def __calc_checksum(stringArray):
        """
        =========================================================================
        Calculate TLE checksum.
        =========================================================================
        """
        total = 0
        for elem in stringArray:
            for char in elem:
                if char.isdigit():
                    total += int(char)
                elif char == '-':
                    total += 1
        return total % 10
