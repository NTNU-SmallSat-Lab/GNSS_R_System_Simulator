from collections import namedtuple
import datetime  # Import the module itself first
# ✅ Monkey patch for Python < 3.11 compatibility
if not hasattr(datetime, "UTC"):
    from datetime import timezone
    datetime.UTC = timezone.utc
from datetime import datetime, timezone, timedelta
import os
import re

# import fortranformat as ff


class ReadWriteSatData:
    def __init__(self):
        pass

    @staticmethod
    def generate_tle_data_string(satellite) -> str:
        """
        =========================================================================
        Generate TLE based on satellite orbit_n data.
        =========================================================================
        """
        if satellite.orbit_n is None:
            print("Warning: Orbit was not specified when this Satellite was created.\n-- Insufficient information to generate a TLE --")
            return None
        if satellite.orbit_n.epoch_str is None:
            print("Warning: Epoch was not specified when this Orbit was created.\n-- Insufficient information to generate a TLE --")
            return None

        balistic_coeff = satellite.meanMotion_1 if satellite.meanMotion_1 is not None else 0
        meanMotionSecDer = satellite.meanMotion_2 if satellite.meanMotion_2 is not None else 0
        B_star = satellite.B_star if satellite.B_star is not None else 0
        element_set_no = satellite.element_set_no if satellite.element_set_no is not None else 1

        # Line 0
        line0_str = satellite.satName  # Satellite name

        # Line 1
        line_no_1_str = "1"  # Line number "1"
        satIDStr = f"{satellite.satID:05d}"  # Sat_<n> n is satellite, resulting in "Sat_1.tle"
        classStr = satellite.classification  # The satellite is unclassified [Can be U: Unclassified, C: Classified, S: Secret]
        IntD_LYear = f"{satellite.launch_year % 100:02d}"  # International Designator (last two digits of launch year) COSPAR ID
        IntD_LNo = f"{int(satellite.lauch_Noyear):03d}"  # International Designator (launch no. of the year)         COSPAR ID
        IntD_POL = satellite.lauch_piece  # International Designator (piece of the launch)            COSPAR ID [hardcoded to 'A']
        epochStr = satellite.orbit_n.epoch_str  # Epoch in TLE format [YYDDD.DDDDDDDD]
        balisticCoefStr = re.sub(r"-0", "-", f"{balistic_coeff:10.8f}")  # Balistic coeffizient / First derivative of the mean motion in TLE format
        balisticCoefStr = balisticCoefStr.replace("0.", " .") if balistic_coeff > 0 else balisticCoefStr.replace("0.", "-.")  # Remove the decimal point
        meanMotionStr_2 = ReadWriteSatData._str2tle_format(
            meanMotionSecDer
        )  # Second derivative of Mean Motion (decimal point assumed) HARDCODED FOR NOW /TODO
        b_star_str = ReadWriteSatData._str2tle_format(
            B_star
        )  # B* the drag term, or radiation pressure coefficient (decimal point assumed) e.g. 1.234e-8 => 12345-8 [https://en.wikipedia.org/wiki/BSTAR] TODO
        ephemeris_str = "0"  # TODO                                                                                         # Ephemeris type (always zero; only used in undistributed TLE data) (0: unclassified, 1: classified) [hardcoded to '0']
        elem_st_no_str = f"{element_set_no:03d}"  # Element set number. Incremented when a new TLE is generated for this object TODO
        checksum1 = ReadWriteSatData._calc_checksum(
            [
                line_no_1_str,
                satIDStr,
                IntD_LYear,
                IntD_LNo,
                epochStr,
                balisticCoefStr,
                meanMotionStr_2,
                b_star_str,
                ephemeris_str,
                elem_st_no_str,
            ]
        )
        # Checksum for line 1 (modulo 10)

        line1_str = f"{line_no_1_str} {satIDStr}{classStr} {IntD_LYear}{IntD_LNo}{IntD_POL}   {epochStr} {balisticCoefStr} {meanMotionStr_2} {b_star_str} {ephemeris_str} {elem_st_no_str}{checksum1}"

        # Line 2
        line_no_2_str = "2"  # Line number "2"
        satcat_str = satIDStr  # Satellite Catalog Number (5 digits) [same as satID]
        i_str = f"{satellite.orbit_n.i:8.4f}"  # Inclination in degrees
        raan_str = f"{satellite.orbit_n.RAAN:8.4f}"  # Right Ascension of Ascending Node in degrees
        eccen_str = f"{int(satellite.orbit_n.e * 1e7):07d}"  # Eccentricity  * 10^7 as decimal value / decimal point assumed
        per_str = f"{satellite.orbit_n.p:8.4f}"  # Argument of perigee in degrees
        mean_anom_str = f"{satellite.orbit_n.M:8.4f}"  # Mean anomaly
        # rev_per_day     = (satellite.orbit_n.n / (2 * math.pi)) * Constants.Sec_per_day
        mean_Mot_str = f"{satellite.orbit_n.n:11.8f}"  # Revolutions per day
        rev_str = f"{0:05d}"  # Revolution number at epoch [hardcoded to '00000']
        checksum2 = ReadWriteSatData._calc_checksum(
            [
                line_no_2_str,
                satcat_str,
                i_str,
                raan_str,
                eccen_str,
                per_str,
                mean_anom_str,
                mean_Mot_str,
                rev_str,
            ]
        )  # Checksum for line 2 (modulo 10)

        line2_str = f"{line_no_2_str} {satcat_str} {i_str} {raan_str} {eccen_str} {per_str} {mean_anom_str} {mean_Mot_str}{rev_str}{checksum2}"

        # Combine the lines to the full TLE
        tle_str = f"{line0_str}\n{line1_str}\n{line2_str}"
        return tle_str

    @staticmethod
    def read_tle_data(tle_file_path) -> list[namedtuple]:
        """
        Read TLE data from a file and return a list of Satellite objects
        """
        # check if the file exists
        if not os.path.exists(tle_file_path):
            print(f"File {tle_file_path} does not exist")
            return None

        # Initialize variables
        satList = []
        SatStruct = namedtuple(
            "SatStruct",
            [
                "satName",
                "satCatalogNumber",
                "classification",
                "launchYear",
                "lauchNoInYear",
                "launchPiece",
                "epoch",
                "firstDerMeanMotion",
                "secDerivMeanMotion",
                "B_star",
                "EphType",
                "ElementSetNo",
                "inclination",
                "raan",
                "eccentricity",
                "argOfPerigee",
                "meanAnomaly",
                "meanMotion",
                "revNoAtEpoch",
            ],
        )
        sat_no = 0

        readNextSat = True

        # Read TLE data from a file
        # open file
        with open(tle_file_path) as file:
            tle_data = file.readlines()
        # Read TLE data line by line
        for line in tle_data:
            if readNextSat == True:
                if line.startswith("1"):
                    line_1 = line.strip()
                    # Extract satellite ID
                    if line_1[0] == "1":
                        satID_1_str = line_1[2:7]
                        classification_str = line_1[7:8]
                        launchYear_str = line_1[9:11]
                        lauchNoYear_str = line_1[11:14]
                        pieceOfLaunch_str = line_1[14:17].strip()
                        epochYear_str = line_1[18:20]
                        epochDay_str = line_1[20:32]
                        firstDerMean_str = line_1[33:43]
                        secondDerMean_str = line_1[44:52]
                        dragTerm_str = line_1[53:61]
                        ephType_str = line_1[62:63]
                        elSetNo_str = line_1[64:68]
                        if len(line_1) > 69:
                            checksum_str1 = line_1[68:69]
                            calc_checksum1 = ReadWriteSatData._calc_checksum([line_1[:68]])
                            if not (calc_checksum1 == int(checksum_str1)):
                                Warning("Checksum error in TLE data line 1")
                    else:
                        ValueError("Invalid TLE data format in line 1")

                elif line.startswith("2"):
                    line_2 = line.strip()
                    # Extract satellite ID
                    if line_2[0] == "2":
                        satID_2_str = line_2[2:7]
                        incl_str = line_2[8:16]
                        raan_str = line_2[17:25]
                        ecc_str = line_2[26:33]
                        argPeri_str = line_2[34:42]
                        meanAno_str = line_2[43:51]
                        meanMotion_str = line_2[52:63]
                        revNoatEpoch_str = line_2[63:68]
                        checksum_str2 = line_2[68:69]

                        calc_checksum2 = ReadWriteSatData._calc_checksum([line_2[:68]])

                        if not (calc_checksum2 == int(checksum_str2)):
                            Warning("Checksum error in TLE data line 2")
                        if not (satID_1_str == satID_2_str):
                            Warning("Satellite ID mismatch in line 1 and line 2")
                    else:
                        ValueError("Invalid TLE data format in line 2")

                    readNextSat = False

                else:
                    # Header (satellite ID / satellite name)
                    sat_name_str = line.strip()

                    # Set TLE data to None before reading the next satellite
                    satID_1_str = None
                    classification_str = None
                    launchYear_str = None
                    lauchNoYear_str = None
                    pieceOfLaunch_str = None
                    epochYear_str = None
                    epochDay_str = None
                    firstDerMean_str = None
                    secondDerMean_str = None
                    dragTerm_str = None
                    ephType_str = None
                    elSetNo_str = None
                    satID_2_str = None
                    incl_str = None
                    raan_str = None
                    ecc_str = None
                    argPeri_str = None
                    meanAno_str = None
                    meanMotion_str = None
                    revNoatEpoch_str = None
                    checksum_str1 = None
                    checksum_str2 = None
            if readNextSat == False:
                satName = sat_name_str.strip()  # Satellite name                       [str]
                satCatalogNumber = int(satID_1_str)
                classification = classification_str
                launchYear = (
                    2000 + int(launchYear_str.strip()) if int(launchYear_str.strip()) < 50 else 1900 + int(launchYear_str.strip())
                )  # Year of epoch                        [YYYY]
                lauchNoInYear = int(lauchNoYear_str.strip())  # Launch number of the year            [-]
                launchPiece = pieceOfLaunch_str.strip()  # Piece of launch                      [-]
                epochYear = (
                    2000 + int(epochYear_str.strip()) if int(epochYear_str.strip()) < 50 else 1900 + int(epochYear_str.strip())
                )  # Year of epoch                        [YYYY]
                epochDay = float(epochDay_str.strip())  # Day of the year                      [DDD.DDDDDDDD]
                epoch = ReadWriteSatData.__get_datetime(epochYear, epochDay)  # Epoch time                           [datetime]
                firstDerMeanMotion = ReadWriteSatData.__tle_to_ballistic_coef(firstDerMean_str)  # Ballistic coefficient                [TODO]
                secDerivMeanMotion = ReadWriteSatData.__tle_to_MeanM_Bstar(secondDerMean_str)  # Second derivative of mean motion     [TODO]
                B_star = ReadWriteSatData.__tle_to_MeanM_Bstar(dragTerm_str)  # B*, the drag term                    [TODO]
                EphType = int(ephType_str.strip())  # Ephemeris type                       [0: unclassified, 1: classified]
                ElementSetNo = int(elSetNo_str.strip())  # Element set number                   [-]

                incl = float(incl_str.strip())  # Inclination                          [deg]                       # i
                raan = float(raan_str.strip())  # Right Ascension of Ascending Node    [deg]                       # Omega
                ecc = float(f"0.{ecc_str.strip()}")  # Eccentricity                         [-]                    # e
                argPeri = float(argPeri_str.strip())  # Argument of perigee                  [deg]                 # omega
                meanAno = float(meanAno_str.strip())  # Mean anomaly                         [deg]                 # M
                meanMotion = float(meanMotion_str.strip())  # Mean motion                          [rev/day]
                revNoatEpoch = int(revNoatEpoch_str.strip())  # Revolution number at epoch           [-]

                # Write data into the namedtuple
                satList.append(
                    SatStruct(
                        satName,
                        satCatalogNumber,
                        classification,
                        launchYear,
                        lauchNoInYear,
                        launchPiece,
                        epoch,
                        firstDerMeanMotion,
                        secDerivMeanMotion,
                        B_star,
                        EphType,
                        ElementSetNo,
                        incl,
                        raan,
                        ecc,
                        argPeri,
                        meanAno,
                        meanMotion,
                        revNoatEpoch,
                    )
                )
                sat_no += 1
                readNextSat = True
        return satList

    @staticmethod
    def download_and_read_tle(path) -> bool:
        """
        =========================================================================
        Download TLE data from a website and read it.
        =========================================================================
        """
        # Download TLE data from a website

    @staticmethod
    def __tle_to_ballistic_coef(balistic_coeff_str) -> float:
        """
        =========================================================================
        Convert the first derivative of mean motion to physical units.
        =========================================================================
        """
        sign = balistic_coeff_str[0]
        if sign == "-":
            sign = -1
        else:
            sign = 1
        return sign * float(balistic_coeff_str[1:])

    @staticmethod
    def __tle_to_MeanM_Bstar(secDerMeanM_str) -> float:
        """
        =========================================================================
        Convert the second derivative of mean motion to physical units.
        =========================================================================
        """
        sign = secDerMeanM_str[0]
        exp_sign = secDerMeanM_str[6]
        if sign == "-":
            sign = -1
        else:
            sign = 1

        if exp_sign == "-":
            exp_sign = -1
        else:
            exp_sign = 1

        mantissa = float(("0." + secDerMeanM_str[1:6]).strip())
        exponent = float(secDerMeanM_str[7:8].strip())

        return sign * mantissa * 10 ** (exp_sign * exponent)

    @staticmethod
    def __get_datetime(year, days) -> datetime:
        """
        =========================================================================
        Convert year and days to datetime.
        =========================================================================
        """
        # Get the start of the year
        start_of_year = datetime(year, 1, 1)

        # Convert the fractional day to a timedelta
        day_fraction, whole_day = days % 1, int(days)
        day_time = timedelta(days=day_fraction)

        # Calculate the datetime by adding the whole days and fractional day time
        result_datetime = start_of_year + timedelta(days=whole_day - 1) + day_time
        result_datetime = result_datetime.replace(tzinfo=timezone.utc)

        return result_datetime

    @staticmethod
    def _calc_checksum(stringArray) -> int:
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
                elif char == "-":
                    total += 1
        return total % 10

    @staticmethod
    def _str2tle_format(value) -> str:
        """
        =========================================================================
        str2tle_format:
            - Convert one of the following numbers:
               "Second derivative of mean motion"
               "B*, the drag term"
              from its numeric value to the format:
              <sign>XXXXX<sign>X e.g. 1.2345e-4 => ' 12345-3
        =========================================================================
        """
        # Sign of the value
        value = value * 10  # Multiply value by 10 to get the exponent right
        if value >= 0:
            sign = " "
        else:
            sign = "-"

        value = abs(value)  # Absolute value
        sci_notation = f"{value:.4e}"
        # sci_notation = f"{value:.4e}".split("e")                      # Scientific notation

        # Split the scientific notation into mantissa and exponent parts
        parts = sci_notation.split("e")
        matissa = parts[0].replace(".", "")[:5]
        exponentStr = parts[1].lstrip("+")
        exponent = int(exponentStr)
        if exponent >= 0:
            exp_sign = "+"
        else:
            exp_sign = "-"

        # Format the string according to TLE format
        exponent_str = f"{exp_sign}{abs(exponent):01d}"

        tle_sub_str = f"{sign}{matissa}{exponent_str}"

        return tle_sub_str
