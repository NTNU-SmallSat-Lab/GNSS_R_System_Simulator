from collections import namedtuple
import datetime as datetime_module  # Import the module itself first
from datetime import datetime, timedelta
import math
import os
import warnings

import numpy as np
from scipy.optimize import least_squares
from sgp4.api import WGS72, Satrec, jday
from skyfield.api import EarthSatellite, load

from object_definitions.Constants_def import Constants

# import fortranformat as ff

_SGP4_EPOCH_JD_OFFSET = 2433281.5
_TIMESCALE = None


def _get_timescale():
    global _TIMESCALE
    if _TIMESCALE is None:
        _TIMESCALE = load.timescale()
    return _TIMESCALE


def _wrap_rad(angle):
    return float(angle % (2.0 * np.pi))


def _wrap_deg(angle_rad):
    return float(np.degrees(angle_rad) % 360.0)


def _tle_epoch_offset_days(epoch):
    jd, fr = jday(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond / 1e6,
    )
    return jd + fr - _SGP4_EPOCH_JD_OFFSET


def _format_tle_epoch(epoch):
    seconds = epoch.hour * 3600 + epoch.minute * 60 + epoch.second + epoch.microsecond / 1e6
    return f"{epoch:%y%j}" + f"{seconds / Constants.SEC_PER_DAY:.8f}"[1:]


def _format_tle_ndot(value):
    field = f"{float(value): .8f}"
    return field.replace(" 0.", " .").replace("-0.", "-.")


def _norad_id_to_tle_str(norad_id):
    norad_id_int = int(norad_id)
    if norad_id_int > 99999:
        raise ValueError(f"TLE format cannot represent catalog number {norad_id_int}. Use OMM for catalog numbers above 99999.")
    return f"{norad_id_int:05d}"


def _mean_anomaly_to_true_anomaly(M, e):
    E = float(M)
    for _ in range(30):
        E -= (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return _wrap_rad(2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)))


def _rotation_1(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_3(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _state_from_classical_elements(elements):
    a = elements["a"]
    e = elements["e"]
    i = elements["i"]
    raan = elements["raan"]
    arg_perigee = elements["arg_perigee"]
    mean_anomaly = elements["mean_anomaly"]

    true_anomaly = _mean_anomaly_to_true_anomaly(mean_anomaly, e)
    semi_latus_rectum = a * (1.0 - e**2)
    radius_pf = semi_latus_rectum / (1.0 + e * np.cos(true_anomaly))
    r_pf = np.array([radius_pf * np.cos(true_anomaly), radius_pf * np.sin(true_anomaly), 0.0])
    v_pf = np.array(
        [
            -np.sqrt(Constants.GM_EARTH_TT / semi_latus_rectum) * np.sin(true_anomaly),
            np.sqrt(Constants.GM_EARTH_TT / semi_latus_rectum) * (e + np.cos(true_anomaly)),
            0.0,
        ]
    )

    rotation = _rotation_3(raan) @ _rotation_1(i) @ _rotation_3(arg_perigee)
    return rotation @ r_pf, rotation @ v_pf


def _orbit_to_classical_elements(orbit):
    return {
        "a": float(orbit.a),
        "e": float(orbit.e),
        "i": np.deg2rad(float(orbit.i)),
        "raan": np.deg2rad(float(orbit.RAAN)),
        "arg_perigee": np.deg2rad(float(orbit.p)),
        "mean_anomaly": np.deg2rad(float(orbit.M)),
    }


def _classical_to_equinoctial_solver_vars(elements):
    e = elements["e"]
    arg_perigee = elements["arg_perigee"]
    mean_anomaly = elements["mean_anomaly"]
    return np.array(
        [
            elements["a"],
            elements["i"],
            elements["raan"],
            e * np.cos(arg_perigee),
            e * np.sin(arg_perigee),
            arg_perigee + mean_anomaly,
        ],
        dtype=float,
    )


def _equinoctial_solver_vars_to_classical(values):
    a, i, raan, e_cos_argp, e_sin_argp, mean_longitude = values
    a = max(float(a), Constants.RADIUS_EARTH_EQUAT + 1.0)
    i = float(np.clip(i, 0.0, np.pi))
    e = float(np.hypot(e_cos_argp, e_sin_argp))
    e = float(np.clip(e, 0.0, 0.999999))
    if e < 1e-12:
        arg_perigee = 0.0
        mean_anomaly = _wrap_rad(mean_longitude)
    else:
        arg_perigee = _wrap_rad(np.arctan2(e_sin_argp, e_cos_argp))
        mean_anomaly = _wrap_rad(mean_longitude - arg_perigee)

    return {
        "a": a,
        "e": e,
        "i": i,
        "raan": _wrap_rad(raan),
        "arg_perigee": arg_perigee,
        "mean_anomaly": mean_anomaly,
    }


def _skyfield_state_from_sgp4_mean_elements(elements, epoch, satnum, bstar=0.0, ndot=0.0, nddot=0.0):
    mean_motion_rad_min = math.sqrt(Constants.GM_EARTH_TT / elements["a"] ** 3) * 60.0
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        int(satnum),
        _tle_epoch_offset_days(epoch),
        float(bstar),
        float(ndot),
        float(nddot),
        float(elements["e"]),
        float(elements["arg_perigee"]),
        float(elements["i"]),
        float(elements["mean_anomaly"]),
        mean_motion_rad_min,
        float(elements["raan"]),
    )
    timescale = _get_timescale()
    geocentric = EarthSatellite.from_satrec(satrec, timescale).at(timescale.from_datetime(epoch))
    return geocentric.position.m, geocentric.velocity.m_per_s


def _solve_sgp4_mean_elements_from_osculating(satellite):
    """
    =========================================================================================================
    Convert the user-specified osculating orbit to SGP4 mean elements.

    A TLE stores SGP4 mean elements, not an instantaneous Keplerian state. This fit chooses mean
    elements whose Skyfield/SGP4 state at the TLE epoch matches the osculating state implied by
    the configuration file.

    INPUTS:
        satellite: [Satellite]   Satellite carrying the osculating Orbit_specification

    OUTPUT:
        mean_elements: [dict]   {a, e, i, raan, arg_perigee, mean_anomaly} in SI units / radians
    =========================================================================================================
    """
    osculating_elements = _orbit_to_classical_elements(satellite.orbit_n)
    target_position, target_velocity = _state_from_classical_elements(osculating_elements)
    initial_guess = _classical_to_equinoctial_solver_vars(osculating_elements)
    epoch = satellite.orbit_n.dt_start
    satnum = int(satellite.satID)
    bstar = satellite.B_star if satellite.B_star is not None else 0.0
    ndot = satellite.meanMotion_1 if satellite.meanMotion_1 is not None else 0.0
    nddot = satellite.meanMotion_2 if satellite.meanMotion_2 is not None else 0.0

    def residual(values):
        mean_elements = _equinoctial_solver_vars_to_classical(values)
        position, velocity = _skyfield_state_from_sgp4_mean_elements(mean_elements, epoch, satnum, bstar=bstar, ndot=ndot, nddot=nddot)
        return np.concatenate(((position - target_position) / 100.0, (velocity - target_velocity) / 0.1))

    result = least_squares(
        residual,
        initial_guess,
        x_scale=np.array([1e4, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3]),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100,
    )
    mean_elements = _equinoctial_solver_vars_to_classical(result.x)

    if not result.success:
        warnings.warn(f"SGP4 mean-element fit did not report convergence for satellite {satellite.satName}: {result.message}")

    return mean_elements


def _generate_tle_from_mean_elements(satellite, mean_elements):
    sat_id_str = _norad_id_to_tle_str(satellite.satID)
    classification = getattr(satellite, "classification", "U")
    classification = classification[0].upper() if classification and classification[0].upper() in ["U", "C", "S"] else "U"
    launch_no_year = getattr(satellite, "launch_no_year", getattr(satellite, "lauch_Noyear", 1))
    launch_piece = getattr(satellite, "launch_piece", getattr(satellite, "lauch_piece", "A"))
    element_set_no = satellite.element_set_no if satellite.element_set_no is not None else 1
    ndot = satellite.meanMotion_1 if satellite.meanMotion_1 is not None else 0.0
    nddot = satellite.meanMotion_2 if satellite.meanMotion_2 is not None else 0.0
    bstar = satellite.B_star if satellite.B_star is not None else 0.0

    line0_str = satellite.satName
    line_no_1_str = "1"
    int_designator_year = f"{satellite.launch_year % 100:02d}"
    int_designator_no = f"{int(launch_no_year):03d}"
    int_designator_piece = f"{launch_piece:<3}"[:3]
    epoch_str = _format_tle_epoch(satellite.orbit_n.dt_start)
    ndot_str = _format_tle_ndot(ndot)
    nddot_str = ReadWriteSatData._str2tle_format(nddot)
    bstar_str = ReadWriteSatData._str2tle_format(bstar)
    ephemeris_str = "0"
    element_set_no_str = f"{int(element_set_no):03d}"

    line1_content = (
        f"{line_no_1_str} {sat_id_str}{classification} {int_designator_year}{int_designator_no}{int_designator_piece} "
        f"{epoch_str} {ndot_str} {nddot_str} {bstar_str} {ephemeris_str}  {element_set_no_str}"
    )
    checksum1 = ReadWriteSatData._calc_checksum([line1_content])
    line1_str = f"{line1_content}{checksum1}"

    line_no_2_str = "2"
    inclination_str = f"{np.degrees(mean_elements['i']):8.4f}"
    raan_str = f"{_wrap_deg(mean_elements['raan']):8.4f}"
    eccentricity_tle = int(np.clip(round(mean_elements["e"] * 1e7), 0, 9999999))
    eccentricity_str = f"{eccentricity_tle:07d}"
    arg_perigee_str = f"{_wrap_deg(mean_elements['arg_perigee']):8.4f}"
    mean_anomaly_str = f"{_wrap_deg(mean_elements['mean_anomaly']):8.4f}"
    mean_motion_rev_day = math.sqrt(Constants.GM_EARTH_TT / mean_elements["a"] ** 3) * Constants.SEC_PER_DAY / (2.0 * math.pi)
    mean_motion_str = f"{mean_motion_rev_day:11.8f}"
    rev_str = f"{0:05d}"

    line2_content = (
        f"{line_no_2_str} {sat_id_str} {inclination_str} {raan_str} {eccentricity_str} "
        f"{arg_perigee_str} {mean_anomaly_str} {mean_motion_str}{rev_str}"
    )
    checksum2 = ReadWriteSatData._calc_checksum([line2_content])
    line2_str = f"{line2_content}{checksum2}"

    return f"{line0_str}\n{line1_str}\n{line2_str}"


class ReadWriteSatData:
    def __init__(self):
        pass

    @staticmethod
    def generate_tle_data_string(satellite) -> str:
        """
        =========================================================================================================
        Generate TLE based on satellite orbit_n data.

        INPUTS:
            satellite: [Satellite]   Satellite with an orbit_n attribute set

        OUTPUT:
            tle_string: [str | None]   Three-line TLE string, or None if information is missing
        =========================================================================================================
        """
        if satellite.orbit_n is None:
            warnings.warn("Orbit was not specified when this Satellite was created. Insufficient information to generate a TLE.")
            return None
        if satellite.orbit_n.epoch_str is None:
            warnings.warn("Epoch was not specified when this Orbit was created. Insufficient information to generate a TLE.")
            return None

        mean_elements = _solve_sgp4_mean_elements_from_osculating(satellite)

        return _generate_tle_from_mean_elements(satellite, mean_elements)

    @staticmethod
    def read_tle_data(tle_file_path) -> list[namedtuple]:
        """
        =========================================================================================================
        Read TLE data from a file and return a list of SatStruct namedtuples.

        INPUTS:
            tle_file_path: [str]   Path to the TLE file

        OUTPUT:
            satList: [list[namedtuple] | None]   One SatStruct per satellite in the file, or None if missing
        =========================================================================================================
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
            if readNextSat:
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
                                warnings.warn("Checksum error in TLE data line 1")
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
                            warnings.warn("Checksum error in TLE data line 2")
                        if not (satID_1_str == satID_2_str):
                            warnings.warn("Satellite ID mismatch in line 1 and line 2")
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
            if not readNextSat:
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
        =========================================================================================================
        Download TLE data from a website and read it.

        INPUTS:
            path: [str]   Local destination for the downloaded TLE file

        OUTPUT:
            success: [bool]   True if the download and parse succeeded
        =========================================================================================================
        """
        # Download TLE data from a website

    @staticmethod
    def __tle_to_ballistic_coef(balistic_coeff_str) -> float:
        """
        =========================================================================================================
        Convert the first derivative of mean motion (TLE field) to physical units.

        INPUTS:
            balistic_coeff_str: [str]   TLE first-derivative-of-mean-motion field

        OUTPUT:
            ndot: [float]   First derivative of mean motion
        =========================================================================================================
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
        =========================================================================================================
        Convert the second derivative of mean motion (TLE B*-style field) to physical units.

        INPUTS:
            secDerMeanM_str: [str]   TLE field encoded as <sign>XXXXX<sign>X

        OUTPUT:
            value: [float]   Decoded floating-point value
        =========================================================================================================
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
        =========================================================================================================
        Convert TLE epoch (year and fractional day-of-year) to a UTC-aware datetime.

        INPUTS:
            year: [int]     4-digit year
            days: [float]   Fractional day-of-year [DDD.DDDDDDDD]

        OUTPUT:
            result: [datetime]   UTC-aware datetime corresponding to (year, days)
        =========================================================================================================
        """
        # Get the start of the year
        start_of_year = datetime(year, 1, 1)

        # Convert the fractional day to a timedelta
        day_fraction, whole_day = days % 1, int(days)
        day_time = timedelta(days=day_fraction)

        # Calculate the datetime by adding the whole days and fractional day time
        result_datetime = start_of_year + timedelta(days=whole_day - 1) + day_time
        result_datetime = result_datetime.replace(tzinfo=datetime_module.UTC)

        return result_datetime

    @staticmethod
    def _calc_checksum(stringArray) -> int:
        """
        =========================================================================================================
        Calculate TLE checksum (sum of digits + 1 per "-", mod 10).

        INPUTS:
            stringArray: [list[str]]   TLE line(s) to checksum (excluding the checksum digit itself)

        OUTPUT:
            checksum: [int]   Single-digit checksum
        =========================================================================================================
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
        =========================================================================================================
        Convert "Second derivative of mean motion" or "B*, the drag term" from its numeric value
        to the TLE-encoded format `<sign>XXXXX<sign>X` (e.g. 1.2345e-4 => ' 12345-3).

        INPUTS:
            value: [float]   Numeric value to encode

        OUTPUT:
            tle_sub_str: [str]   Encoded TLE substring
        =========================================================================================================
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
