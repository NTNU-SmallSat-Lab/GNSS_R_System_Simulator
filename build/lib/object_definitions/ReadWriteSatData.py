from collections import namedtuple
import datetime  # Import the module itself first
# Monkey patch for Python < 3.11 compatibility
if not hasattr(datetime, "UTC"):
    from datetime import timezone
    datetime.UTC = timezone.utc
from datetime import datetime, timezone, timedelta
import os
import re

# import fortranformat as ff


# ---------------------------------------------------------------------------------------------------------- #
#                                            Module constants                                                #
# ---------------------------------------------------------------------------------------------------------- #
SPUTNIK_LAUNCHDATE = datetime(1957, 10, 4, tzinfo=timezone.utc)  # Pivot for 2-digit TLE year decoding (Sputnik 1, first catalogued satellite).
_TLE_MAX_NORAD_ID = 99999  # Largest catalog number representable in the 5-character TLE field.


# ---------------------------------------------------------------------------------------------------------- #
#                                       NORAD-ID validation helpers                                          #
# ---------------------------------------------------------------------------------------------------------- #
def _check_tle_id_is_5_digit_numeric(id_field: str, line_name: str) -> None:
    """Ensure a TLE catalog field contains only numeric characters."""
    id_clean = id_field.strip()
    if not id_clean.isdigit():
        raise ValueError(
            f"read_tle_data() received a non-numeric catalog field in {line_name} "
            f"('{id_field}'). This parser only supports standard numeric TLE "
            "catalog IDs in the 5-character field."
        )


def _check_tle_id_is_not_alpha5_like(id_field: str, line_name: str) -> None:
    """Reject Alpha-5-style catalog IDs (e.g. 'A0001')."""
    id_clean = id_field.strip().upper()
    if len(id_clean) == 5 and id_clean[0].isalpha() and id_clean[1:].isdigit():
        raise ValueError(
            f"read_tle_data() received an Alpha-5-like catalog field in {line_name} "
            f"('{id_field}'). This parser only supports classic numeric TLE IDs "
            "up to 99999. Use OMM (CSV/JSON/XML) for newer catalog schemes."
        )


def _check_tle_id_no_overflow(line: str, line_name: str) -> None:
    """Detect a 6+ digit catalog number forced into the 5-character TLE field."""
    if len(line) > 7 and line[7].isdigit():
        raise ValueError(
            f"read_tle_data() detected a catalog-ID overflow in {line_name}. "
            "This is likely a 6+ digit object ID forced into the 5-character "
            "TLE field. Use OMM (CSV/JSON/XML) for these objects."
        )


def _norad_id_to_tle_str(norad_id) -> str:
    """
    Format a NORAD catalog number as a zero-padded 5-character string for TLE output.
    Raises if the number does not fit in the fixed-width TLE field.
    """
    norad_id_int = int(norad_id)
    if norad_id_int > _TLE_MAX_NORAD_ID:
        raise ValueError(
            f"TLE format cannot represent catalog number {norad_id_int}. "
            f"The TLE 5-character field supports integers 0-{_TLE_MAX_NORAD_ID}. "
            "Objects with 6+ digit catalog numbers (assigned from ~July 2026) require "
            "the OMM format (CSV/JSON/XML) instead. "
            "See: https://celestrak.org/NORAD/elements/gp-faq.php"
        )
    return f"{norad_id_int:05d}"


def _parse_tle_year(year_str: str) -> int:
    """
    Convert a 2-digit TLE year to a full 4-digit year using Sputnik (1957) as the pivot.
    Years >= 57 map to 19YY (1957..1999); years < 57 map to 20YY (2000..2056).
    """
    year = int(year_str)
    return 1900 + year if year >= SPUTNIK_LAUNCHDATE.year % 100 else 2000 + year


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
        satIDStr = _norad_id_to_tle_str(satellite.satID)  # NORAD ID formatted as zero-padded 5-char field; raises if > 99999.
        classStr = satellite.classification  # The satellite is unclassified [Can be U: Unclassified, C: Classified, S: Secret]
        IntD_LYear = f"{satellite.launch_year % 100:02d}"  # International Designator (last two digits of launch year) COSPAR ID
        IntD_LNo = f"{int(satellite.launch_no_year):03d}"  # International Designator (launch no. of the year)         COSPAR ID
        IntD_POL = f"{satellite.launch_piece:<3}"[:3]  # International Designator (piece of the launch) padded to fixed-width 3 chars (TLE col 15-17).
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

        line1_str = f"{line_no_1_str} {satIDStr}{classStr} {IntD_LYear}{IntD_LNo}{IntD_POL} {epochStr} {balisticCoefStr} {meanMotionStr_2} {b_star_str} {ephemeris_str}  {elem_st_no_str}{checksum1}"

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
    def read_tle_data(tle_file_path) -> list:
        """
        Read TLE data from a file and return a list of SatStruct namedtuples.

        Auto-detects 2LE (no title) vs 3LE (title + line 1 + line 2) format,
        validates each block (header characters, catalog-ID consistency,
        and checksums) before parsing, and raises ``ValueError`` on malformed
        input. Stray lines that do not fit the expected pattern are skipped
        with a warning and the in-progress block is discarded.
        """
        # check if the file exists
        if not os.path.exists(tle_file_path):
            print(f"File {tle_file_path} does not exist")
            return None

        SatStruct = namedtuple(
            "SatStruct",
            [
                "satName",
                "satCatalogNumber",
                "classification",
                "launchYear",
                "launchNoInYear",
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

        with open(tle_file_path) as file:
            tle_lines = file.read().splitlines()

        # Strip blank lines
        tle_lines = [ln for ln in tle_lines if ln.strip() != ""]

        n_lines = len(tle_lines)
        if n_lines < 2:
            raise ValueError(
                f"read_tle_data() received a TLE with {n_lines} lines ({tle_file_path}). "
                "A valid TLE must have at least two lines."
            )

        n_line1 = sum(1 for ln in tle_lines if ln.startswith("1"))
        n_line2 = sum(1 for ln in tle_lines if ln.startswith("2"))
        n_other = n_lines - n_line1 - n_line2

        if n_line1 != n_line2:
            raise ValueError(
                f"read_tle_data() found {n_line1} lines starting with '1' and {n_line2} "
                f"lines starting with '2' in {tle_file_path}. The counts must be equal."
            )
        if n_line1 == 0:
            raise ValueError(f"read_tle_data() found no TLE data lines in {tle_file_path}.")

        # 3LE if there are at least as many "other" lines as TLE blocks; otherwise 2LE.
        tle_type = 3 if n_other >= n_line1 else 2
        reading_order = ["title", "line1", "line2"] if tle_type == 3 else ["line1", "line2"]

        block = {"title": "", "line1": "", "line2": ""}
        order_idx = 0
        sat_list = []

        for line_no, line in enumerate(tle_lines, start=1):
            expected = reading_order[order_idx]

            if expected == "title":
                if line.startswith("1") or line.startswith("2"):
                    raise ValueError(
                        f"read_tle_data() expected a title line at line {line_no} "
                        f"of {tle_file_path} but got a TLE data line."
                    )
                block["title"] = line
            elif expected == "line1":
                if not line.startswith("1"):
                    print(
                        f"WARNING: read_tle_data() skipping unexpected line {line_no} "
                        f"of {tle_file_path}: {line.strip()!r}. The in-progress TLE block is discarded."
                    )
                    block = {"title": "", "line1": "", "line2": ""}
                    order_idx = 0
                    continue
                _check_tle_id_no_overflow(line, "line1")
                block["line1"] = line
            elif expected == "line2":
                if not line.startswith("2"):
                    print(
                        f"WARNING: read_tle_data() skipping unexpected line {line_no} "
                        f"of {tle_file_path}: {line.strip()!r}. The in-progress TLE block is discarded."
                    )
                    block = {"title": "", "line1": "", "line2": ""}
                    order_idx = 0
                    continue
                _check_tle_id_no_overflow(line, "line2")
                block["line2"] = line

            order_idx += 1

            if order_idx == len(reading_order):
                sat_list.append(
                    ReadWriteSatData._parse_tle_block(
                        block["title"], block["line1"], block["line2"],
                        SatStruct, tle_file_path,
                    )
                )
                block = {"title": "", "line1": "", "line2": ""}
                order_idx = 0

        return sat_list

    @staticmethod
    def _parse_tle_block(title_line, line_1, line_2, SatStruct, tle_file_path):
        """Parse one validated TLE block into a SatStruct (title may be empty for 2LE)."""
        # ---- Line 1 ----
        if line_1[0] != "1":
            raise ValueError(
                f"Invalid TLE data format in line 1 ({tle_file_path}): "
                f"expected leading '1', got '{line_1[0]}'."
            )
        satID_1_str = line_1[2:7]
        _check_tle_id_is_not_alpha5_like(satID_1_str, "line1")
        _check_tle_id_is_5_digit_numeric(satID_1_str, "line1")
        classification_str = line_1[7:8]
        launchYear_str = line_1[9:11]
        launchNoYear_str = line_1[11:14]
        pieceOfLaunch_str = line_1[14:17].strip()
        epochYear_str = line_1[18:20]
        epochDay_str = line_1[20:32]
        firstDerMean_str = line_1[33:43]
        secondDerMean_str = line_1[44:52]
        dragTerm_str = line_1[53:61]
        ephType_str = line_1[62:63]
        elSetNo_str = line_1[64:68]
        if len(line_1) > 68:
            checksum_str1 = line_1[68:69]
            calc_checksum1 = ReadWriteSatData._calc_checksum([line_1[:68]])
            if calc_checksum1 != int(checksum_str1):
                raise ValueError(
                    f"Checksum mismatch in TLE line 1 ({tle_file_path}): "
                    f"computed {calc_checksum1}, file has {checksum_str1}."
                )

        # ---- Line 2 ----
        if line_2[0] != "2":
            raise ValueError(
                f"Invalid TLE data format in line 2 ({tle_file_path}): "
                f"expected leading '2', got '{line_2[0]}'."
            )
        satID_2_str = line_2[2:7]
        _check_tle_id_is_not_alpha5_like(satID_2_str, "line2")
        _check_tle_id_is_5_digit_numeric(satID_2_str, "line2")
        if satID_1_str.strip() != satID_2_str.strip():
            raise ValueError(
                f"Catalog-ID mismatch between line 1 ('{satID_1_str}') and line 2 "
                f"('{satID_2_str}') in {tle_file_path}."
            )
        incl_str = line_2[8:16]
        raan_str = line_2[17:25]
        ecc_str = line_2[26:33]
        argPeri_str = line_2[34:42]
        meanAno_str = line_2[43:51]
        meanMotion_str = line_2[52:63]
        revNoatEpoch_str = line_2[63:68]
        checksum_str2 = line_2[68:69]
        calc_checksum2 = ReadWriteSatData._calc_checksum([line_2[:68]])
        if calc_checksum2 != int(checksum_str2):
            raise ValueError(
                f"Checksum mismatch in TLE line 2 ({tle_file_path}): "
                f"computed {calc_checksum2}, file has {checksum_str2}."
            )

        satName = title_line.strip()
        satCatalogNumber = int(satID_1_str)
        classification = classification_str
        launchYear = _parse_tle_year(launchYear_str)
        launchNoInYear = int(launchNoYear_str.strip())
        launchPiece = pieceOfLaunch_str.strip()
        epochYear = _parse_tle_year(epochYear_str)
        epochDay = float(epochDay_str.strip())
        epoch = ReadWriteSatData.__get_datetime(epochYear, epochDay)
        firstDerMeanMotion = ReadWriteSatData.__tle_to_ballistic_coef(firstDerMean_str)
        secDerivMeanMotion = ReadWriteSatData.__tle_to_MeanM_Bstar(secondDerMean_str)
        B_star = ReadWriteSatData.__tle_to_MeanM_Bstar(dragTerm_str)
        EphType = int(ephType_str.strip())
        ElementSetNo = int(elSetNo_str.strip())

        incl = float(incl_str.strip())
        raan = float(raan_str.strip())
        ecc = float(f"0.{ecc_str.strip()}")
        argPeri = float(argPeri_str.strip())
        meanAno = float(meanAno_str.strip())
        meanMotion = float(meanMotion_str.strip())
        revNoatEpoch = int(revNoatEpoch_str.strip())

        return SatStruct(
            satName,
            satCatalogNumber,
            classification,
            launchYear,
            launchNoInYear,
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
