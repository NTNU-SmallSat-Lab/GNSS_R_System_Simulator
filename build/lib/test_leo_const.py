import re


# ==========================================================
# Mock helper class (your project already has this)
# ==========================================================
class ReadWriteSatData:

    @staticmethod
    def _str2tle_format(value):
        """Convert float to TLE scientific format."""
        if value == 0:
            return " 00000-0"

        s = f"{value:.5e}"
        mantissa, exp = s.split("e")
        mantissa = mantissa.replace(".", "")[:5]
        exp = int(exp)
        return f" {mantissa}{exp:+1d}"

    @staticmethod
    def _calc_checksum(parts):
        line = "".join(parts)
        checksum = 0
        for c in line:
            if c.isdigit():
                checksum += int(c)
            if c == "-":
                checksum += 1
        return checksum % 10


def _norad_id_to_tle_str(norad_id):
    return f"{norad_id:05d}"


# ==========================================================
# Orbit object (satellite.orbit_n)
# Creating orbit to model the CELESTE IOD mission
# Quasi-polar, sun-synchronous orbit with an inclination of ** degrees
# IOD-1 tle:
# 1 68408U 26065A   26118.19090573  .00003320  00000-0  16421-3 0  9992
# 2 68408  97.4089 359.7482 0017212 172.6896 187.4592 15.18387302  4676

# ==========================================================
class Orbit:
    def __init__(self):
        self.epoch_str = "24123.50000000"  # YYDDD.DDDDDDDD

        self.i = 97.4       # inclination [deg]
        self.RAAN = 45.0    # RAAN [deg]
        self.e = 0.0001     # eccentricity
        self.p = 10.0       # argument of perigee
        self.M = 20.0       # mean anomaly
        self.n = 14.8       # mean motion [rev/day]


# ==========================================================
# Satellite object
# ==========================================================
class Satellite:
    def __init__(self):

        # ---- metadata ----
        self.satName = "TEST-SAT-1"
        self.satID = 00001
        self.classification = "U"

        self.launch_year = 2025
        self.launch_no_year = 1
        self.launch_piece = "A"

        # ---- drag / dynamics ----
        self.meanMotion_1 = 0.0
        self.meanMotion_2 = 0.0
        self.B_star = 0.00001
        self.element_set_no = 1

        # ---- orbit ----
        self.orbit_n = Orbit()


# ==========================================================
# YOUR FUNCTION
# ==========================================================
def generate_tle_data_string(satellite) -> str:

    if satellite.orbit_n is None:
        print("No orbit!")
        return None

    if satellite.orbit_n.epoch_str is None:
        print("No epoch!")
        return None

    balistic_coeff = satellite.meanMotion_1 or 0
    meanMotionSecDer = satellite.meanMotion_2 or 0
    B_star = satellite.B_star or 0
    element_set_no = satellite.element_set_no or 1

    # ---------------- Line 0 ----------------
    line0_str = satellite.satName

    # ---------------- Line 1 ----------------
    line_no_1_str = "1"
    satIDStr = _norad_id_to_tle_str(satellite.satID)
    classStr = satellite.classification

    IntD_LYear = f"{satellite.launch_year % 100:02d}"
    IntD_LNo = f"{int(satellite.launch_no_year):03d}"
    IntD_POL = f"{satellite.launch_piece:<3}"[:3]

    epochStr = satellite.orbit_n.epoch_str

    balisticCoefStr = re.sub(r"-0", "-", f"{balistic_coeff:10.8f}")

    meanMotionStr_2 = ReadWriteSatData._str2tle_format(meanMotionSecDer)
    b_star_str = ReadWriteSatData._str2tle_format(B_star)

    ephemeris_str = "0"
    elem_st_no_str = f"{element_set_no:03d}"

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

    line1_str = (
        f"{line_no_1_str} {satIDStr}{classStr} "
        f"{IntD_LYear}{IntD_LNo}{IntD_POL} "
        f"{epochStr} {balisticCoefStr} "
        f"{meanMotionStr_2} {b_star_str} "
        f"{ephemeris_str}  {elem_st_no_str}{checksum1}"
    )

    # ---------------- Line 2 ----------------
    line_no_2_str = "2"

    i_str = f"{satellite.orbit_n.i:8.4f}"
    raan_str = f"{satellite.orbit_n.RAAN:8.4f}"
    eccen_str = f"{int(satellite.orbit_n.e * 1e7):07d}"
    per_str = f"{satellite.orbit_n.p:8.4f}"
    mean_anom_str = f"{satellite.orbit_n.M:8.4f}"
    mean_Mot_str = f"{satellite.orbit_n.n:11.8f}"
    rev_str = f"{0:05d}"

    checksum2 = ReadWriteSatData._calc_checksum(
        [
            line_no_2_str,
            satIDStr,
            i_str,
            raan_str,
            eccen_str,
            per_str,
            mean_anom_str,
            mean_Mot_str,
            rev_str,
        ]
    )

    line2_str = (
        f"{line_no_2_str} {satIDStr} {i_str} {raan_str} "
        f"{eccen_str} {per_str} {mean_anom_str} "
        f"{mean_Mot_str}{rev_str}{checksum2}"
    )

    return f"{line0_str}\n{line1_str}\n{line2_str}"


# ==========================================================
# RUN TEST
# ==========================================================
if __name__ == "__main__":

    sat = Satellite()

    tle = generate_tle_data_string(sat)

    print("\nGenerated TLE:\n")
    print(tle)