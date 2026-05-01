from datetime import UTC, datetime
import math
import os

from object_definitions.Constants_def import Constants
from object_definitions.Orbit_def import Orbit_specification
from object_definitions.ReadWriteSatData import ReadWriteSatData
from object_definitions.Reflection_def import Reflection
from object_definitions.SatConstellation_def import SatelliteConstellation
from object_definitions.Satellite_def import Satellite


def testObjReflection() -> bool:
    passed = 1
    spec_non = Reflection()
    spec_elli = Reflection("elliptical")
    spec_sph = Reflection("spherical")
    spec_typo = Reflection("sperical")  # Intentional typo for testing

    if not (isinstance(spec_non, Reflection) and spec_non.earth_model == "spherical"):
        passed = 0
    if not (isinstance(spec_elli, Reflection) and spec_elli.earth_model == "elliptical"):
        passed = 0
    if not (isinstance(spec_sph, Reflection) and spec_sph.earth_model == "spherical"):
        passed = 0
    if not (isinstance(spec_typo, Reflection) and not hasattr(spec_typo, "earth_model")):
        passed = 0
    return passed


def testObjOrbit() -> bool:
    passed = 1
    start_time = datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC)  # Start time in UTC (YYYY:MM:DD HH:mm:ss)
    apogee = 421e3 + Constants.R_EARTH_MEAN  # Apogee in [m]

    orbit_noEpoch = Orbit_specification(
        51.6374,  # inclination         [°]
        0.0009376,  # eccentricity        [-]
        1.5935,  # RAAN                [°]
        121.8064,  # argument of perigee [°]
        341.7727,  # Mean anomaly        [°]
        apogee,  # apogee              [m]
    )

    orbit_withEpoch = Orbit_specification(
        51.6374,  # inclination         [°]
        0.0009376,  # eccentricity        [-]
        1.5935,  # RAAN                [°]
        121.8064,  # argument of perigee [°]
        341.7727,  # Mean anomaly        [°]
        apogee,  # apogee              [m]
        start_time,  # epoch               [YYYY:MM:DD HH:mm:ss UTC]
    )
    if not (isinstance(orbit_noEpoch, Orbit_specification) and orbit_noEpoch.epoch_str == None):
        passed = 0
    if not (isinstance(orbit_withEpoch, Orbit_specification) and orbit_withEpoch.epoch_str == "24305.21246528"):
        passed = 0
    return passed


def testObjSatellite() -> bool:
    passed = 1
    start_time = datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC)
    apogee = 421e3 + Constants.R_EARTH_MEAN
    orbit_noEpoch = Orbit_specification(
        51.6374,  # inclination          [°]
        0.0009376,  # eccentricity         [-]
        1.5935,  # RAAN                 [°]
        121.8064,  # argument of perigee  [°]
        341.7727,  # Mean anomaly         [°]
        apogee,  # apogee               [m]
    )

    sat = Satellite(1)

    orbit_withEpoch = Orbit_specification(
        51.6374,  # inclination          [°]
        0.0009376,  # eccentricity         [-]
        1.5935,  # RAAN                 [°]
        121.8064,  # argument of perigee  [°]
        341.7727,  # Mean anomaly         [°]
        apogee,  # apogee               [m]
        start_time,  # epoch                [YYYY:MM:DD HH:mm:ss UTC]
    )

    sat_noOrbit = Satellite(1)
    sat_orbitNoEpoch = Satellite(2, orbit_noEpoch)
    sat_orbitWithEpoch = Satellite(3, orbit_withEpoch)
    satellite_newOrbit = Satellite(
        4,  # Satellite ID        [number]
        [
            51.6374,  # inclination         [°]
            0.0009376,  # eccentricity        [-]
            1.5935,  # RAAN                [°]
            121.8064,  # argument of perigee [°]
            341.7727,  # Mean anomaly        [°]
            apogee,  # apogee              [m]
            start_time,  # epoch               [YYYY:MM:DD HH:mm:ss UTC]
        ],
    )
    satellite_newOrbit2 = Satellite(
        5,
        [
            51.6374,  # inclination         [°]
            0.0009376,  # eccentricity        [-]
            1.5935,  # RAAN                [°]
            121.8064,  # argument of perigee [°]
            341.7727,  # Mean anomaly        [°]
            apogee,  # apogee              [m]
        ],
        satName="newName",
    )
    satellite_noOrbit_newName = Satellite(6, satName="newName")

    if not (isinstance(sat_noOrbit, Satellite) and sat_noOrbit.orbit_n == None):
        passed = 0
    if not (isinstance(sat_orbitNoEpoch, Satellite) and sat_orbitNoEpoch.orbit_n == orbit_noEpoch and sat_orbitNoEpoch.tle_exists() == False):
        passed = 0
    if not (isinstance(sat_orbitWithEpoch, Satellite) and sat_orbitWithEpoch.orbit_n == orbit_withEpoch and sat_orbitWithEpoch.tle_exists() == True):
        passed = 0
    if not (isinstance(satellite_newOrbit, Satellite) and satellite_newOrbit.orbit_n.epoch_str == "24305.21246528" and satellite_newOrbit.tle_exists() == True):
        passed = 0
    # Generate TLE file  for satellite_newOrbit
    #    satellite_newOrbit.write_tle_file()
    if not (satellite_newOrbit.tle_exists() == True):
        passed = 0

    if not (isinstance(satellite_newOrbit2, Satellite) and satellite_newOrbit2.orbit_n.epoch_str == None and satellite_newOrbit2.tle_exists() == False):
        passed = 0

    # Set the classification of the satellite
    satellite_newOrbit2.set_classification("C")
    if not (satellite_newOrbit2.classification == "C"):
        passed = 0

    # Calculate the TLE for the satellite | This should not work as the epoch is not set
    satellite_newOrbit2.write_tle_file()
    if not (satellite_newOrbit2.tle_exists() == False):
        passed = 0

    if not (isinstance(satellite_noOrbit_newName, Satellite) and satellite_noOrbit_newName.satName == "newName"):
        passed

    # Initialize Satellite with TLE file
    sat_tle = Satellite(7, tle_file_path=os.path.join(Constants.tle_folderpath, "ISS.tle"))
    sat_tle.write_tle_file()
    return passed


def testReadWriteData() -> bool:
    passed = 1
    # Read GALILEO TLE data
    filepath = os.path.join(Constants.tle_folderpath, "GALILEO.tle")
    satList = ReadWriteSatData.read_tle_data(filepath)

    if not (isinstance(satList, list) and len(satList) == 31):
        passed = 0
    if satList[10].argOfPerigee != 12.6669:
        passed = 0

    filepath2 = os.path.join(Constants.tle_folderpath, "ISS.tle")
    issList = ReadWriteSatData.read_tle_data(filepath2)

    if not (isinstance(issList, list) and len(issList) == 1):
        passed = 0
    # https://www.heavens-above.com/orbit_n.aspx?satid=25544
    if (
        issList[0].satCatalogNumber != 25544  # Satellite Catalog Number
        and issList[0].inclination != 51.638  # Inclination
        and issList[0].eccentricity != 0.0008444  # Eccentricity
        and issList[0].raan != 301.8966  # Right Ascension of Ascending Node
        and issList[0].argOfPerigee != 165.9795  # Argument of Perigee
        and issList[0].meanAnomaly != 194.1431  # Mean Anomaly at Epoch
        and issList[0].meanMotion != 15.51387213  # Mean Motion [rev/day]
        and issList[0].epoch != "24304.21246528"
    ):
        passed = 0

    # Calculate apogee hight for ISS
    semi_major_axis = (Constants.GM_EARTH_TT / (issList[0].meanMotion * 2 * math.pi / Constants.SEC_PER_DAY) ** 2) ** (1 / 3)

    apogee_above_earth = semi_major_axis * (1 + issList[0].eccentricity) - Constants.R_EARTH_MEAN
    perigee_above_earth = semi_major_axis * (1 - issList[0].eccentricity) - Constants.R_EARTH_MEAN

    if not (apogee_above_earth == 425.5460901848506e3 and perigee_above_earth == 414.07776699984353e3):
        passed = 0

    return passed


def testFreeFlyerConnection() -> bool:
    """
    =========================================================================================================
    Test connection to FreeFlyer AGI.
    =========================================================================================================
    """
    passed = 1
    show_FF_GUI = True
    GNSS_Constellation = "GPS"
    sat_1 = Satellite(
        4,  # Satellite ID        [number]
        [
            51.6374,  # inclination         [°]
            0.0009376,  # eccentricity        [-]
            1.5935,  # RAAN                [°]
            121.8064,  # argument of perigee [°]
            341.7727,  # Mean anomaly        [°]
            421e3 + Constants.R_EARTH_MEAN,  # apogee              [m]
            datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC),  # epoch               [YYYY:MM:DD HH:mm:ss UTC]
        ],
    )
    sat_1.write_tle_file()


def testSatConstelation() -> bool:
    passed = 1
    # Initialize with an array of Satellite objects
    sat1 = Satellite(
        1,
        [
            51.6374,
            0.0009376,
            1.5935,
            121.8064,
            341.7727,
            421e3 + Constants.R_EARTH_MEAN,
            datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC),
        ],
    )
    sat2 = Satellite(
        2,
        [
            51.6374,
            0.0009376,
            1.5935,
            121.8064,
            341.7727,
            441e3 + Constants.R_EARTH_MEAN,
            datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC),
        ],
    )
    sat3 = Satellite(
        3,
        [
            51.6374,
            0.0009376,
            1.5935,
            121.8064,
            341.7727,
            461e3 + Constants.R_EARTH_MEAN,
            datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC),
        ],
    )
    sat4 = Satellite(
        4,
        [
            51.6374,
            0.0009376,
            1.5935,
            121.8064,
            341.7727,
            481e3 + Constants.R_EARTH_MEAN,
            datetime(2024, 10, 31, 5, 5, 57, tzinfo=UTC),
        ],
    )

    satConfiguration_native = SatelliteConstellation([sat1, sat2, sat3, sat4])
    if not (isinstance(satConfiguration_native, SatelliteConstellation) and satConfiguration_native.no_satellites == 4):
        passed = 0

    # Initialize with a TLE file
    satConfiguration_tle = SatelliteConstellation(tle_file=os.path.join(Constants.tle_folderpath, "GALILEO.tle"))
    return passed


## ----------------------------------------------------- ##
## ------------------- Test Function ------------------- ##
## ----------------------------------------------------- ##
def testClassOrFunction(testFunction) -> bool:
    try:
        success = testFunction()
        if not success:
            print(f"Test failed: -- {testFunction.__name__} --")
        return success
    except Exception as e:
        print(f"Error: {e} in {testFunction.__name__}")
        return False


if __name__ == "__main__":
    passed = 1

    passed = passed & testClassOrFunction(testSatConstelation)

    passed = passed & testClassOrFunction(testObjReflection)
    passed = passed & testClassOrFunction(testObjOrbit)
    passed = passed & testClassOrFunction(testObjSatellite)
    passed = passed & testClassOrFunction(testReadWriteData)
    passed = passed & testClassOrFunction(testFreeFlyerConnection)
    passed = passed & testClassOrFunction(testGNSSConstellation)

    if passed:
        print("All tests passed")
