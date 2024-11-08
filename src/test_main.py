from datetime import datetime, timezone
import os
from object_definitions.Reflection_def import Reflection
from object_definitions.Orbit_def      import Orbit
from object_definitions.Constants_def  import Constants
from object_definitions.Satellite_def  import Satellite
from FF_sockets.FF_TLE_to_Eph          import FF_TLE_to_Eph

# Test Object "Reflection"


def testObjReflection() -> bool:
    passed = 1
    spec_non   = Reflection()
    spec_elli  = Reflection('elliptical')
    spec_sph   = Reflection('spherical')
    spec_typo  = Reflection('sperical')    # Intentional typo for testing

    if not (isinstance(spec_non, Reflection) and spec_non.earth_model == 'spherical'):
        passed = 0
    if not (isinstance(spec_elli, Reflection) and spec_elli.earth_model == 'elliptical'):
        passed = 0
    if not (isinstance(spec_sph, Reflection) and spec_sph.earth_model == 'spherical'):
        passed = 0
    if not (isinstance(spec_typo, Reflection) and not hasattr(spec_typo, 'earth_model')):
        passed = 0
    return passed
def testObjOrbit() -> bool:
    passed     = 1
    start_time = datetime(2024, 10, 31, 5, 5, 57, tzinfo=timezone.utc)  # Start time in UTC (YYYY:MM:DD HH:mm:ss)
    apogee     = 421e3 + Constants.R_earth                                                # Apogee in [m]

    orbit_noEpoch = Orbit(
        51.6374,                   # inclination         [°]
        0.0009376,                 # eccentricity        [-]
        1.5935,                    # RAAN                [°]
        121.8064,                  # argument of perigee [°]
        341.7727,                  # Mean anomaly        [°]
        apogee                     # apogee              [m]
    )

    orbit_withEpoch = Orbit(
        51.6374,                   # inclination         [°]
        0.0009376,                 # eccentricity        [-]
        1.5935,                    # RAAN                [°]
        121.8064,                  # argument of perigee [°]
        341.7727,                  # Mean anomaly        [°]
        apogee,                    # apogee              [m]
        start_time                 # epoch               [YYYY:MM:DD HH:mm:ss UTC]
    )
    if not (isinstance(orbit_noEpoch, Orbit) and orbit_noEpoch.epoch == None):
        passed = 0
    if not (isinstance(orbit_withEpoch, Orbit) and orbit_withEpoch.epoch == '24304.21246528'):
        passed = 0
    return passed
def testObjSatellite() -> bool:
    passed        = 1
    start_time    = datetime(2024, 10, 31, 5, 5, 57, tzinfo=timezone.utc)
    apogee        = 421e3 + Constants.R_earth
    orbit_noEpoch = Orbit(
        51.6374,                # inclination          [°]
        0.0009376,              # eccentricity         [-]
        1.5935,                 # RAAN                 [°]
        121.8064,               # argument of perigee  [°]
        341.7727,               # Mean anomaly         [°]
        apogee                  # apogee               [m]
    )

    orbit_withEpoch = Orbit(
        51.6374,                # inclination          [°]
        0.0009376,              # eccentricity         [-]
        1.5935,                 # RAAN                 [°]
        121.8064,               # argument of perigee  [°]
        341.7727,               # Mean anomaly         [°]
        apogee,                 # apogee               [m]
        start_time              # epoch                [YYYY:MM:DD HH:mm:ss UTC]
    )

    sat_noOrbit        = Satellite(1)
    sat_orbitNoEpoch   = Satellite(2, orbit_noEpoch)
    sat_orbitWithEpoch = Satellite(3, orbit_withEpoch)
    satellite_newOrbit = Satellite(
                                   4,                          # Satellite ID        [number]
                                   51.6374,                    # inclination         [°]
                                   0.0009376,                  # eccentricity        [-]
                                   1.5935,                     # RAAN                [°]
                                   121.8064,                   # argument of perigee [°]
                                   341.7727,                   # Mean anomaly        [°]
                                   apogee,                     # apogee              [m]
                                   start_time                  # epoch               [YYYY:MM:DD HH:mm:ss UTC]
    )

    if not (isinstance(sat_noOrbit, Satellite) and sat_noOrbit.orbit == None):
        passed = 0
    if not (isinstance(sat_orbitNoEpoch, Satellite) and sat_orbitNoEpoch.orbit == orbit_noEpoch and sat_orbitNoEpoch.tle_exists() == False):
        passed = 0
    if not (isinstance(sat_orbitWithEpoch, Satellite) and sat_orbitWithEpoch.orbit == orbit_withEpoch):
        passed = 0
    if not (isinstance(satellite_newOrbit, Satellite) and satellite_newOrbit.orbit.epoch == '24304.21246528'):
        passed = 0
    return passed

# FreeFlyer connection "socket" => Send TLE to FF
def testFreeFlyerConnection() -> bool:
    passed = 1
    show_FF_GUI        = True
    GNSS_Constellation = 'GPS'
    sat_1 = Satellite(
        4,                                                      # Satellite ID        [number]
        51.6374,                                                # inclination         [°]
        0.0009376,                                              # eccentricity        [-]
        1.5935,                                                 # RAAN                [°]
        121.8064,                                               # argument of perigee [°]
        341.7727,                                               # Mean anomaly        [°]
        421e3 + Constants.R_earth,                              # apogee              [m]
        datetime(2024, 10, 31, 5, 5, 57, tzinfo=timezone.utc)   # epoch               [YYYY:MM:DD HH:mm:ss UTC]
    )

    # Call the function to send TLE to FreeFlyer
    (success1, eph_file_path1) = FF_TLE_to_Eph(
        sat_1,               # Satellite object            [Satellite]
        GNSS_Constellation,  # GNSS constellation name     [GPS, GLONASS, GALILEO, BEIDOU]
        20,                  # integration step size       [sec]
        1,                   # Simulation duration         [days]
        1,                   # Number of GNSS-R satellites [-]
        show_FF_GUI          # Show FreeFlyer GUI          [True/False]
    )

    if not success1:
        passed = 0
    if not (eph_file_path1 == os.path.join(Constants.tle_eph_folder_path, f"ephemeris_satID_{sat_1.satID}.e")):
        passed = 0


    (success2, eph_file_path2) = FF_TLE_to_Eph(
        sat_1,               # Satellite object            [Satellite]
        'GPS',               # GNSS constellation name     [GPS, GLONASS, GALILEO, BEIDOU]
        20,                  # integration step size       [sec]
        1,                   # Simulation duration         [days]
        1,                   # Number of GNSS-R satellites [-]
        False                # Show FreeFlyer GUI          [True/False]
    )

    if not success2:
        passed = 0
    if not (eph_file_path2 == os.path.join(Constants.tle_eph_folder_path, f"ephemeris_satID_{sat_1.satID}.e")):
        passed = 0
    return passed

def update_test_results(passed: bool, function) -> None:
    if passed:
        print("Test passed")
    else:
        print("Test failed")

if __name__ == '__main__':
    passed = 1
    passed = passed & testObjReflection()
    passed = passed & testObjOrbit()
    passed = passed & testObjSatellite()
    passed = passed & testFreeFlyerConnection()
    if passed:
        print("All tests passed")
