from datetime import datetime
import math
import os

from skyfield.api import load

from object_definitions.Constants_def import Constants
from object_definitions.ReadWriteSatData import ReadWriteSatData
from object_definitions.Rx_antenna_def import RxAntenna
from object_definitions.Satellite_def import Satellite


class SatelliteConstellation:
    """
    =========================================================================================================
    SatelliteConstellation class:
        - Initialized with either an array of Satellite objects OR a TLE file containing the
          satellite configuration (not both).

        INPUTS:
            satArray:      [list]        List of Satellite objects
            tle_file:      [str]         Path to the TLE file containing satellite data
            satConfigName: [str]         Name of the satellite configuration
            rx_antenna:    [RxAntenna]   Receiver antenna object (optional)
            verbose:       [bool]        Verbose output (default: False)
    =========================================================================================================
    """

    def __init__(
        self,
        satArray=None,
        tle_file=None,
        satConfigName=None,
        rx_antenna=None,
        verbose=False,
    ):
        # Check if the anenna is provided and is an instance of RxAntenna
        if isinstance(rx_antenna, RxAntenna):
            antenna = rx_antenna
        elif rx_antenna is not None:
            raise ValueError("Invalid input for receiver antenna. Must be an instance of RxAntenna class.")
        else:
            antenna = None

        # Check if the TLE file is provided / Robustness checks
        if tle_file is not None and satArray is not None:
            raise ValueError("Cannot initialize SatelliteConfiguration with both TLE file and Satellite array")
        elif satArray is None and tle_file is None:
            raise ValueError("SatelliteConfiguration must be initialized with either a TLE file or a Satellite array")
        elif satArray is None and tle_file is not None:
            if os.path.exists(tle_file):
                # Read the TLE data from the file
                if verbose:
                    print(f"Initializing Satellite Configuration with TLE file: {tle_file}")
                satList = ReadWriteSatData.read_tle_data(tle_file)
                satArray = []
                for satNo, sat_data in enumerate(satList):
                    sat = Satellite(
                        sat_data.satCatalogNumber,
                        satName=sat_data.satName,
                        tle_file_path=tle_file,
                        satNo_in_tle=satNo,
                        rx_antenna=antenna,
                        verbose=verbose,
                    )
                    satArray.append(sat)
            else:
                raise FileNotFoundError("TLE file not found")
        elif satArray is not None and tle_file is None:
            if isinstance(satArray, list):
                # Use the provided satellite array
                print("Initializing Satellite Configuration with satellite array")
                pass
            else:
                raise ValueError("Invalid input for Satellite array")
        else:
            raise ValueError("Invalid input")

        # check if the constellation name is valid
        self.satConfigName = satConfigName
        self.satellites = satArray  # An array of Satellite objects
        self.no_satellites = len(satArray)  # Number of satellites in the constellation
        self.orbits_array = [sat_i.get_orbit() for sat_i in self.satellites]  # Array of Orbit objects
        self.times_skyfield = None  # Skyfield timescale object for the simulation duration (set in set_simulation_duration)
        self.duration_sec = None  # Duration in seconds (set in set_simulation_duration)
        self.propagation_step_size = None  # step size in seconds (set in set_simulation_duration)

    def set_simulation_duration(self, startTime, duration, stepSize) -> None:
        """
        =========================================================================================================
        Set the simulation duration for the constellation.

        INPUTS:
            startTime: [datetime]   Start time as datetime object
            duration:  [int|float]  Simulation duration in hours
            stepSize:  [int]        Step size in seconds
        =========================================================================================================
        """
        # Robustness checks for the inputs
        if isinstance(startTime, datetime) and isinstance(duration, int | float) and isinstance(stepSize, int):
            self.duration_sec = math.ceil(duration * Constants.SEC_PER_HOUR)  # Duration in seconds
            num_steps = math.ceil((self.duration_sec) / stepSize)  # derive no. of steps from duration and step size
            self.times_skyfield = load.timescale().utc(  # This  start time is after both GPS and GALILEO TLE Epoch, so I will use this atm
                startTime.year,
                startTime.month,
                startTime.day,
                startTime.hour,
                startTime.minute,
                range(0, num_steps * stepSize, stepSize),
            )
            self.propagation_step_size = stepSize

            # Update the satellites with the new timescale
            for sat in self.satellites:
                sat.set_timescale_simulation(self.times_skyfield)
        else:
            if not isinstance(startTime, datetime):
                raise ValueError("Invalid input for start time. Must be a datetime object.\n  => Set the start time in the configuration file")
            if not isinstance(duration, int | float):
                raise ValueError(
                    "Invalid input for simulation duration. Must be an integer or float representing hours.\n  => Set the simulation duration in the configuration file or generate if from the moving target object (MovingTarget_def.py)"
                )
            if not isinstance(stepSize, int):
                raise ValueError("Invalid input for step size. Must be an integer representing seconds.\n  => Set the step size in the configuration file")
            raise ValueError("Invalid input for simulation duration")

    def add_satellite(self, satellite) -> bool:
        # Add a satellite to the constellation
        self.satellites.append(satellite)
        self.no_satellites += 1
        self.orbits_array.append(satellite.get_orbit())
        return True

    def remove_satellite(self, satellite_id) -> bool:
        # Remove a satellite from the constellation
        for sat in self.satellites:
            if sat.satID == satellite_id:
                self.satellites.remove(sat)
                self.no_satellites -= 1
                self.orbits_array.remove(sat.get_orbit())
                return True
        return False

    def get_constellation_name(self) -> str:
        return self.satConfigName

    def get_constellation_size(self) -> int:
        return self.no_satellites

    def get_constellation_orbits(self) -> list:
        return self.orbits_array

    def propagate_constellation(self, start_time, duration, step_size) -> dict:
        # Propagate the entire constellation
        for sat in self.satellites:
            sat.propagate_orbit(start_time, duration, step_size)
        pass


# TODO - Add the following functions:
# - get_satellite_by_id
# - get_satellite_by_name

# - get_satellite_positions
