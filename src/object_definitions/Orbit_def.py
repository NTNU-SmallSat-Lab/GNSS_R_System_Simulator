#from datetime import UTC, datetime
from datetime import UTC, datetime

import numpy as np

from object_definitions.Constants_def import Constants


class Orbit_specification:
    """A set of classical (Keplerian) orbital elements, with a TLE-format epoch.

    The mean motion ``n`` is derived from the semi-major axis. With an epoch set, a Satellite can turn
    these elements into a TLE and propagate.
    """

    def __init__(self, i, e, raan, p, M, a, epochTime=None) -> None:
        """Store the orbital elements and derive the mean motion from the semi-major axis.

        Parameters
        ----------
        i : float
            Inclination [deg].
        e : float
            Eccentricity [-].
        raan : float
            Right ascension of the ascending node [deg].
        p : float
            Argument of perigee [deg].
        M : float
            Mean anomaly [deg].
        a : float
            Semi-major axis [m].
        epochTime : datetime, optional
            Orbit epoch (UTC); sets the TLE epoch string.
        """
        # Check if enough variables are provided
        if any(var is None for var in [i, e, raan, p, M, a]):
            raise ValueError("Orbit_specification undefined, not sufficient variables provided")
        try:
            i, e, raan, p, M, a = (float(value) for value in (i, e, raan, p, M, a))
        except (TypeError, ValueError) as exc:
            raise ValueError("Orbital elements must be finite numeric values.") from exc
        if not np.isfinite([i, e, raan, p, M, a]).all():
            raise ValueError("Orbital elements must be finite numeric values.")
        if not (0.0 <= i <= 180.0):
            raise ValueError(f"Inclination must be in [0, 180] deg; got {i}.")
        if not (0.0 <= e < 1.0):
            raise ValueError(f"Eccentricity must be in [0, 1); got {e}.")
        if a * (1.0 - e) <= Constants.RADIUS_EARTH_EQUAT:
            raise ValueError(
                f"Orbit perigee radius a*(1-e) must exceed the WGS-84 equatorial radius; got {a * (1.0 - e)} m."
            )
        if epochTime is not None and (
            not isinstance(epochTime, datetime)
            or epochTime.tzinfo is None
            or epochTime.utcoffset() != UTC.utcoffset(epochTime)
        ):
            raise ValueError("epochTime must be a timezone-aware UTC datetime.")

        # Initialize orbit
        self.i = i  # Inclination in degrees
        self.e = e  # Eccentricity
        self.RAAN = raan  # Right Ascension of Ascending Node in degrees
        self.p = p  # Argument of perigee in degrees
        self.M = M  # Mean anomaly
        self.a = a  # Semi-major axis in m
        self.epoch_str = self.conv_DateTime_to_Epoch(epochTime) if epochTime else None  # Eopch in YYDDD.DDDDDDDD format (TLE format)
        self.dt_start = epochTime if epochTime else None  # Start time in datetime format
        self.n = Constants.SEC_PER_DAY / (2 * np.pi * np.sqrt(self.a**3 / Constants.GM_EARTH_TT))  # Mean motion in rev/day

    def conv_DateTime_to_Epoch(self, dt_epoch):
        # Convert datetime to TLE epoch_str format YYDDD.DDDDDDDD
        tle_year = dt_epoch.year % 100
        start_of_year = datetime(dt_epoch.year, 1, 1, tzinfo=UTC)
        days_since_start_of_year = (dt_epoch - start_of_year).total_seconds() / Constants.SEC_PER_DAY + 1  # Add 1 day to accont for the first day of the year

        # Format strings to TLE Epoch format
        epoch_year_str = f"{tle_year:02d}"
        epoch_days_str = f"{days_since_start_of_year:012.8f}"
        tle_epoch = epoch_year_str + epoch_days_str
        return tle_epoch

    """
    =========================================================================================================
    Setter functions
    =========================================================================================================
    """

    def set_start_time(self, dt_start):
        # Set the start time of the orbit
        # Check if start time is [YYYY:MM:DD HH:mm:ss UTC] format
        if not isinstance(dt_start, datetime):
            raise ValueError("Invalid start time format. Please provide a datetime object")
        # Check if dt_start is timezone-aware and in UTC
        if dt_start.tzinfo is None or dt_start.tzinfo != UTC:
            raise ValueError("Start time must be a timezone-aware datetime object in UTC")
        self.dt_start = dt_start
        self.epoch_str = self.conv_DateTime_to_Epoch(dt_start)

    """
    =========================================================================================================
    Getter functions
    =========================================================================================================
    """

    @staticmethod
    def generate_satID():
        # Generate satellite ID based on files in the TLE directory
        import os

        tle_files = [f for f in os.listdir(Constants.tle_eph_folder_path) if f.endswith(".tle")]

        # Generate a unique ID by counting files (example logic)
        satID = len(tle_files) + 1
        return satID
