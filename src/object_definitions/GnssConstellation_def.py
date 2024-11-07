from datetime import datetime
from object_definitions.Constants_def import Constants

class GNSSConstellation:
    def __init__(self, constellation_name, epoch=None):
        self.constellation_name = constellation_name
        self.epoch = epoch or self.get_current_epoch()
        self.no_sat = 0
        self.orbits_array = []
        self.sat_ids = []
        self.initialize_gnss()

    def initialize_gnss(self):
        # Initialize the GNSS constellation with TLE data
        print(f"Initializing GNSS constellation: {self.constellation_name} at epoch {self.epoch}")
        # Additional code to load TLE data and setup satellites here

    @staticmethod
    def get_current_epoch():
        # Returns the current date and time as an epoch
        return datetime.utcnow()

# Example usage
gnss_constellation = GNSSConstellation('GPS')
print(f"Constellation Name: {gnss_constellation.constellation_name}")
print(f"Epoch: {gnss_constellation.epoch}")
