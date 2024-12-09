import os
import geopandas         as gpd
import numpy             as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

from skyfield.api import load

class Coverage:
    def __init__(self, earth_model=None):

            self.position    = []
            self.tx_satID    = []
            self.rx_satID    = []
            self.times       = []
            self.angle       = []
            self.shape       = []
            self.power       = []

    def calculate_coverage(self, tx_sat_const, rx_sat_const, start_time, end_time, time_step, earth_model=None):
        # Calculate the coverage between the transmitter and receiver constellations
        