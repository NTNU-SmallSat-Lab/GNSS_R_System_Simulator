import json               as json
import configparser       as cp
import numpy              as np
import xmltodict          as xd

__version__ = "0.1.0"

def initialize():
    config = cp.ConfigParser()
    config.read('Configuration.ini')

    return config 

# Export the initialize function
__all__ = ['initialize']
#__all__ = ['initialize', 'optical_calc', 'georeferencing']