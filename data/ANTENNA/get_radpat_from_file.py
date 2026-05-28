import json
import os

import numpy as np


def read_radpat_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # mesh = np.array(data['data'])
    # freq = data['frequency']
    # beta_m = data['beta_m']
    # beta_n = data['beta_n']
    return data

path = 'DATA\\ANTENNA\\antenna_patterns\\array_factor_1176.45MHz_90.00az_0.00el.json'
#path = 'array_factor_1176.45MHz_90.00az_0.00el.json'
# show current global path
file_path = os.path.join(os.getcwd(), path)
print("Current global path:", file_path)

mydata = read_radpat_file(path)
mesh = np.array(mydata['data'])
freq = mydata['frequency']
beta_m = mydata['azimuth_pointing']
beta_n = mydata['elevation_pointing']

print(freq)
print(beta_m)
print(beta_n)
