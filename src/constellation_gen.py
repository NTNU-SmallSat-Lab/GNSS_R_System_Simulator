import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import random 
from object_definitions.Constants_def import Constants


def wrapto360(angle):
    return angle % 360

def delta_pattern_hybrid(Nsat, Nplane, f, incl, a, RAAN_0):
    """
    Generate and plot the given Walker delta constellation.

    Parameters:
    Nsat (int): Number of satellites
    Nplane (int): Number of orbital planes
    f (int): Anomaly difference between successive satellites in adjacent orbits
    incl (float): Orbit inclination (degrees)
    a (float): Orbit semi-major axis (km)
    plots (str): 'y' to plot the constellation, 'n' otherwise
    theta_rel (float): 0 for SAR subset, otherwise Optical subset
    fig (int): Figure number

    Returns:
    tuple: Satellite and plane constellation matrices (if applicable)
    """
    if f < 0 or f > (Nplane - 1) or f % 1 != 0 or Nsat % 1 != 0 or Nplane % 1 != 0 or Nsat % Nplane != 0:
        print("Wrong parameter choice")
        print(f"{Nsat}, {Nplane}, {f}")
        return None, None
    
    # Calculate Pattern Unit for Walker constellation
    PU = 360  / Nsat
    # Calculate number of satellites per plane
    NSsat = Nsat / Nplane
    # Angle division between satellites in the plane
    beta = PU * Nplane
    # RAAN spacing
    dRAAN = PU * NSsat
    TAo = np.zeros(Nsat)
    RAANo = np.zeros(Nsat)

    for i in range(Nsat):
        pindex = int(np.ceil((i + 1) / NSsat))  # Plane index
        Spindex = i - (pindex - 1) * NSsat
        TAo[i] = wrapto360(Spindex * beta + (pindex - 1) * f * PU)
        RAANo[i] = wrapto360((pindex - 1) * dRAAN + RAAN_0)

    CONSTELLATION_SC = np.zeros((Nsat, 6))

    for i in range(Nsat):
        #CONSTELLATION_SC[i, :] = [a, 0, incl, RAANo[i], 0, TAo[i]]
        CONSTELLATION_SC[i, :] = [incl, 1e-5, RAANo[i], 0, TAo[i], a]
    return CONSTELLATION_SC

def define_orbit_vector(max_inputs, start_time_dt):
    # Generate 30 random configurations
    random_configs = []
    results = []
    for _ in range(max_inputs*10):
        Nsat = random.randint(1, 3)  # Random Nsat (1 to 3)
        Nplane = random.choice([1, Nsat])  # Random Nplane (1 or 3)
        f    = random.randint(0, Nplane - 1)  # Random f (0 to Nplane-1)
        incl = np.round(random.uniform(60, 80), 2)  # Random inclination (60 to 80 degrees)
        a    = random.randint( 6878*1e3, 7078*1e3)  # Random semi-major axis (7800 to 8000 km)
        RAAN_0 = np.round(random.uniform(0, 360), 2)  # Random RAAN (0 to 360)
        if Nplane <= Nsat:  # Only add the config if Nplane <= Nsat
            config = (Nsat, Nplane, f, incl, a, RAAN_0)
            if config not in random_configs:  # Ensure uniqueness
                random_configs.append(config)
    # Evaluate each configuration


    for idx, (Nsat, Nplane, f, incl, a, RAAN_0) in enumerate(random_configs, start=1):
        if len(results) >= max_inputs:
            #print("More than 30 results collected. Exiting loop.")
            break

        #print(f"Configuration {idx}: Nsat={Nsat}, Nplane={Nplane}, f={f}, incl={incl:.1f}, a={a:.1f}")
        constellation = delta_pattern_hybrid(Nsat, Nplane, f, incl, a, RAAN_0)
        if constellation is not None:
            results.append(constellation)
            #print(f"Output {idx}:")
            #print(constellation)
        #print("\n")

    orbits_input_list = []
    from datetime import datetime

    #print(results)
    for array in results:
        orbit_list = []
        for row in array:
            # Extracting parameters
            semi_major_axis = row[5]  # Semi-major axis
            eccentricity = row[1]  # Eccentricity
            inclination = row[0]  # Inclination
            RAAN = row[2]
            ArgPer = row[3]
            Theta = row[4]
            # Constructing each orbit configuration (adding start_time_dt to each)
            orbit_list.append([inclination, eccentricity, RAAN, ArgPer,Theta, semi_major_axis, start_time_dt])
            # Adding a second configuration with 90 degree inclination
        
        orbits_input_list.append(orbit_list)

    # Printing the result
    return orbits_input_list

#from datetime import datetime

#start_time_dt = datetime.now()
#define_orbit_vector(5, start_time_dt)