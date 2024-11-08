import socket, time, os, subprocess, struct
import numpy as np
from object_definitions.Orbit_def import Orbit
from object_definitions.Constants_def import Constants

from typing import Tuple

def FF_TLE_to_Eph(satObj, GNSS_const, integrationStep, simDuration, no_GNSS_R_sat, showGUI=False) -> Tuple[bool, str]:
    """
    ==========================
    = Robustness checks      =
    ==========================
    """
    if not isinstance(satObj.orbit, Orbit):
        print(f"Warning: Satellite {satObj.satID} does not contain an Orbit")
        return Tuple(None, None)

    if satObj.tle_file_path:
        tle_file_path = satObj.tle_file_path
        satID         = str(satObj.satID)
    else:
        print("Warning: Filepath to TLE file not found")
        return Tuple(None, None)

    eph_file_path = os.path.join(Constants.tle_eph_folder_path, f"ephemeris_satID_{satID}.e")

    """
    ==========================
    = File paths             =
    ==========================
    """
    terminationCode = '10191980'
    MissionPlanPath = Constants.FFmissionPlanTLE_EPH

    """
    ==========================================
    = Try to using the OS-determined ports   =
    ==========================================
    """
    # Start a timer
    start_time = time.time()

    # Initialize lists to store sockets and port numbers
    socket_server = [None, None]
    port_num      = [None, None]

    # Open socket 1
    try:
        socket_server[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_server[0].bind(('', 0))                       # Bind to a free port assigned by the OS
        socket_server[0].listen(1)                             # Start listening for connections
        port_num[0] = socket_server[0].getsockname()[1]      # Retrieve the assigned port number
    except Exception as err:
        raise RuntimeError(f"Error: Unable to open socket. {err}")

    # Open socket 2
    try:
        socket_server[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_server[1].bind(('', 0))                   # Bind to a free port assigned by the OS
        socket_server[1].listen(1)                       # Start listening for connections
        port_num[1] = socket_server[1].getsockname()[1]  # Retrieve the assigned port number
    except Exception as err:
        raise RuntimeError(f"Error: Unable to open socket. {err}")

    # Display elapsed time
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

    """
    ==============================
    = Launch FreeFlyer instance  =
    ==============================
    """
    print(f"\n- Launching FreeFlyer instance on port numbers {port_num[0]} and {port_num[1]}. -\n")

    # Create the command string based on whether the GUI is to be shown
    if showGUI:
        command_string = (
            f'"{Constants.FreeFlyerInstallationPath}FreeFlyer.exe" -r -mp "{MissionPlanPath}" '
            f'-ui {port_num[0]} -ui {port_num[1]} -ui {terminationCode} &'
        )
    else:
        command_string = (
            f'"{Constants.FreeFlyerInstallationPath}ff.exe" -mp "{MissionPlanPath}" '
            f'-ui {port_num[0]} -ui {port_num[1]} -ui {terminationCode} &'
        )

    
    #os.system(command_string)
#    subprocess.run(command_string, shell=True)
    subprocess.Popen(command_string, shell=True)

    """
    ==========================================
    = Wait for FreeFlyer response/connect    =
    ==========================================
    """
    # Define timeouts for the sockets in seconds
    timeout = 60  # seconds

    # Wait for the FreeFlyer connection on the 1st socket
    socket_server[0].settimeout(timeout)
    try:
        print("\nWaiting for FreeFlyer connection on the first socket...")
        socket_client_1, addr_1 = socket_server[0].accept()  # Accept a connection
        print(f"Connection established with FreeFlyer client at {addr_1}")
    except socket.timeout:
        print("Error: Unable to accept MissionPlan as client on the first socket (timeout)")
        raise RuntimeError("Error: Unable to accept MissionPlan as client on the first socket")
    except Exception as err:
        print("Error: Unable to accept MissionPlan as client on the first socket")
        raise RuntimeError(f"Error: Unable to accept MissionPlan as client on the first socket. {err}")

    # Wait for the FreeFlyer connection on the 2nd socket
    socket_server[1].settimeout(timeout)
    try:
        print("Waiting for FreeFlyer connection on the second socket...")
        socket_client_2, addr_2 = socket_server[1].accept()  # Accept a connection
        print(f"Connection established with FreeFlyer client at {addr_2}")
    except socket.timeout:
        print("Error: Unable to accept MissionPlan as client on the second socket (timeout)")
        raise RuntimeError("Error: Unable to accept MissionPlan as client on the second socket")
    except Exception as err:
        print("Error: Unable to accept MissionPlan as client on the second socket")
        raise RuntimeError(f"Error: Unable to accept MissionPlan as client on the second socket. {err}")

    print("  Both sockets opened.")

    """
    ==========================================
    = Create write/read buffer and           =
    = interchange (send) data with FF client =
    ==========================================
    """
    str_to_FF = f"{satID} {tle_file_path} {eph_file_path} {GNSS_const}"

    ############################
    # Send String to FreeFlyer #
    ############################
    with socket_client_1.makefile('wb') as output_stream:
        try:
            output_stream.write(str_to_FF.encode('utf-8') + b'\0')  # Null-terminate the message
            #output_stream.flush()                                   # Ensure that the data is sent
            print(f"Message string sent to FreeFlyer:")
        except Exception as err:
            print(f"Error: Unable to send string to FreeFlyer. {err}")

    output_array = [np.float64(showGUI), np.float64(integrationStep), np.float64(simDuration), np.float64(no_GNSS_R_sat)]
    array_size   = [1, np.float64(len(output_array))]

    ############################
    # Send output array size   #
    ############################

    with socket_client_1.makefile('wb') as output_stream:
        try:
            for value in array_size:
                output_stream.write(struct.pack('>d', value)) # '>d' is big-endian double precision (This is required for FreeFlyer)
            output_stream.flush()
        except Exception as err:
            print(f"Error: Unable to send array size to FreeFlyer. {err}") 

    #############################
    # Send output array details #
    #############################
    with socket_client_1.makefile('wb') as output_stream:
        try:
            for value in output_array:
                output_stream.write(struct.pack('>d', value)) # '>d' is big-endian double precision (This is required for FreeFlyer)
        except Exception as err:
            print(f"Error: Unable to send array details to FreeFlyer. {err}")

    """
    ============================================
    = Wait for FF Instances to finish running  =
    =   and receive all the FreeFlyer data     =
    ============================================
    """
    print("'  Waiting for responses from FreeFlyer instance before continuing... \n'")

    # Connect to the FreeFlyer socket or use an existing socket connection
    raw_data          = []
    stringResponse    = []
    dblResponse       = []
    arrResponse       = []
    bigArrResponse    = []
    stringArrResponse = []

    with socket_client_2.makefile('rb') as input_stream:
        print("  Waiting for responses from FreeFlyer instance before continuing... ")
        while True:
            # Read a line of input from the stream
            line = input_stream.readline().decode('utf-8').strip()  # Decodes to string and removes any newline characters
            raw_data.append(line)
            # Check if the received line matches the termination code and break the loop if it does
            if line == terminationCode:
                print("- All data received. -")
                break

    """
    ============================================
    = Parse the FreeFlyer data                 =
    ============================================
    """
    idx        = 0
    str_idx    = 0
    dbl_idx    = 0
    arr_idx    = 0
    bigArr_idx = 0
    strArr_idx = 0
    while idx < len(raw_data):
        line = raw_data[idx]
        if line == terminationCode:
            print("All data parsed.")
            break
        else:
            # Check the identifier and read the next line accordingly
            if line == "STR":
                stringResponse.append(raw_data[idx + 1])
                str_idx += 1
            elif line == "DBL":
                dblResponse.append(float(raw_data[idx + 1]))
                dbl_idx += 1
            elif line == "ARR":
                arrResponse.append([float(x) for x in raw_data[idx + 1].split()])
                arr_idx += 1
            elif line == "bigARR":
                bigArrResponse.append([float(x) for x in raw_data[idx + 1].split()])
                bigArr_idx += 1
            elif line == "strARR":
                stringArrResponse.append(raw_data[idx + 1])
                strArr_idx += 1
            else:
                print(f"Unknown data type: {line}")

            # Increment index by 2 to skip to the next identifier
            idx += 2


    """
    ============================================
    = Get path to Ephemiris file               =
    ============================================
    """
    # arrResponse[0][0] == 1 FreeFlyer has executed mission plan successfully
    # arrResponse[0][0] == 0 FreeFlyer has NOT executed mission plan successfully
    ephemeris_file_path = None
    success             = False
    if arrResponse[0][0] == 1 and len(stringArrResponse) == 1:
        file_name = os.path.basename(stringArrResponse[0])
        if "ephemeris" in file_name:
            ephemeris_file_path = stringArrResponse[0]
            success             = True
            print("Ephemeris file received from FreeFlyer:", ephemeris_file_path)
        else:
            print("Ephemeris file was NOT received from FreeFlyer.")

    """
    ============================================
    = Close the sockets                        =
    ============================================
    """
    # CARRY ON FROM HERE
    # Close all connections
    for sock in socket_server:
        sock.close()

    """
    ============================================
    = Return ephemiris file path               =
    ============================================
    """
    return (success, ephemeris_file_path) if success else (None, None)
