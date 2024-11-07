import socket, time, os, io, subprocess
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

    mission_Plan  = Constants.FFmissionPlanTLE_EPH
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

    print("REACHED HERE.")
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
    = Create write/read buffer               =
    = and Data interchange with FF client    =
    ==========================================
    """
    str_to_FF = f"{satID} {tle_file_path} {eph_file_path} {GNSS_const}"

    # Create write buffer (equivalent to DataOutputStream in Java)
    with socket_client_1.makefile('wb') as output_stream:
        output_stream.write(str_to_FF.encode('utf-8') + b'\0')  # Null-terminate the message
        output_stream.flush()                                   # Ensure that the data is sent
        print(f"Message string sent to FreeFlyer:")

    # Send [showGUI, integrationStep, simDuration, no_GNSS_R_sat] as an array to FreeFlyer
    







    # CARRY ON FROM HERE
    # Create read buffer (equivalent to DataInputStream in Java)
    with socket_client_2.makefile('rb') as input_stream:
        pass
#    input_stream = socket_client_2.makefile('rb')   # For binary input (read)

    # Create buffered reader (similar to BufferedReader in Java)
    # In Python, we can use the `io` module for buffered reading
    input_reader = io.BufferedReader(input_stream)

















#CARRY ON FROM HERE
    # Socket setup
    portNum        = []
    server_sockets = []
    client_sockets = []
    for i in range(2):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', 0))
        server_socket.listen(1)
        portNum.append(server_socket.getsockname()[1])
        server_sockets.append(server_socket)

    # FreeFlyer connection
    print(f'Waiting for FreeFlyer connection on ports {portNum[0]} and {portNum[1]}...')

    for i in range(2):
        client_socket, _ = server_sockets[i].accept()
        client_sockets.append(client_socket)
        print(f'Socket {i + 1} connected.')

    # Create write/read buffer
    outputStream = client_sockets[0].makefile('wb')
    inputStream = client_sockets[1].makefile('rb')

    # Data interchange with FreeFlyer client
    str_to_FF = f"{satID} {tle_file_path} {eph_file_path} {GNSS_const}"
    outputStream.write(f"{str_to_FF}\0".encode())  # Send the data with null terminator
    outputStream.flush()

    # Send output array details
    outputArray = [int(showGUI), integrationStep, simDuration, no_GNSS_R_sat]
    for value in outputArray:
        outputStream.write(value.to_bytes(8, 'little'))
    outputStream.flush()

    # Wait for data from FreeFlyer
    print("Waiting for responses from FreeFlyer instance before continuing...")
    rawData = []
    while True:
        data = inputStream.readline().decode()
        if data.strip() == terminationCode:
            break
        rawData.append(data.strip())

    print("All data received. Processing data...")

    # Process received data
    stringResponse, dblResponse, arrResponse, bigArrResponse, stringArrResponse = [], [], [], [], []
    i = 0
    while i < len(rawData):
        if rawData[i] == "STR":
            stringResponse.append(rawData[i + 1])
            i += 1
        elif rawData[i] == "DBL":
            dblResponse.append(float(rawData[i + 1]))
            i += 1
        elif rawData[i] == "ARR":
            arrResponse.append([float(x) for x in rawData[i + 1].split()])
            i += 1
        elif rawData[i] == "bigARR":
            bigArrResponse.append([float(x) for x in rawData[i + 1].split()])
            i += 1
        elif rawData[i] == "strARR":
            stringArrResponse.append(rawData[i + 1])
            i += 1
        i += 1

    # Decode and display results
    success = False
    ephemeris_file = None
    if arrResponse and arrResponse[0][0] == 1 and len(stringArrResponse) == 1:
        file_name = os.path.basename(stringArrResponse[0])
        if "ephemeris" in file_name:
            ephemeris_file = stringArrResponse[0]
            success = True
            print("Ephemeris file received from FreeFlyer:", ephemeris_file)
        else:
            print("Ephemeris file was NOT received from FreeFlyer.")

    # Close all connections
    for sock in server_sockets:
        sock.close()
    for sock in client_sockets:
        sock.close()
    
    if diary:
        diary.close()

    return Tuple(success, eph_file_path) if success else Tuple(None, None)
