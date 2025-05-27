from datetime import datetime, timedelta
from io import StringIO
from spacetrack import SpaceTrackClient
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import spacetrack.operators as op



def extract_lines_in_range(file_path, start_line, end_line):
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Extract lines in the specified range (line numbers are 0-indexed in Python, so we adjust by -1)
    relevant_lines = lines[start_line-1:end_line]    
    return relevant_lines


def GetUpdatedTLE(start_time):
    file_path = '../DatabaseGPS.txt'  # Replace with your file path
    start_line = 864
    end_line = 1146

    # Read the full file content
    file = "../satellite_info.txt"
    with open(file, "r") as f:
        content = f.read()

    # Split into Part 1 and ignore Part 2
    part1 = content.split("PART 2: ON-BOARD SENSORS")[0]

    # Extract only the data lines (skip headers and blank lines)
    lines = [line for line in part1.splitlines() if line.strip().startswith(tuple(str(i) for i in range(1, 10)))]

    # Fixed-width column specs for Part 1
    colspecs = [
        (0, 3), (5, 8), (9, 14), (15, 27),
        (35, 54), (55, 74)
    ]

    columns = [
        "PRN", "SVN", "BLOCK", "COSPAR",
        "START_TIME", "END_TIME", 
    ]

    # Parse fixed-width data
    df = pd.read_fwf(StringIO('\n'.join(lines)), colspecs=colspecs, names=columns)
    df = df.iloc[:-15]
    df['SVN'] = df['SVN'].astype(int) # Convert to int

    # Helper to convert date fields
    def parse_datetime(val):
        try:
            parts = str(val).strip().split()
            return datetime(*map(int, parts)) if len(parts) == 6 else None
        except:
            return None

    # Apply datetime and numeric parsing
    df["START_TIME"] = df["START_TIME"].apply(parse_datetime)
    df["END_TIME"] = df["END_TIME"].apply(parse_datetime)


    def get_system(prn):
        if 0 <= prn < 100:
            return "GPS"
        elif 100 <= prn < 200:
            return "Glonass"
        elif 200 <= prn < 300:
            return "Galileo"
        else:
            return "Other"

    df["SYSTEM"] = df["PRN"].apply(get_system)
    # Split by system
    gps_df = df[df["SYSTEM"] == "GPS"]
    glonass_df = df[df["SYSTEM"] == "Glonass"]
    galileo_df = df[df["SYSTEM"] == "Galileo"]

    gps_df['SVN'] = 'G' + '0' + gps_df['SVN'].astype(str)  # Prefix 'G' for GPS
    glonass_df['SVN'] = 'R' + glonass_df['SVN'].astype(str)  # Prefix 'R' for Glonass
    galileo_df['SVN'] = 'E' + galileo_df['SVN'].astype(str)  # Prefix 'E' for Galileo

    # Helper function to check if the input time is within the satellite's operational period
    def is_valid_satellite(row, input_time):
        # Check if END_TIME is NaN or if the input time is between START_TIME and END_TIME
        if pd.isna(row['END_TIME']):
            return row['START_TIME'] <= input_time  # If END_TIME is NaN, it's valid as long as START_TIME is before the input time
        return row['START_TIME'] <= input_time <= row['END_TIME']

    # Function to get valid COSPAR IDs for a given system
    def get_valid_satellites(input_time, df):
        # Convert input time to datetime
        input_time = datetime.strptime(input_time, "%Y-%m-%d %H:%M:%S")
        
        # Filter satellites that are valid at the input time using apply with a custom function (is_valid_satellite)
        valid_satellites = df[df.apply(is_valid_satellite, input_time=input_time, axis=1)]
        
        # Return a list of tuples (SVN, COSPAR)
        valid_satellites_data = list(zip(valid_satellites['SVN'], valid_satellites['COSPAR']))
        
        return valid_satellites_data

    # Example usage:

    data_lines = extract_lines_in_range(file_path, start_line, end_line)
    # Join the list of strings into a single string to simulate a file-like object
    data_str = '\n'.join(data_lines)

    # Define the column specs (adjust these for your data's fixed-width format)
    colspecs = [
        (0, 6),    # SVN
        (6, 17),   # COSPAR
        (17, 23),  # SatCat
        (25, 37),  # Block
        (37, None) # Comment
    ]

    # Define the column names
    columns = ["SVN", "COSPAR", "SatCat", "Block", "Comment"]

    # Use pandas.read_fwf to read the fixed-width formatted text from the StringIO object
    df_SATCAT = pd.read_fwf(StringIO(data_str), colspecs=colspecs, names=columns)

    # Display the resulting DataFrame
    print(df_SATCAT)


    # Parse the original string to a datetime object
    dt = datetime.strptime(start_time, "%d.%m.%Y %H:%M:%S")
    # Format it to the desired format
    input_time = dt.strftime("%Y-%m-%d %H:%M:%S")  # You can still keep this for printing
    print(input_time)
    print(type(dt))  # Not input_time

    date_1 = dt.date()- timedelta(days=3)
    date_2 = dt.date() 
    drange = op.inclusive_range(date_1, date_2)

    # Get valid COSPAR IDs for each system
    valid_gps = get_valid_satellites(input_time, gps_df)
    valid_glonass = get_valid_satellites(input_time, glonass_df)
    valid_galileo = get_valid_satellites(input_time, galileo_df)

    svn_list_GPS = [svn  for svn, _ in valid_gps]
    filtered_df = df_SATCAT[df_SATCAT['SVN'].isin(svn_list_GPS)]
    # Now, you can select the 'SatCat' from the filtered rows
    satcat_values_GPS = filtered_df['SatCat'].values
    print(satcat_values_GPS)

    svn_list_galileo = [svn  for svn, _ in valid_galileo]
    filtered_df = df_SATCAT[df_SATCAT['SVN'].isin(svn_list_galileo)]
    # Now, you can select the 'SatCat' from the filtered rows
    satcat_values_galileo = filtered_df['SatCat'].values
    print(satcat_values_galileo)

    svn_list_glonass = [svn  for svn, _ in valid_glonass]
    filtered_df = df_SATCAT[df_SATCAT['SVN'].isin(svn_list_glonass)]
    # Now, you can select the 'SatCat' from the filtered rows
    satcat_values_glonass = filtered_df['SatCat'].values
    print(svn_list_glonass)

    # Retrieve access to TLEs to SpaceTrack
    dotenv_path = os.path.join(os.path.dirname(os.getcwd()), '.env')
    load_dotenv(dotenv_path)
    identity = os.getenv("SPACETRACK_ID")
    pw = os.getenv("SPACETRACK_PASS")
    # Let's keep it private it....
    st = SpaceTrackClient(identity=identity, password=pw)
    import time
    for norad_cat_id_input in satcat_values_glonass:
        try:
            lines = list(st.tle(iter_lines=True, norad_cat_id=norad_cat_id_input, epoch=drange, orderby="EPOCH", format="tle"))
            time.sleep(2)  # gentle pause to avoid rate limits
            #print(lines)
            if lines:
                last_elements = lines[-2:]
                print(last_elements)
            else:
                time.sleep(2)  # gentle pause to avoid rate limits
                print("Retrying", norad_cat_id_input)
                #lines = list(st.tle(iter_lines=True, norad_cat_id=norad_cat_id_input, epoch=drange, orderby="EPOCH", format="tle"))
                last_elements = lines[-2:]
                print(last_elements)

        except Exception as e:
            print(f"Error with {norad_cat_id_input}: {e}")







startTime = "10.01.2004 12:00:00"
GetUpdatedTLE(startTime)