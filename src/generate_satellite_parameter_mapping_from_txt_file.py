

import datetime
import json

def parse_datetime(date_str):
    """ Convert date string 'YYYY:DDD:SSSSS' into a datetime object. """
    if date_str == "0000:000:00000":
        return None  # Represent active satellites with None or a future date
    
    try:
        year, day_of_year, seconds = date_str.split(":")
        year = int(year)
        day_of_year = int(day_of_year)
        seconds = int(seconds)

        # Convert seconds into HHMMSS format
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = (seconds % 60)

        # Format into a valid string for strptime()
        formatted_time = f"{hours:02d}{minutes:02d}{seconds:02d}"

        # Now parse with strptime()
        return datetime.datetime.strptime(f"{year}:{day_of_year}:{formatted_time}", "%Y:%j:%H%M%S")
    except ValueError:
        print(f"Could not convert {date_str} to datetime object")
        return None


# Format of the line to start reading from:
# *SVN_ Valid_From____ Valid_To______ PRN Comment_________________________________
# *
# G001 1978:053:00000 1985:199:00000 G04 
# [...]
# Stop reading when the line starts with a '*'


if __name__ == '__main__':
    with open('DatabaseGPS.txt', 'r') as f:
        lines = f.readlines()

        # Create a SVN to SatCat mapping
        start_of_svn_to_satcat_index = None
        for i, line in enumerate(lines):
            if '*SVN_ COSPAR ID SatCat Block__________ Comment__________________________________' in line:
                start_of_svn_to_satcat_index = i + 2
                break
            
        for i, line in enumerate(lines[start_of_svn_to_satcat_index:]):
            if line.startswith('*'):
                end_of_svn_to_satcat_index = start_of_svn_to_satcat_index + i
                break

        if (start_of_svn_to_satcat_index or end_of_svn_to_satcat_index) is None:
            raise ValueError('Could not find the start or end index')

        svn_to_satcat_map = {}
        for line in lines[start_of_svn_to_satcat_index:end_of_svn_to_satcat_index]:
            columns = line.split()
            svn = columns[0]
            cospar_id = columns[1]
            satcat = columns[2]
            svn_to_satcat_map[svn] = {
                "COSPAR_ID": cospar_id,
                "SatCat": satcat
            }


        # Find the index of the line that starts with '*'
        start_index = None
        for i, line in enumerate(lines):
            if '*SVN_ Valid_From____ Valid_To______ PRN Comment_________________________________' in line:
                start_index = i + 2 # Jump over the '*' line
                break
        
        end_index = None
        for i, line in enumerate(lines[start_index+1:]):
            if line.startswith('*'):
                end_index = start_index + i
                break
        
        if (start_index or end_index) is None:
            raise ValueError('Could not find the start or end index')

        # Format: SVN, (START, STOP), PRN
        svn_to_prn_mapping = []
        # Format: SVN, (START, STOP), PRN
        svn_to_prn_mapping = []
        for line in lines[start_index:end_index]:
            columns = line.split()
            svn = columns[0]
            start = columns[1]
            stop = columns[2]
            prn = columns[3]

            # Convert dates
            start_dt = parse_datetime(start)
            stop_dt = parse_datetime(stop)

            if start_dt is None:
                print(f"Failed the line: {line}")
                break 

            # svn_to_prn_mapping.append((svn, (start_dt, stop_dt), prn))
            svn_to_prn_mapping.append({
                "SVN": svn,
                "NORAD_ID": svn_to_satcat_map[svn]["SatCat"],
                "COSPAR_ID": svn_to_satcat_map[svn]["COSPAR_ID"],
                "Start": start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else None,
                "Stop": stop_dt.strftime("%Y-%m-%d %H:%M:%S") if stop_dt else None,
                "PRN": prn
            })
            

    with open('satellite_parameter_mapping.json', 'w') as f:
        json.dump(svn_to_prn_mapping, f, indent=4)
    print("Data saved to svn_to_prn_mapping.json")


            
