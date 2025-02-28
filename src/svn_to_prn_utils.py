import json
import datetime
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

def load_svn_to_prn_mapping(json_filename: str) -> List[Dict[str, Union[str, int, datetime.datetime, None]]]:
    """ 
    Load SVN to PRN mappings from a JSON file into a list of dictionaries.
    This function is agnostic to the underlying JSON structure and will automatically parse all fields.
    
    Args:
        json_filename (str): Path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries where all fields from the JSON are preserved,
                    and datetime fields are converted to `datetime.datetime` objects.
    """
    with open(json_filename, 'r') as f:
        data = json.load(f)

    parsed_data = []
    
    for item in data:
        parsed_item = {}

        for key, value in item.items():
            if isinstance(value, str):
                try:
                    parsed_item[key] = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    parsed_item[key] = value
            else:
                parsed_item[key] = value  # Keep other types as-is

        parsed_data.append(parsed_item)

    return parsed_data

def prn_from_svn(data:List, svn:str, time_range:Optional[datetime.datetime]=None):
    """ Show all entries have ever been broadcast by a given SVN. 
    If a range is provided, only return PRNs that were broadcast during that time.
    If no range is provided, return all PRNs that were ever broadcast by the SVN."""

    matching_entries = []
    for item in data:
        if item["SVN"] == svn:
            if time_range:
                if item["Start"] <= time_range <= item["Stop"]:
                    matching_entries.append(item['PRN'])
            else:
                matching_entries.append(item['PRN'])

    return matching_entries

def svn_from_prn(data:List, prn:str, time_range:Optional[datetime.datetime]=None):
    """ Show all entries that have ever broadcast a given PRN. 
    If a date range is provided, only return SVN that were broadcast during that time.
    If no range is provided, return all SVN that ever broadcast the PRN."""

    matching_entries = []
    for item in data:
        if item["PRN"] == prn:
            if time_range:
                if item["Start"] <= time_range <= item["Stop"]:
                    matching_entries.append(item['SVN'])
            else:
                matching_entries.append(item['SVN'])

    return matching_entries


def prn_to_norad_id(data: List, prn: str, time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None):
    """ Return the NORAD Id of all satellites that have ever broadcast a given PRN. 
    If a date range is provided, only return NORAD IDs that broadcasted the PRN 
    during that time. If no range is provided, return all NORAD IDs that ever 
    broadcast the PRN."""
    
    matching_entries = []
    for item in data:
        if item["PRN"] == prn:
            if time_range:
                start, end = time_range
                if item["Stop"] == None: # If the satellite is still broadcasting
                    if item["Start"] <= start:
                        matching_entries.append(item["NORAD_ID"])
                elif item["Start"] <= start and item["Stop"] >= end:
                    matching_entries.append(item["NORAD_ID"])
            else:
                matching_entries.append(item["NORAD_ID"])

    return matching_entries

JSON_MAPPING_FILENAME = 'satellite_parameter_mapping.json'
if __name__ == '__main__':
    data = load_svn_to_prn_mapping(JSON_MAPPING_FILENAME)

    # svn = 'G034'
    # matching_entries = prn_from_svn(data, svn)
    # print(matching_entries)

    # prn = 'G08'
    # matching_entries = svn_from_prn(data, prn)
    # print(matching_entries)

    prn = 'G20'
    matching_entries = prn_to_norad_id(data, prn)
    #print(matching_entries)
    for match in matching_entries:
        print(match)
        #print(match["PRN"], match["NORAD_ID"])

