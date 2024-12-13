# run_simulation.py
import os
from function_main import main, save_results_to_file
from datetime import timezone, datetime
from constellation_gen import define_orbit_vector
import traceback
# Define input parameters
start_time_dt = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)
step_size = 30  # Seconds
sim_duration = 2  # Hours
bounding_box = [-17, 30, 56, 80]  # [min_lon, max_lon, min_lat, max_lat]
gnss_tle_file_path = os.path.join("TLE_EPHEMIRIS_files", "GPS_operational.tle")

# Define multiple orbits_input sets

orbits_input_list = define_orbit_vector(3, start_time_dt)

print(len(orbits_input_list))
print(orbits_input_list)
print(len(orbits_input_list))

'''
orbits_input_list = [
    [[99.1, 0.00001, 0, 0, 0, 7300e3, start_time_dt], [99.1, 0.00001, 90.0, 0, 0, 7300e3, start_time_dt]],  # Orbit 1
    [[73, 0.00001, 0, 0, 0, 7000e3, start_time_dt], [73, 0.00001, 90.0, 0, 0, 7000e3, start_time_dt]],
    [[80, 0.00001, 0, 0, 0, 7000e3, start_time_dt], [80, 0.00001, 90.0, 0, 0, 7000e3, start_time_dt]]  # Orbit 3
]
'''

# Output file path to save results
output_file_path = "simulation_results.txt"

# Clear the file by opening it in write mode ('w') and immediately closing it
with open(output_file_path, 'w') as result_file:
    result_file.write("")  # This ensures the file is empty before appending new results


with open(output_file_path, 'a') as result_file:
    # Loop through each orbits_input set and run the simulation
    for idx, orbits_input in enumerate(orbits_input_list, start=1):
        try:
            print(f"Processing orbit set {idx}: {orbits_input}...")
            # Run the simulation for the current set of orbital parameters by calling main directly
            average_power, coverage_percentage, sunlight_percentage, count_avg = main(
                start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box
            )
            cleaned_orbits_input = [orbit[:-1] for orbit in orbits_input]  # Remove the last element (datetime) from each orbit
            #print(cleaned_orbits_input)

            # Save the input parameters and results to the file (appending mode)
            save_results_to_file(
                result_file,  # Pass the open file object to the function
                start_time_dt, step_size, sim_duration, bounding_box, cleaned_orbits_input,
                gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage, count_avg
            )
            # Explicitly flush the file to ensure data is written to disk
            result_file.flush()
            # Log success
            print(f"Results for orbits_input {orbits_input} saved successfully.")
        except Exception as e:
            # Get the full traceback details, including the line number
            error_details = traceback.format_exc()
            print(f"Error processing orbit set {orbits_input}: {e}")
            print(f"Traceback details:\n{error_details}")
            print("Skipping to the next orbit set...\n")
            continue
