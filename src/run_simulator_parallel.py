import os
from function_main import main, save_results_to_file
from datetime import timezone, datetime
from constellation_gen import define_orbit_vector
import traceback
from multiprocessing import Pool

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


# Define input parameters
start_time_dt = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)
step_size = 30  # Seconds
sim_duration = 48  # Hours
bounding_box = [-17, 30, 56, 80]  # [min_lon, max_lon, min_lat, max_lat]
gnss_tle_file_path = os.path.join("TLE_EPHEMIRIS_files", "GPS_operational.tle")

# Define multiple orbits_input sets
orbits_input_list = define_orbit_vector(200, start_time_dt)


print(len(orbits_input_list))
print(orbits_input_list)
print(len(orbits_input_list))

# Output file path to save results
output_file_path = "simulation_results_parallel.txt"

# Clear the file by opening it in write mode ('w') and immediately closing it
with open(output_file_path, 'w') as result_file:
    result_file.write("")  # This ensures the file is empty before appending new results

# Define the worker function to run the simulation and save results
def process_orbit_set(orbits_input):
    try:
        print(f"Processing orbit set {orbits_input}...")
        # Run the simulation for the current set of orbital parameters by calling main directly
        average_power, coverage_percentage, sunlight_percentage, count_avg = main(
            start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box
        )
        cleaned_orbits_input = [orbit[:-1] for orbit in orbits_input]  # Remove the last element (datetime) from each orbit
        #print(cleaned_orbits_input)

        # Open file in append mode and save the input parameters and results to the file
        with open(output_file_path, 'a') as result_file:
            save_results_to_file(
                result_file,  # Pass the open file object to the function
                start_time_dt, step_size, sim_duration, bounding_box, cleaned_orbits_input,
                gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage, count_avg
            )
            # Log success
            print(f"Results for orbits_input {orbits_input} saved successfully.")
    except Exception as e:
        # Get the full traceback details, including the line number
        error_details = traceback.format_exc()
        print(f"Error processing orbit set {orbits_input}: {e}")
        print(f"Traceback details:\n{error_details}")
        print("Skipping to the next orbit set...\n")

# Initialize the Pool and process the orbits_input_list in parallel
if __name__ == "__main__":
    # Create a Pool of workers (you can specify the number of processes, or leave it empty to use the default number of CPU cores)
    with Pool(14) as pool:
        pool.map(process_orbit_set, orbits_input_list)  # Distribute the orbit sets to the pool workers