# run_simulation.py
import os
from function_main import main, save_results_to_file
from datetime import timezone, datetime
from constellation_gen import define_orbit_vector
# Define input parameters
start_time_dt = datetime(2024, 10, 23, 12, 00, 00, tzinfo=timezone.utc)
step_size = 30  # Seconds
sim_duration = 48  # Hours
bounding_box = [-17, 30, 56, 80]  # [min_lon, max_lon, min_lat, max_lat]
gnss_tle_file_path = os.path.join("TLE_EPHEMIRIS_files", "GPS_operational.tle")

# Define multiple orbits_input sets
N = 50

orbits_input_list = define_orbit_vector(2, start_time_dt)

print(orbits_input_list)

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

# Open the file once in append mode to add results
with open(output_file_path, 'a') as result_file:
    # Loop through each orbits_input set and run the simulation
    for orbits_input in orbits_input_list:
        # Run the simulation for the current set of orbital parameters by calling main directly
        average_power, coverage_percentage, sunlight_percentage, count_avg = main(
            start_time_dt, step_size, sim_duration, orbits_input, gnss_tle_file_path, bounding_box
        )
        print(orbits_input)
        # Save the input parameters and results to the file (appending mode)
        save_results_to_file(
            result_file,  # Pass the open file object to the function
            start_time_dt, step_size, sim_duration, bounding_box, orbits_input,
            gnss_tle_file_path, average_power, coverage_percentage, sunlight_percentage, count_avg
        )

        # Optionally, print a message to show progress
        print(f"Results for orbits_input {orbits_input} saved.")

print(f"All simulations completed. Results saved to {output_file_path}.")
