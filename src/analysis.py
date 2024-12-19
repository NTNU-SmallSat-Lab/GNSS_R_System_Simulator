import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg', force=True)
import re
import json

# Input file path
#input_file = "simulation_results_parallel_1512_ss.txt"
input_file = "simulation_results_parallel_1412.txt"

# Input file path

# Lists to store extracted values
matrix_list = []  # List of matrices
param1_list = []  # First parameter after the matrix
param2_list = []  # Second parameter after the matrix
param3_list = []  # Third parameter after the matrix
param4_list = []  # Fourth parameter after the matrix

# Function to clean a value (removes unwanted characters, preserves numbers)
def clean_value(value):
    clean = re.sub(r'[^\d.\-+]', '', value.strip())  # Remove non-numeric trailing characters
    try:
        return float(clean)  # Convert to float if possible
    except ValueError:
        return None

# Function to extract the matrix and trailing parameters
def extract_values(line):
    match = re.search(r"(TLE_EPHEMIRIS_files/[\w\.]+),(\[\[.*?\]\]),(.*)", line)
    if match:
        matrix = match.group(2).strip()  # Extract matrix as a string
        after_matrix = match.group(3).strip().rstrip('-')  # Clean trailing dashes
        
        # Parse the matrix string into a nested list using json.loads
        try:
            matrix_values = json.loads(matrix)
        except json.JSONDecodeError:
            matrix_values = []  # Empty list if parsing fails
        
        # Extract and clean trailing parameters
        after_values = [clean_value(val) for val in after_matrix.split(",")]
        return matrix_values, after_values
    return [], []

# Process the file to extract values
with open(input_file, "r") as infile:
    for line in infile:
        matrix_values, after_values = extract_values(line)
        if matrix_values and len(after_values) >= 4:
            # Append the cleaned values
            matrix_list.append(matrix_values)  # Append the full nested matrix
            param1_list.append(after_values[0])  # First parameter
            param2_list.append(after_values[1])  # Second parameter
            param3_list.append(after_values[2])  # Third parameter
            param4_list.append(after_values[3])  # Fourth parameter



# Function to extract desired parameters
def extract_matrix_info(matrix):
    inclination = matrix[0][0]  # Inclination is the first element
    semi_major_axis = matrix[0][-1]/1e3  # Semi-major axis is the last element
    Nsat = len(matrix)  # Number of vectors in the matrix
    planes = len(set(row[2] for row in matrix))  # Count unique values in the third element
    return inclination, semi_major_axis, Nsat, planes

# Print the extracted results
print("Matrices:")
for matrix in matrix_list:
    print(matrix)
    inclinations, semi_major_axis, Nsat, plane = extract_matrix_info(matrix)
    print(inclinations, semi_major_axis, Nsat, plane)



# Call the function and print results
# Lists to store the extracted values for plotting
satellites = []  # X-axis: Satellite index
inclinations = []  # Y-axis: Inclination
'''
param1_values = []
param2_values = []
param3_values = []
param4_values = []  # Z-axis: Parameter 3
'''
semi_major_axes = []
Nsats = []
planes = []
# Loop through the matrices and extract the data for plotting
for idx, matrix in enumerate(matrix_list):
    print(matrix)
    inclination, semi_major_axis, Nsat, plane = extract_matrix_info(matrix)  # Extract inclination (Y-axis)
    '''
    # Assuming that `param3_list` has the correct values for Parameter 3 (Z-axis)
    param1 = param1_list[idx]  # Extract the third parameter value (Z-axis)
    param2 = param2_list[idx]  # Extract the third parameter value (Z-axis)
    param3 = param3_list[idx]  # Extract the third parameter value (Z-axis)
    param4 = param4_list[idx]  # Extract the third parameter value (Z-axis)
    '''
    # Add the extracted values to the lists
    satellites.append(Nsat)  # Use the index as a satellite proxy (X-axis)
    inclinations.append(inclination)  # Inclination (Y-axis)
    '''
    param1_values.append(param1)  # Parameter 3 (Z-axis)
    param2_values.append(param2)  # Parameter 3 (Z-axis)
    param3_values.append(param3)  # Parameter 3 (Z-axis)
    param4_values.append(param4)  # Parameter 3 (Z-axis)
    '''
    semi_major_axes.append(semi_major_axis)
    Nsats.append(Nsat)
    planes.append(plane)


# Convert lists to numpy arrays for easier handling
satellites = np.array(satellites)
print(satellites)
inclinations = np.array(inclinations)
'''
param1_values = np.array(param1_list)
param2_values = np.array(param2_list)
param3_values = np.array(param3_list)
param4_values = np.array(param4_list)
'''


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plotting the data in vertically stacked subplots
fig = plt.figure(figsize=(10, 14))  # Increase figure height for stacked plots

# Create two subplots (nrows=2, ncols=1 for vertical stacking)
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

# First subplot: Nsat vs Inclination vs Zonal Coverage
scatter1 = ax1.scatter(satellites, inclinations, param2_list, c=planes, cmap='viridis', marker='o')
ax1.set_title('Nsat vs Inclination vs Zonal Coverage')
ax1.set_xlabel('Satellites')
ax1.set_ylabel('Inclination [degrees]')
ax1.set_zlabel('Zonal Coverage [%]')
ax1.view_init(elev=20, azim=-160)  # Adjust these values to change the viewpoint
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax1.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')

# Second subplot: Nsat vs Inclination vs Average Captures (or Parameter 4)
scatter2 = ax2.scatter(satellites, inclinations, param4_list, c=planes, cmap='viridis', marker='o')
ax2.set_title('Nsat vs Inclination vs Average Captures')
ax2.set_xlabel('Number of Satellites')
ax2.set_ylabel('Inclination [degrees]')
ax2.set_zlabel('Average Captures')  # Change to Parameter 3 or Parameter 4 if needed
ax2.view_init(elev=20, azim=-160)  # Adjust these values to change the viewpoint
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax2.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')

# Adjust layout to make sure subplots don't overlap
plt.tight_layout()

# Show the plot
plt.savefig("Figure_1_paper.png", dpi = 400)






# Plotting the data in vertically stacked subplots
fig = plt.figure(figsize=(10, 14))  # Increase figure height for stacked plots

# Create two subplots (nrows=2, ncols=1 for vertical stacking)
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

# First subplot: Nsat vs Inclination vs Zonal Coverage
scatter1 = ax1.scatter(satellites, semi_major_axes, param2_list, c=planes, cmap='viridis', marker='o')
ax1.set_title('Nsat vs Semi_major_axis vs Zonal Coverage')
ax1.set_xlabel('Satellites')
ax1.set_ylabel('Semi_major_axis [km]')
ax1.set_zlabel('Zonal Coverage [%]')
ax1.view_init(elev=20, azim=-160)  # Adjust these values to change the viewpoint
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax1.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')

# Second subplot: Nsat vs Inclination vs Average Captures (or Parameter 4)
scatter2 = ax2.scatter(satellites, semi_major_axes, param4_list, c=planes, cmap='viridis', marker='o')
ax2.set_title('Nsat vs Semi_major_axes vs Average Captures')
ax2.set_xlabel('Number of Satellites')
ax1.set_ylabel('Semi_major_axis [km]')
ax2.set_zlabel('Average Captures')  # Change to Parameter 3 or Parameter 4 if needed
ax2.view_init(elev=20, azim=-160)  # Adjust these values to change the viewpoint
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax2.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')

# Adjust layout to make sure subplots don't overlap
plt.tight_layout()

# Show the plot
plt.savefig("Figure_3_paper.png", dpi = 400)


'''
##################################################
# Plotting the data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
scatter = ax.scatter(satellites, inclinations, param2_list, c=planes, cmap='viridis', marker='o')
# Add a color bar to indicate the number of planes
# Set title
ax.set_title('Nsat vs Inclination vs Zonal Coverage ')
# Format the X-axis to show only integer ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
# Add a custom legend based on planes values (for simplicity, using a set of fixed ranges of planes)
unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')


# Set labels for the axes
ax.set_xlabel('Satellites')
ax.set_ylabel('Inclination [degrees]')
ax.set_zlabel('Zonal Coverage [%]')
ax.view_init(elev=20, azim=-130)  # Adjust these values to change the viewpoint

# Set title
ax.set_title('Satellite vs Inclination vs Zonal Coverage')
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


plt.show()


##################################################

# Plotting the data for Nsat vs Inclination vs Parameter 2
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
#ax.scatter(satellites, inclinations, param2_list, c='r', marker='o')
# Create scatter plot with color based on the planes (unique values in third element of matrix)
scatter = ax.scatter(satellites, inclinations, param4_list, c=planes, cmap='viridis', marker='o')
# Add a color bar to indicate the number of planes
# Set title
ax.set_title('Nsat vs Inclination vs Average Captures')
# Format the X-axis to show only integer ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
# Add a custom legend based on planes values (for simplicity, using a set of fixed ranges of planes)
unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')


# Set labels for the axes
ax.set_xlabel('Number of Satellites')
ax.set_ylabel('Inclination [degrees]')
ax.set_zlabel('Average Captures')  # Change to Parameter 3 or Parameter 4 if needed
ax.view_init(elev=20, azim=-130)  # Adjust these values to change the viewpoint

# Set title
ax.set_title('Nsat vs Inclination vs Zonal Coverage')
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


##################################################
'''
# Plotting the data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Create scatter plot
ax.scatter(semi_major_axes, param1_list, c='r', marker='o')

# Set labels for the axes
ax.set_xlabel('Semi_major_axis [km]')
ax.set_ylabel('Average Relative Signal Power ')

# Set title
ax.set_title('Semi_major_axis vs Signal Power')
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.savefig("Figure_2_paper.png", dpi = 400)


##################################################

# Plotting the data for Nsat vs Inclination vs Parameter 2
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
#ax.scatter(satellites, inclinations, param2_list, c='r', marker='o')
# Create scatter plot with color based on the planes (unique values in third element of matrix)
scatter = ax.scatter(satellites, semi_major_axes, param4_list, c=planes, cmap='viridis', marker='o')
# Set labels for the axes
ax.set_xlabel('Number of Satellites')
ax.set_ylabel('Semi_major_axis [km]')
ax.set_zlabel('Average Captures')  # Change to Parameter 3 or Parameter 4 if needed

# Set title
ax.set_title('Nsat vs Semi_major_axis vs Average Captures')
# Format the X-axis to show only integer ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
# Add a custom legend based on planes values (for simplicity, using a set of fixed ranges of planes)
unique_planes = sorted(set(planes))  # Get unique planes values
# Create dummy handles for the legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_planes)-1)), markersize=10) 
           for i in range(len(unique_planes))]
# Add a legend with labels corresponding to the unique number of planes
ax.legend(handles, [f'{plane} planes' for plane in unique_planes], title='Number of Planes')
ax.view_init(elev=20, azim=-150)  # Adjust these values to change the viewpoint


# Show the plot
plt.show()
