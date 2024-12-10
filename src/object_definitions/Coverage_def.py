import os
import geopandas         as gpd
import numpy             as np
import matplotlib.pyplot as plt

from shapely.geometry       import Polygon, LineString
from scipy.spatial.distance import cdist
from skyfield.api           import load
from typing                 import List

class Coverage:
    def __init__(self, output_shapefile):
            self.shapefile               = output_shapefile
            self.grid_gdf                = gpd.read_file(self.shapefile)
            self.coverage_vector         = None

            self.intialize_coverage()

    def intialize_coverage(self):
        # Initialize coverage vector for storing coverage over each cell in the fishnet
        # Make a list of coverage_struct instanced with the length N
        N                    = len(self.grid_gdf)
        self.coverage_vector = [coverage_struct() for _ in range(N)]

        # TODO: IMPROVE THIS
        for idx in range(len(self.grid_gdf)):
            # Compute the centroid of the current grid cell
            centroid                           = self.grid_gdf.loc[idx].geometry.centroid
            self.coverage_vector[idx].position = (centroid.x, centroid.y)
            self.coverage_vector[idx].shape    = self.grid_gdf.loc[[idx]].boundary

    def calculate_coverage(self, bounding_box, reflection_vec, times_2):
        [min_lon, max_lon, min_lat, max_lat] = bounding_box

        for i in range(len(reflection_vec)):
            refl_temp     = reflection_vec[i].refl_LL_vec
            points_in_box = []
            check         = False

            points_in_box = []
            '''
            # Filter points within the bounding box and ignore points outside it
            for point in refl_temp:
                # Check if the point is a valid iterable (tuple or list) with two elements
                if not (isinstance(point, (tuple, list)) and len(point) == 2):
                    points_in_box.append((np.nan, np.nan))  # Add (NaN, NaN) for invalid entries
                    continue
                # Safely unpack the point
                lat, lon = point            
                '''
            for lat, lon in refl_temp:
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    points_in_box.append((lat, lon))
                    check = True
                else:
                    points_in_box.append((np.nan, np.nan))


            points_in_box_1 = np.array(points_in_box)
            valid           = ~np.isnan(points_in_box_1[:, 0])  # Check latitude for NaN
            clusters        = np.split(np.arange(len(points_in_box_1)), np.where(~valid)[0])  # Split indices by NaN gaps
            for cluster in clusters:
                intersecting_rectangles = []
                # Skip clusters with no valid data
                if len(cluster) <= 1: #not valid[cluster].any() &
                    continue
                print(cluster)
                cluster = cluster[1::]
                if len(cluster) <= 1: #not valid[cluster].any() &
                    continue        
                # Extract valid times and positions for this cluster
                valid_cluster  = cluster[valid[cluster]]
                valid_position = points_in_box_1[cluster]
                #print(valid_position)
                line = LineString(zip(valid_position[:, 1], valid_position[:, 0]))
                for idx, polygon in self.grid_gdf.iterrows():
                    # Check if the polygon intersects with the line
                    #print(polygon)
                    intersection = polygon['geometry'].intersection(line)
                    if not intersection.is_empty:
                        intersecting_rectangles.append(idx)
                print(intersecting_rectangles)
#                ax.plot(valid_position[:, 1], valid_position[:, 0], color='red', transform=ccrs.PlateCarree()) # TODO add to plotting function
                if len(intersecting_rectangles) > 0:
#                    intersecting_gdf = self.grid_gdf.loc[intersecting_rectangles]
#                    intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")  # TODO add to plotting function

                    # Save results
                    self.grid_gdf.loc[intersecting_rectangles]
                    valid_position_clean     = valid_position[~np.isnan(valid_position).any(axis=1)]
                    valid_positions_flipped  = valid_position_clean[:, [1, 0]]
                    centroids_array          = np.array([(centroid.x, centroid.y) for centroid in self.grid_gdf.loc[intersecting_rectangles].centroid])
                    distances                = cdist(centroids_array, valid_positions_flipped)
                    nearest_indices          = np.argmin(distances, axis=1)
                    nearest_points           = valid_positions_flipped[nearest_indices]
                    times_at_nearest_points  = [times_2[i] for i in cluster[nearest_indices]]
                    angle_at_nearest_points  = reflection_vec[i].refl_ang_vec[cluster[nearest_indices]]
                    power_at_nearest_points  = reflection_vec[i].power[cluster[nearest_indices]]

                    # Save the nearest points
                    for rect_idx, time, angle, power in zip(intersecting_rectangles, times_at_nearest_points, angle_at_nearest_points, power_at_nearest_points):
                        #print(f"Rectangle Index: {rect_idx}")
                        #print(f"Nearest Time: {time}")
                        # Assign time to coverage_vec
                        self.coverage_vector[rect_idx].times.append(time)
                        self.coverage_vector[rect_idx].tx_satID.append(reflection_vec[i].tx_satID)
                        self.coverage_vector[rect_idx].rx_satID.append(reflection_vec[i].rx_satID)
                        self.coverage_vector[rect_idx].angles.append(angle)
                        self.coverage_vector[rect_idx].power.append(power)




class coverage_struct:
    def __init__(self):
        self.position:                List = []
        self.tx_satID:                List = []
        self.rx_satID:                List = []
        self.times:                   List = []
        self.angles:                  List = []
        self.shape:                   List = []
        self.power:                   List = []
class Coverage_OLD:
    def __init__(self, earth_model=None):

            self.position    = []
            self.tx_satID    = []
            self.rx_satID    = []
            self.times       = []
            self.angle       = []
            self.shape       = []
            self.power       = []

    def calculate_coverage(self, tx_sat_const, rx_sat_const, start_time, end_time, time_step, earth_model=None):
        # Calculate the coverage between the transmitter and receiver constellations
        pass