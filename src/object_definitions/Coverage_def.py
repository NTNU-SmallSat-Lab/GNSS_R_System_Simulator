import os
import geopandas         as gpd
import numpy             as np
import matplotlib.pyplot as plt

from shapely.geometry       import Polygon, LineString
from scipy.spatial.distance import cdist
from skyfield.api           import load
from typing                 import List
from dataclasses            import dataclass, field

class Coverage:
    def __init__(self, output_shapefile):
            self.shapefile               = output_shapefile
            self.grid_gdf                = gpd.read_file(self.shapefile)
            self.coverage_vector         = None

            self.intialize_coverage()

    def intialize_coverage(self) -> None:
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

    def calculate_coverage(self, bounding_box, reflection_vec, times_2) -> None:
        '''
        Filter points within the bounding box and ignore points outside it:
            # Check if the point is a valid iterable (tuple or list) with two elements
            if not (isinstance(point, (tuple, list)) and len(point) == 2):
                points_in_box.append((np.nan, np.nan))  # Add (NaN, NaN) for invalid entries
                continue
            # Safely unpack the point
            lat, lon = point            
        '''
        for i in range(len(reflection_vec)):
            validity_mask, ref_points_in_box = self.validity_mask2(reflection_vec[i], bounding_box)
            # Assembly of points in the bounding box for one GNSS-R satellite and one GNSS satellite
            refl_clusters = self.__get_reflection_clusters(reflection_vec[i], validity_mask)
            for cluster in refl_clusters:
                intersecting_rectangles = []
                # Skip refl_clusters with no valid data
                if len(cluster) <= 1: #not validity_mask[cluster].any() &
                    continue

                cluster = cluster[1::]
                if len(cluster) <= 1: #not validity_mask[cluster].any() &
                    continue        
                # Extract valid times and positions for this cluster
#                valid_cluster  = cluster[validity_mask[cluster]]
                valid_position = ref_points_in_box[cluster]
                #print(valid_position)
                line = LineString(zip(valid_position[:, 1], valid_position[:, 0]))
                for idx, polygon in self.grid_gdf.iterrows():
                    # Check if the polygon intersects with the line
                    #print(polygon)
                    intersection = polygon['geometry'].intersection(line)
                    if not intersection.is_empty:
                        intersecting_rectangles.append(idx)

                if len(intersecting_rectangles) > 0:
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
                        # Assign time to coverage_vec
                        self.coverage_vector[rect_idx].times.append(time)
                        self.coverage_vector[rect_idx].tx_satID.append(reflection_vec[i].tx_satID)
                        self.coverage_vector[rect_idx].rx_satID.append(reflection_vec[i].rx_satID)
                        self.coverage_vector[rect_idx].angles.append(angle)
                        self.coverage_vector[rect_idx].power.append(power)

    @staticmethod
    def validity_mask(reflection_obj, bounding_box) -> np.ndarray:
        if not np.isnan(reflection_obj.refl_LL_vec).all():
            refl_in_bounding_box                        = []
            [min_lon, max_lon, min_lat, max_lat] = bounding_box

            for lat, lon in reflection_obj.refl_LL_vec: # TODO: IMPROVE THIS Why append points in box for both valid and invalid points? Better skip invalid points???
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                     refl_in_bounding_box.append((lat, lon))
                else:
                    refl_in_bounding_box.append((np.nan, np.nan))

            refl_in_bb    = np.array(refl_in_bounding_box)
            validity_mask = ~np.isnan(refl_in_bb[:,0])                                       # Check latitude for NaN 
        else:
            # There are only NaN values in the refl_LL_vec
            validity_mask = np.zeros(len(reflection_obj.refl_LL_vec))
            refl_in_bb    = np.array(refl_in_bounding_box)
        return validity_mask, refl_in_bb

    @staticmethod
    def validity_mask2(reflection_obj, bounding_box):
        refl_LL_vec = np.array(reflection_obj.refl_LL_vec)
    
        if not np.isnan(refl_LL_vec).all():
            min_lon, max_lon, min_lat, max_lat = bounding_box
        
            # Create boolean masks for latitude and longitude
            lat_mask = (min_lat <= refl_LL_vec[:, 0]) & (refl_LL_vec[:, 0] <= max_lat)
            lon_mask = (min_lon <= refl_LL_vec[:, 1]) & (refl_LL_vec[:, 1] <= max_lon)
        
            # Combine masks
            validity_mask = lat_mask & lon_mask
        
            # Create refl_in_bounding_box with NaN for invalid points
            refl_in_bounding_box                 = refl_LL_vec.copy()
            refl_in_bounding_box[~validity_mask] = np.nan
        else:
            # There are only NaN values in the refl_LL_vec
            validity_mask           = np.zeros(len(reflection_obj.refl_LL_vec))
            refl_in_bounding_box    = np.full_like(reflection_obj.refl_LL_vec, np.nan)
    
        return validity_mask, refl_in_bounding_box
    
    def __get_reflection_clusters(self, reflection_obj, validity_mask):
        clusters = np.split(np.arange(len(reflection_obj.refl_LL_vec)), np.where(~validity_mask)[0])  # Split indices by NaN gaps
        return clusters

@dataclass
class coverage_struct:
    position:   List = field(default_factory=list)
    tx_satID:   List = field(default_factory=list)
    rx_satID:   List = field(default_factory=list)
    times:      List = field(default_factory=list)
    angles:     List = field(default_factory=list)
    shape:      List = field(default_factory=list)
    power:      List = field(default_factory=list)
