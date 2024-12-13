import os
import numpy           as np
import cartopy.crs     as ccrs
import cartopy.feature as cfeature

import matplotlib
#matplotlib.use('TkAgg', force=True)
from matplotlib             import pyplot as plt
from skyfield.api           import load

class Plotting:
    @staticmethod
    def plot_reflection_traces(tx_sat_const, rx_sat_const, reflection_vec, file_name):
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()
        
        for i_sat in range(rx_sat_const.no_satellites):
            for i_gnss in range(tx_sat_const.no_satellites):
                ax.scatter(reflection_vec[i_sat*tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 1], reflection_vec[i_sat*tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())
        #        ax.scatter(refl_gps.refl_LL_vec[:, 1], refl_gps.refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())
        
        # Plot reflection points
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gridlines              = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
        gridlines.top_labels   = False  # Disable top labels
        gridlines.right_labels = False  # Disable right-side labels
        
        plt.title('Specular Reflection Points for GPS and LEO Satellite Over Time')
        plt.savefig("Reflection_traces.png", dpi=300)

    @staticmethod
    def plot_coverage(coverage_obj, reflection_vec, bounding_box, file_name):
        [min_lon, max_lon, min_lat, max_lat] = bounding_box
        grid_gdf = coverage_obj.grid_gdf
        # Initialize figure and axis with projection
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()

        ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent

        # Plot the grid as the background (light gray color)
        grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")

        # Loop through the reflection vectors
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()
        ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent
        grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")

        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()
        ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent
        grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")
        for coverage_vect in coverage_obj.coverage_vector:
            if len(coverage_vect.val_pos) > 0:
                ax.plot(coverage_vect.val_pos[0][1], coverage_vect.val_pos[0][0], 'ro', transform=ccrs.PlateCarree())


        '''
        if len(coverage_obj.intersecting_rectangles) > 0:
#            intersecting_gdf = self.grid_gdf.loc[intersecting_rectangles]
#            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")  # TODO add to plotting function
            intersecting_gdf = coverage_obj.grid_gdf.loc[coverage_obj.intersecting_rectangles]
            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")
        
        '''
        plt.savefig("Coverage1.png", dpi = 300)
