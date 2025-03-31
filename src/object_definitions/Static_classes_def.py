import os
import numpy           as np
import cartopy.crs     as ccrs
import cartopy.feature as cfeature

import matplotlib
matplotlib.use('TkAgg', force=True)
from matplotlib                       import pyplot as plt
from skyfield.api                     import load
from object_definitions.Constants_def import Constants
class Plotting:
    @staticmethod
    def plot_reflection_traces_gnss_const(rx_SatConst, tx_SatConst, reflection_vec, file_name="Reflection_traces.png", title=None):
        '''
        Plot the reflection traces for a given GNSS constellation.
        '''
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()

        for i_sat in range(tx_SatConst.no_satellites):
            for i_gnss in range(rx_SatConst.no_satellites):
                ax.scatter(reflection_vec[i_sat*rx_SatConst.no_satellites + i_gnss].refl_LL_vec[:, 1], reflection_vec[i_sat*rx_SatConst.no_satellites + i_gnss].refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())
        #        ax.scatter(refl_gps.refl_LL_vec[:, 1], refl_gps.refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())

        # Plot reflection points
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gridlines              = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
        gridlines.top_labels   = False  # Disable top labels
        gridlines.right_labels = False  # Disable right-side labels

        if title is None:
            title_string = f'Specular Reflection Points for {tx_SatConst.satConfigName} constellation and {rx_SatConst.no_satellites} LEO satellites'
        else:
            title_string = title
        plt.title(title_string)
        save_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(save_path, dpi=600)

    @staticmethod
    def plot_reflection_traces_total(rx_SatConst, tx_SatConstDict, reflection_dict, file_name="Reflection_traces_total.png", title=None):
        '''
        Plot the reflection traces for all GNSS constellations.
        '''
        def _color(gnss_const):
            if gnss_const == 'GPS':
                color = 'red'
            elif gnss_const == 'GLONASS':
                color = 'orange'
            elif gnss_const == 'GALILEO':
                color = 'blue'
            elif gnss_const == 'BEIDOU':
                color = 'green'
            elif gnss_const == 'QZSS':
                color = 'purple'
            elif gnss_const == 'IRNSS':
                color = 'pink'
            else:
                color = None
            return color
        
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()

        # Loop through the reflection vectors for each GNSS constellation
        for gnss_key in tx_SatConstDict:
            tx_sat_const       = tx_SatConstDict[gnss_key]
            reflections_vector = reflection_dict[gnss_key]
            col                = _color(gnss_key)
            for i_sat in range(rx_SatConst.no_satellites):
                for i_gnss in range(tx_sat_const.no_satellites):
                    ax.scatter(reflections_vector[i_sat*tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 1], reflections_vector[i_sat*tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 0], color=col, s=1, transform=ccrs.PlateCarree())
        #        ax.scatter(refl_gps.refl_LL_vec[:, 1], refl_gps.refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        gridlines              = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
        gridlines.top_labels   = False  # Disable top labels
        gridlines.right_labels = False  # Disable right-side labels

        if title is None:
            constellations = ', '.join(tx_SatConstDict.keys())
            title_string = f'Specular Reflection Points for the constellations: {constellations} and {rx_SatConst.no_satellites} LEO satellites'
        else:
            title_string = title
        plt.title(title_string)

        # Add a legend
        legend_labels = [f'{gnss_key}' for gnss_key in tx_SatConstDict.keys()]
        legend_colors = [_color(gnss_key) for gnss_key in tx_SatConstDict.keys()]
        handles       = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=5) for label, color in zip(legend_labels, legend_colors)]
        ax.legend(handles=handles, loc='upper right', fontsize=8, title="Constellations")

        save_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(save_path, dpi=600)

    @staticmethod
    def plot_coverage(coverage_obj, reflection_vec, bounding_box, file_name="Coverage1.png"):
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
        for i in range(len(reflection_vec)):
            ax.plot(reflection_vec[i].refl_LL_vec[:, 1], reflection_vec[i].refl_LL_vec[:, 0], color='red', transform=ccrs.PlateCarree())

        #for i in range(len(coverage_obj.coverage_vector)):
#        for cluster in clusters:
            #ax.plot(coverage_obj.coverage_vector[i].position[0], coverage_obj.coverage_vector[i].position[1], 'ro', transform=ccrs.PlateCarree())
#            ax.plot(valid_position[:, 1], valid_position[:, 0], color='red', transform=ccrs.PlateCarree()) # TODO add to plotting function

#        if len(intersecting_rectangles) > 0:
        '''
        if len(coverage_obj.intersecting_rectangles) > 0:
#            intersecting_gdf = self.grid_gdf.loc[intersecting_rectangles]
#            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")  # TODO add to plotting function
            intersecting_gdf = coverage_obj.grid_gdf.loc[coverage_obj.intersecting_rectangles]
            intersecting_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Intersecting Rectangles")
        
        '''
        plt.savefig(file_name, dpi = 300)
