import os
import numpy           as np
import cartopy.crs     as ccrs
import cartopy.feature as cfeature
import geopandas       as gpd

import matplotlib
matplotlib.use('TkAgg', force=True)
from matplotlib                       import pyplot as plt
from skyfield.api                     import load
from object_definitions.Constants_def import Constants

class Plotting:
    @staticmethod
    def plot_reflection_traces_gnss_const(rx_SatConst, tx_SatConst, reflection_vec, file_name="Reflection_traces.png", title=None) -> None:
        '''
        Plot the reflection traces for a given GNSS constellation.

        INPUTS:
            rx_SatConst:     Satellite constellation object for the receiver (LEO) satellites.
            tx_SatConst:     Satellite constellation object for the transmitter (GNSS) satellites.
            reflection_vec:  List of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            file_name:       Name of the output file to save the plot (optional).
            title:           Title of the plot (optional).
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

        INPUTS:
            rx_SatConst:     Satellite constellation object for the receiver (LEO) satellites.
            tx_SatConstDict: Dictionary of satellite constellation objects for the transmitter (GNSS) satellites.
            reflection_dict: Dictionary of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            file_name:       Name of the output file to save the plot (optional).
            title:           Title of the plot (optional).
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
    def plot_coverage_shapefile(coverage_obj, reflection_vec, bounding_box, file_name="Coverage1.png"):
        '''
        Plot the coverage area of the GNSS constellation and the reflection vectors.
        
        INPUTS:
            coverage_obj:    Coverage object containing the grid and coverage information.
            reflection_vec:  List of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            bounding_box:    Bounding box coordinates for the plot (min_lon, max_lon, min_lat, max_lat).
            file_name:       Name of the output file to save the plot (optional).
        '''
        [min_lon, max_lon, min_lat, max_lat] = bounding_box
        grid_gdf                             = coverage_obj.grid_gdf
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
        file_path = os.path.join(Constants.output_folder_path, file_name)
#        plt.title(f'Coverage Area with Reflection Vectors')
        plt.savefig(file_path, dpi = 600)
    @staticmethod
    def plot_coverage_gridded_shapefile(coverage_obj, reflection_obj, file_name="Coverage2.png"):
        '''
        Plot the coverage area of the GNSS constellation and the reflection vectors using a gridded shapefile.
        Matches the original code's visualization style.
    
        INPUTS:
            coverage_obj:    Coverage object containing the grid and coverage information.
            reflection_obj:  List of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            bounding_box:    Bounding box for the plot region [min_lon, max_lon, min_lat, max_lat].
            file_name:       Name of the output file to save the plot (optional).
        '''
        def _get_color(count): # TODO move to Coverage_def.py or plotting_def.py
            if count == 1:
                return 'red', 1  # Color for 1-time intersection
            elif count > 1:
                return 'blue', 0.5  # Color for more than 1-time intersection
            else:
                return 'none', 0  # If the cell isn't intersected at all

        # Use the grid directly from the coverage object
        grid_gdf = coverage_obj.grid_gdf

        # Create intersection count dictionary
        intersection_count = {}
        min_length         = min(len(grid_gdf), len(coverage_obj.coverage_vector))

        for i in range(min_length):
            idx = grid_gdf.index[i]
            intersection_count[idx] = len(coverage_obj.coverage_vector[i].times)

        # Define color based on the count
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()

        # Set extent based on bounding box
        ax.set_extent([coverage_obj.bounding_box[0], coverage_obj.bounding_box[1], coverage_obj.bounding_box[2], coverage_obj.bounding_box[3]])

        # Plot the grid background
        grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', 
                         linestyle='-', alpha=0.5, label="Grid Background")

        # First plot all cells with count=1 (red)
        for i in range(min_length):
            idx   = grid_gdf.index[i]
            count = intersection_count[idx]
            color, alpha_val = _get_color(count)

            if color != 'none':  # If there's at least one intersection
                coverage_obj.coverage_vector[i].shape.plot(
                ax=ax, edgecolor='red', facecolor=color, alpha=alpha_val)

        # Then plot cells with count>1 (blue) on top
        for i in range(min_length):
            idx              = grid_gdf.index[i]
            count            = intersection_count[idx]
            color, alpha_val = _get_color(count)
            
            if color == 'blue':  # Only plot blue cells (more than 1 intersection)
                coverage_obj.coverage_vector[i].shape.plot(
                ax=ax, edgecolor='blue', facecolor=color, alpha=alpha_val)

        # Plot reflection lines
        for i in range(len(reflection_obj)):
            # Skip NaN values to avoid plotting issues
            valid_mask = ~np.isnan(reflection_obj[i].refl_LL_vec).any(axis=1)
            if np.any(valid_mask):
                lon = reflection_obj[i].refl_LL_vec[valid_mask, 1]
                lat = reflection_obj[i].refl_LL_vec[valid_mask, 0]
                ax.plot(lon, lat, color='red', transform=ccrs.PlateCarree())

        # Add labels and gridlines
        ax.set_title('Coverage of grid cells by reflection lines')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels   = False
        gl.right_labels = False
        
        ## Add legend
        #from matplotlib.patches import Patch
        #legend_elements = [
        #    Patch(facecolor='red', edgecolor='red', alpha=0.3, label='1 intersection'),
        #    Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='2+ intersections')
        #]
        #ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()

        file_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(file_path, dpi=600)

        print(f"Coverage plot saved to {file_path}")

        # Close the plot to free up memory
        plt.close()
        ###########################################################################################################################
