#from main_simulator import create_coverage 
import geopandas       as gpd
from matplotlib             import pyplot as plt
from shapely.geometry       import Polygon, LineString
import numpy as np
import os
grid_size = 1


###########################################################################################################################
# Function to create fishnet grid
def create_coverage(grid_size):
    # Define the path to your shapefile (iho.shp)
    cwd = os.getcwd()
    shapefile = os.path.join(cwd, 'iho', 'iho.shp')

    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile)

    # Get the bounds of the shapefile (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define the grid size (e.g., 0.1 degrees for latitude/longitude)
    grid_size = 1

    # Generate the fishnet grid (list of polygons)
    grid_polygons = []
    for x0 in np.arange(minx, maxx, grid_size):
        for y0 in np.arange(miny, maxy, grid_size):
            x1 = x0 + grid_size
            y1 = y0 + grid_size
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # Create a GeoDataFrame for the fishnet
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs=gdf.crs)
    # Filter out grid cells that do not intersect with the IHO shapefile
    grid_gdf = grid_gdf[grid_gdf.intersects(gdf.unary_union)]

    # Save the grid as a shapefile
    output_shapefile = os.path.join(cwd, 'iho', 'fishnet.shp')
    grid_gdf.to_file(output_shapefile)

    print(f"Fishnet shapefile saved as: {output_shapefile}")


    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original shapefile
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5, label="Original Shapefile")

    # Plot the fishnet grid
    grid_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='-', label="Fishnet Grid")
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    # Add title and legend
    ax.set_title('Fishnet Grid Overlayed on Original Shapefile')
    plt.savefig("rectangles_from_grid_points.png")




create_coverage(grid_size)