import os
import cartopy.crs     as ccrs
import cartopy.feature as cfeature

import matplotlib
matplotlib.use('TkAgg', force=True)
from matplotlib             import pyplot as plt
from skyfield.api           import load

class Plotting:
    #def __init__(self):
    #    self.fig = plt.figure(figsize=(10, 10))
    #    self.ax = self.fig.add_subplot(111, projection='3d')
    #    self.ax.set_xlabel('X')
    #    self.ax.set_ylabel('Y')
    #    self.ax.set_zlabel('Z')
    #    self.ax.set_xlim(-1, 1)
    #    self.ax.set_ylim(-1, 1)
    #    self.ax.set_zlim(-1, 1)
    #    self.ax.set_title('3D plot')
    #    self.ax.grid(True)
    #    self.ax.view_init(elev=30, azim=30)
    #    self.ax.legend()

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