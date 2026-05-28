import json

import matplotlib.pyplot as plt
import numpy as np


class SphericalToCartesian:
    def __init__(self, spherical_coordinates):
        '''
        Converts spherical coordinates to carthesian coordinates.

        Antenna radiation patterns use spherical coordinates when performing different calculations, but
        matplotlib must have carthesian coordinates to plot in 3D.

        Input: array object containing spherical coordinates of radiation pattern.
        Output: array object containing radiation pattern in corresponding carthesian coordinates.
        '''


        self.rho = 10**(spherical_coordinates/20)
        # self.rho = spherical_coordinates

        elevation = np.linspace(0, 180, spherical_coordinates.shape[0]) * np.pi / 180
        azimuth = np.linspace(0, 360, spherical_coordinates.shape[1]) * np.pi / 180

        self.elevation, self.azimuth = np.meshgrid(elevation, azimuth, indexing='ij')

    def calc(self):

            x = self.rho * np.sin(self.elevation) * np.cos(self.azimuth)
            y = self.rho * np.sin(self.elevation) * np.sin(self.azimuth)
            z = self.rho * np.cos(self.elevation)

            return x, y, z

class PlotRadiationPattern:
    def __init__(self, carthesian_coordinates, arrow = None):
        '''
        Plots radiation pattern in 3D. Must have carthesian coordinates as input.
        '''

        x = carthesian_coordinates[0]
        y = carthesian_coordinates[1]
        z = carthesian_coordinates[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, antialiased=False)

        if arrow is not None:
            r = np.max(np.abs(carthesian_coordinates))
            el = arrow[0] * np.pi / 180
            az = arrow[1] * np.pi / 180
            arr_x = r * np.sin(el) * np.cos(az)
            arr_y = r * np.sin(el) * np.sin(az)
            arr_z = r * np.cos(el)

            ax.quiver(0, 0, 0, arr_x, arr_y, arr_z, length=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.title('Antenna Radiation Pattern')
        plt.show()

def read_radpat_file(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # mesh = np.array(data['data'])
    # freq = data['frequency']
    # beta_m = data['beta_m']
    # beta_n = data['beta_n']
    return data


path = 'output/array_factor_1176.45MHz_90.00az_0.00el.json'

mydata = read_radpat_file(path)
mesh = np.array(mydata['data'])
freq = mydata['frequency']
az0 = mydata['azimuth_pointing']
el0 = mydata['elevation_pointing']

print(freq)
print(az0)
print(el0)


carthesian_object = SphericalToCartesian(mesh)
carthesian_radpat = carthesian_object.calc()

#calling plotting object
plotting_object = PlotRadiationPattern(carthesian_coordinates=carthesian_radpat, arrow = (el0, az0))
