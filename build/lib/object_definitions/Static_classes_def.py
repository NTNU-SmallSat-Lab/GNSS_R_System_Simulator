import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

import functions.functions as fn
from object_definitions.Constants_def import Constants


class Plotting:
    @staticmethod
    def plot_reflection_traces_gnss_const(
        rx_SatConst,
        tx_SatConst,
        reflection_vec,
        file_name="Reflection_traces.png",
        title=None,
    ) -> None:
        """
        Plot the reflection traces for a given GNSS constellation.

        INPUTS:
            rx_SatConst:     Satellite constellation object for the receiver (LEO) satellites.
            tx_SatConst:     Satellite constellation object for the transmitter (GNSS) satellites.
            reflection_vec:  List of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            file_name:       Name of the output file to save the plot (optional).
            title:           Title of the plot (optional).
        """
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_global()

        for i_sat in range(tx_SatConst.no_satellites):
            for i_gnss in range(rx_SatConst.no_satellites):
                color = fn.color_gnss_const(tx_SatConst.satConfigName)
                ax.scatter(
                    reflection_vec[i_sat * rx_SatConst.no_satellites + i_gnss].refl_LL_vec[:, 1],
                    reflection_vec[i_sat * rx_SatConst.no_satellites + i_gnss].refl_LL_vec[:, 0],
                    color=color,
                    s=1,
                    transform=ccrs.PlateCarree(),
                )

        # Plot reflection points
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        gridlines = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
        gridlines.top_labels = False  # Disable top labels
        gridlines.right_labels = False  # Disable right-side labels

        if title is None:
            title_string = f"Specular Reflection Points for {tx_SatConst.satConfigName} constellation and {rx_SatConst.no_satellites} LEO satellites"
        else:
            title_string = title
        plt.title(title_string)
        save_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(save_path, dpi=600)

    @staticmethod
    def plot_reflection_traces_total(
        rx_SatConst,
        tx_SatConstDict,
        reflection_dict,
        file_name="Reflection_traces_total.png",
        title=None,
    ):
        """
        Plot the reflection traces for all GNSS constellations.

        INPUTS:
            rx_SatConst:     Satellite constellation object for the receiver (LEO) satellites.
            tx_SatConstDict: Dictionary of satellite constellation objects for the transmitter (GNSS) satellites.
            reflection_dict: Dictionary of reflection vectors for the GNSS and GNSS-R (LEO) constellation.
            file_name:       Name of the output file to save the plot (optional).
            title:           Title of the plot (optional).
        """
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_global()

        # Loop through the reflection vectors for each GNSS constellation
        for gnss_key in tx_SatConstDict:
            tx_sat_const = tx_SatConstDict[gnss_key]
            reflections_vector = reflection_dict[gnss_key]
            col = fn.color_gnss_const(gnss_key)
            for i_sat in range(rx_SatConst.no_satellites):
                for i_gnss in range(tx_sat_const.no_satellites):
                    ax.scatter(
                        reflections_vector[i_sat * tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 1],
                        reflections_vector[i_sat * tx_sat_const.no_satellites + i_gnss].refl_LL_vec[:, 0],
                        color=col,
                        s=1,
                        transform=ccrs.PlateCarree(),
                    )
        #        ax.scatter(refl_gps.refl_LL_vec[:, 1], refl_gps.refl_LL_vec[:, 0], color='red', s=1, transform=ccrs.PlateCarree())

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        gridlines = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
        gridlines.top_labels = False  # Disable top labels
        gridlines.right_labels = False  # Disable right-side labels

        if title is None:
            constellations = ", ".join(tx_SatConstDict.keys())
            title_string = f"Specular Reflection Points for the constellations: {constellations} and {rx_SatConst.no_satellites} LEO satellites"
        else:
            title_string = title
        plt.title(title_string)

        # Add a legend
        legend_labels = [f"{gnss_key}" for gnss_key in tx_SatConstDict.keys()]
        legend_colors = [fn.color_gnss_const(gnss_key) for gnss_key in tx_SatConstDict.keys()]
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=color,
                markersize=5,
            )
            for label, color in zip(legend_labels, legend_colors, strict=False)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, title="Constellations")

        save_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(save_path, dpi=600)

    @staticmethod
    def plot_orbits(sat_const):
        """
        Plot the satellite orbits in 3D space with Earth as a sphere.
        Works with any number of satellites in the constellation.
        Includes equator line and polar axis for orientation.

        INPUT:
            sat_const: Satellite constellation object
        """
        # Get number of satellites
        num_satellites = len(sat_const.satellites)

        if num_satellites == 0:
            print("No satellites to plot.")
            return

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Earth radius in km
        earth_radius = Constants.R_EARTH_MEAN * 1e-3

        # Plot the Earth as a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = earth_radius * np.outer(np.cos(u), np.sin(v))
        y = earth_radius * np.outer(np.sin(u), np.sin(v))
        z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Render the Earth sphere with a light blue color
        ax.plot_surface(x, y, z, color="lightblue", alpha=0.3, edgecolor="none")

        # Add the equator (a circle in the x-y plane)
        theta = np.linspace(0, 2 * np.pi, 100)
        equator_x = earth_radius * np.cos(theta)
        equator_y = earth_radius * np.sin(theta)
        equator_z = np.zeros_like(theta)
        ax.plot(equator_x, equator_y, equator_z, "k--", linewidth=0.8, label="Equator")

        # Add the polar axis (a line through the north and south poles)
        polar_x = np.zeros(2)
        polar_y = np.zeros(2)
        polar_z = np.array([-earth_radius * 1.1, earth_radius * 1.1])  # Extend slightly beyond the Earth
        ax.plot(polar_x, polar_y, polar_z, "k--", linewidth=0.8, label="Polar Axis")

        # Add small markers for North and South poles
        ax.scatter(0, 0, earth_radius, color="red", s=30, label="North Pole")
        ax.scatter(0, 0, -earth_radius, color="blue", s=30, label="South Pole")

        # Create a color cycle for multiple satellites
        colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 distinct colors

        # Store all positions to calculate proper axis limits later
        all_positions = []

        # Plot each satellite trajectory
        for i, satellite in enumerate(sat_const.satellites):
            # Get color from the cycle (repeats if more than 10 satellites)
            color = colors[i % len(colors)]

            # Get the skyfield object and time vector
            satellite_skf = satellite.skf
            time_vec = satellite.time_prop_vec

            # Calculate ECI positions
            positions_ECI = np.array([satellite_skf.at(t).position.m for t in time_vec])
            positions_ECI = np.transpose(positions_ECI / 1e3)  # Convert to km

            # Store for axis limits calculation
            all_positions.append(positions_ECI)

            # Plot the trajectory
            ax.plot(
                positions_ECI[0],
                positions_ECI[1],
                positions_ECI[2],
                label=f"Satellite {i + 1}",
                color=color,
            )

        # Improve the appearance
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title(f"Orbits of {num_satellites} Satellites")

        # Only show satellite labels in legend if there's a reasonable number
        handles, labels = ax.get_legend_handles_labels()

        # Keep non-satellite elements (equator, polar axis, poles) and
        # add satellites only if there aren't too many
        if num_satellites <= 10:
            ax.legend()
        else:
            # Just show the key elements without individual satellites
            # Find the indices of non-satellite elements
            key_indices = [i for i, label in enumerate(labels) if "Satellite" not in label]
            ax.legend([handles[i] for i in key_indices], [labels[i] for i in key_indices])

        # Set equal aspect ratio to make the Earth look spherical
        # Calculate limits based on all satellite positions
        if all_positions:
            # Combine all position arrays
            all_x = np.concatenate([pos[0] for pos in all_positions])
            all_y = np.concatenate([pos[1] for pos in all_positions])
            all_z = np.concatenate([pos[2] for pos in all_positions])

            # Calculate the range for each dimension
            max_range = (
                np.max(
                    [
                        all_x.max() - all_x.min(),
                        all_y.max() - all_y.min(),
                        all_z.max() - all_z.min(),
                    ]
                )
                / 2.0
            )

            # Calculate the midpoints
            mid_x = (all_x.max() + all_x.min()) * 0.5
            mid_y = (all_y.max() + all_y.min()) * 0.5
            mid_z = (all_z.max() + all_z.min()) * 0.5

            # Set limits to ensure equal scale on all axes
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Save the plot
        save_path = os.path.join(Constants.output_folder_path, "Orbits.png")
        plt.savefig(save_path, dpi=600)
        print(f"Orbit plot saved to {save_path}")
        plt.close()
