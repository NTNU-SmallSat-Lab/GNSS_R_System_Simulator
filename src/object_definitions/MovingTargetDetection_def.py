from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
import datetime as dt
from datetime import timedelta
import logging
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import imageio
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for off-screen rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity
from tqdm import tqdm

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.Constants_def import Constants
from object_definitions.Reflection_def import Reflection
from object_definitions.BackScattPoint_def import BRP as Backscatter
from object_definitions.ShipTrajectory_def import ShipTrajectory

logger = logging.getLogger(__name__)


class MovingTargetDetection:
    def __init__(
        self,
        sat_const,
        gnss_const,
        moving_target_object,
        antenna_obj,
        create_shapefile_for_visualization=False,
        deltaT=60,
        earth_model="spherical",
        reflection_mode="specular",  # <--- add this argument
        link_budget_settings=None,
        bbox_buffer_deg=2.0,
        max_central_angle_deg=80.0,
        max_incidence_angle_deg=45.0,
    ) -> None:
        """
        =========================================================================================================
        MovingTargetDetection class:
            - Determine when reflection points (specular or backscatter) intersect a moving target's
              trajectory, and accumulate per-time-step detection events.

            INPUTS:
                sat_const:                          [SatelliteConstellation]   LEO Rx satellite constellation
                gnss_const:                         [dict]                     {name: SatelliteConstellation} of GNSS Tx constellations
                moving_target_object:               [ShipTrajectory]           Moving target with a populated shapefile path
                antenna_obj:                        [RxAntenna]                Receiver antenna
                create_shapefile_for_visualization: [bool]                     Generate visualization shapefile (default: False)
                deltaT:                             [int]                      Time tolerance [s] for matching reflections (default: 60)
                earth_model:                        [str]                      "spherical" or "elliptical"
                reflection_mode:                    [str]                      "specular" or "backscatter"
                link_budget_settings:               [dict | BistaticLinkBudgetSettings | None]   Link budget settings
                bbox_buffer_deg:                    [float]                    Padding in degrees added around the trajectory bounding box when rendering detection plots (default: 2.0)
                max_central_angle_deg:              [float]                    Max Tx-Rx central angle [deg] (default: 80.0)
                max_incidence_angle_deg:            [float]                    Max BRP incidence angle [deg], backscatter only (default: 45.0)

            ATTRIBUTES:
                moving_target_object:        [ShipTrajectory]   Moving target object
                gdf_time_series:             [gpd.GeoDataFrame] GeoDataFrame for the input time-series shapefile
                bounding_box:                [tuple]            (min_lon, max_lon, min_lat, max_lat)
                reflection_dict_timeSorted:  [dict]             Reflection points indexed by time, per GNSS constellation
                target_detect_dict:          [dict]             Per-constellation detection accumulators
                sat_antenna:                 [RxAntenna]        Receiver antenna
                sat_constellation:           [SatelliteConstellation]   LEO constellation
                gnss_constellation:          [dict]             GNSS constellations
        =========================================================================================================
        """
        if isinstance(moving_target_object, ShipTrajectory):
            self.moving_target_object = moving_target_object
            if moving_target_object.shapefile_path["time_series"] is None:
                raise ValueError(
                    'ERROR: Shapefile path is None in "moving target object". \n Please provide an moveing_target_object with a valid shapefile path.'
                )
            self.gdf_time_series = gpd.read_file(
                moving_target_object.shapefile_path["time_series"]
            )  # GeoDataFrame for the input shapefile (outline of the area of interest)

        self.bounding_box = self._get_bounding_box(buffer_meters=1000)  # Bounding box for filtering points (min_lon, max_lon, min_lat, max_lat)
        self.reflection_dict_timeSorted = {}  # Reflection dictionary for storing reflection vector containing reflection points for each GNSS constellations
        self.target_detect_dict = {}  # Coverage vector for storing coverage information over each grid-cell in the gridded shapefile
        self.sat_antenna = antenna_obj  # Antenna object for the satellite (RxAntenna)

        self.sat_constellation = sat_const  # Satellite constellation object
        self.gnss_constellation = gnss_const  # Dictionary with GNSS constellation objects

        self.deltaT_sec = deltaT  # Time tolerance in seconds for matching reflection points to moving target positions (default: 60 seconds) | This should match the time step of the simulation
        self.earth_model = earth_model  # Earth model to use for the reflection calculation (default: 'spherical')
        self.reflection_mode = reflection_mode  # Either Specular reflection, or backscattering calculation
        self.link_budget_settings = link_budget_settings
        self.link_budget_frequency_hz = lb.resolve_single_link_budget_frequency_hz(self.sat_antenna)
        self.bbox_buffer_deg = float(bbox_buffer_deg)
        self.max_central_angle_deg = float(max_central_angle_deg)
        self.max_incidence_angle_deg = float(max_incidence_angle_deg)
        self.footprint_stats = {"total": 0, "rejected_empty": 0}  # populated by _reflection_point_footprint, summarised at the end of calculate_moving_target_reflection_intersection
        self.initialization()

    def initialization(self) -> None:
        ####################################################################################################################
        # Initialize coverage vector for storing coverage over each grid-cell in the gridded shapefile                     #
        # Make a list of moving_target_detection_struct instanced with the length no_trajectory_points                     #
        ####################################################################################################################
        for gnss_name, _ in self.gnss_constellation.items():
            self.target_detect_dict[gnss_name] = []

    def calculate_moving_target_reflection_intersection(self, verbose=False) -> None:
        """
        =========================================================================================================
        Calculate the intersection of the moving target trajectory with the reflection points.

        A moving target is considered "detected" if the reflection point (or its footprint) overlaps
        the target's location within `self.deltaT_sec` of the matching trajectory timestamp.

        INPUTS:
            verbose: [bool]   Print per-constellation progress (default: False)
        =========================================================================================================
        """

        def show_reflection_and_target_position(pos_tgt_i: pd.Series, footprint_reflection: Polygon, spec_refl_point: np.ndarray, gnss_name: str) -> None:
            """
            =========================================================================================================
            Debug plot: one reflection point and the target position at a given time.

            INPUTS:
                pos_tgt_i:            [pd.Series]    Target position row (with geometry)
                footprint_reflection: [Polygon]      Footprint polygon at this time step
                spec_refl_point:      [np.ndarray]   Specular reflection point [lat, lon]
                gnss_name:            [str]          Name of the GNSS constellation
            =========================================================================================================
            """
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})

            buffer_deg = self.bbox_buffer_deg
            min_lon, max_lon, min_lat, max_lat = self.bounding_box
            ax.set_extent([min_lon - buffer_deg, max_lon + buffer_deg, min_lat - buffer_deg, max_lat + buffer_deg], crs=ccrs.PlateCarree())

            ax.set_title("GNSS-R Reflection and Target Position")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.add_feature(cfeature.LAKES, facecolor="lightblue")

            # Replace the problematic plotting section with this:
            if isinstance(footprint_reflection, Polygon):
                # Handle single Polygon
                ax.plot(
                    *footprint_reflection.exterior.xy,
                    color="red",
                    linewidth=2,
                    label="Footprint -3dB",
                    transform=ccrs.PlateCarree(),
                )
            elif isinstance(footprint_reflection, MultiPolygon):
                # Handle MultiPolygon - plot each constituent polygon
                for i, poly in enumerate(footprint_reflection.geoms):
                    label = "Footprint -3dB" if i == 0 else ""  # Only label the first polygon
                    ax.plot(
                        *poly.exterior.xy,
                        color="red",
                        linewidth=2,
                        label=label,
                        transform=ccrs.PlateCarree(),
                    )
            else:
                logger.warning(f"Unexpected geometry type for footprint: {type(footprint_reflection)}")

            ax.plot(
                pos_tgt_i.geometry.x,
                pos_tgt_i.geometry.y,
                "o",
                color="blue",
                markersize=8,
                label="Target Position",
                transform=ccrs.PlateCarree(),
            )
            ax.plot(
                spec_refl_point[1],
                spec_refl_point[0],
                ".",
                color="orange",
                markersize=8,
                label="Specular Reflection",
                transform=ccrs.PlateCarree(),
            )

            # Print the coordinates to check if they're inside the bounding box
            # Print the coordinates to check if they're inside the bounding box
            if isinstance(footprint_reflection, Polygon):
                lon_coords = footprint_reflection.exterior.xy[0]
                lat_coords = footprint_reflection.exterior.xy[1]
                logger.debug(
                    f"Circle coords range: Lon [{min(lon_coords)}-{max(lon_coords)}], "
                    f"Lat [{min(lat_coords)}-{max(lat_coords)}]"
                )
            elif isinstance(footprint_reflection, MultiPolygon):
                # For MultiPolygon, get bounds of the entire geometry
                bounds = footprint_reflection.bounds  # (min_lon, min_lat, max_lon, max_lat)
                logger.debug(
                    f"MultiPolygon coords range: Lon [{bounds[0]}-{bounds[2]}], "
                    f"Lat [{bounds[1]}-{bounds[3]}]"
                )
            else:
                logger.debug(f"Geometry bounds: {footprint_reflection.bounds}")
            logger.debug(f"Bounding box: {self.bounding_box}")
            logger.debug(f"Reflection point: Lon {spec_refl_point[0]:.6f}, Lat {spec_refl_point[1]:.6f}")

            # Add grid lines for better reference
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

            ax.legend()
            plt.tight_layout()
            plt.show(block=True)

        ###########################################
        # 0: Initializations                      #
        ###########################################

        LightweightRef = namedtuple('LightweightRef', ['refl_obj', 'time_index', 'refl_id'])
        ###########################################
        # 1: Calculate all the reflection points  #
        ###########################################
        # Get the satellite constellation times in UTC format
        # Reflection times are identical to the satellite constellation times
        times_reflections = [t.utc_datetime() for t in self.sat_constellation.satellites[0].time_prop_vec]
        times_refl_array = np.array(times_reflections)
        times_reflections_sec = np.array([t.timestamp() for t in times_reflections])

        # Calculate the total number of reflection points for all GNSS constellations (for the progress bar)
        total_comp_refl = sum(
            len(self.sat_constellation.satellites) * len(gnss_const.satellites)
            for gnss_const in self.gnss_constellation.values()
        )
        reflection_dict = {}  # Dictionary to store the reflection points for each GNSS constellation

        # Loop through all GNSS computations and calculate all the reflection points
        with tqdm(total=total_comp_refl, desc="Reflection calculations", unit="reflection") as pbar:
            # Loop through each GNSS constellations
            for gnss_name, gnss_const in self.gnss_constellation.items():
                if verbose:
                    logger.info("-----------------------------------------------------------------")
                    logger.info(f'Calculating reflections for GNSS constellation "{gnss_name}"')
                reflections_vec = []

                # Loop through the GNSS-R satellites in the GNSS-R LEO constellation / Rx
                for sat_no, sat_rx in enumerate(self.sat_constellation.satellites):
                    # Update progress bar description (not advancing)
                    pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {gnss_name}")

                    # Loop through the GNSS satellites in the GNSS constellation / Tx
                    for i, sat_tx in enumerate(gnss_const.satellites):
                        if i % 5 == 0:
                            # Update progress bar description (only every 5th iteration)
                            pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {gnss_name} [{i}/{len(gnss_const.satellites)}]")

                        if self.reflection_mode == "specular":
                            reflection = Reflection(
                                sat_tx.satID,
                                sat_rx.satID,
                                earth_model=self.earth_model,
                                max_central_angle_deg=self.max_central_angle_deg,
                                gnss_const=gnss_name,
                            )
                            reflection.compute_reflection_coordinates(
                                sat_tx,
                                sat_rx,
                                times_refl_array,
                                antenna=self.sat_antenna,
                                frequency_hz=self.link_budget_frequency_hz,
                                link_budget_settings=self.link_budget_settings,
                            )
                            reflections_vec.append(reflection)
                        elif self.reflection_mode == "backscatter":
                            reflection = Backscatter(
                                sat_tx.satID,
                                sat_rx.satID,
                                earth_model=self.earth_model,
                                max_central_angle_deg=self.max_central_angle_deg,
                                max_incidence_angle_deg=self.max_incidence_angle_deg,
                                gnss_const=gnss_name,
                            )
                            reflection.compute_brp_coordinates(
                                sat_tx,
                                sat_rx,
                                times_refl_array,
                                antenna=self.sat_antenna,
                                frequency_hz=self.link_budget_frequency_hz,
                                link_budget_settings=self.link_budget_settings,
                            )
                            reflections_vec.append(reflection)

                        # Update progress bar after each reflection calculation
                        pbar.update(1)

                # Store the reflection points in the reflection dictionary
                reflection_dict[gnss_const.satConfigName] = reflections_vec

        ################################################################
        # 2: Re-sort reflections_dict to reflection_dict_timeSorted    #
        ################################################################
        for gnss_const_name, reflection_objects in reflection_dict.items():
            # Add the gnss_constallation name to reflection_dict_timeSorted (initialize for this constellation)
            time_sorted_dict = self.reflection_dict_timeSorted[gnss_const_name] = defaultdict(list)

            # Track refl_ids per time point for O(1) duplicate checking
            refl_ids_per_time = defaultdict(set)

            # Iterate through each reflection object for this constellation
            for refl_obj in tqdm(
                reflection_objects,
                desc=f"Sorting reflections for {gnss_const_name}",
                unit="reflection",
            ):
                # Direct attribute access (faster than getattr) - use try/except for safety
                try:
                    time_vector_all_refl = refl_obj.time_vec
                except AttributeError:
                    logger.warning(f"Reflection object missing time_vec attribute for {gnss_const_name}")
                    continue

                # Single check for None or empty (removed duplicate check)
                if time_vector_all_refl is None or len(time_vector_all_refl) == 0:
                    logger.warning(f"Empty or None time vector in reflection object for {gnss_const_name}")
                    continue

                # Cache satellite IDs and generate reflection ID once
                tx_satID = refl_obj.tx_satID
                rx_satID = refl_obj.rx_satID
                refl_id = (tx_satID, rx_satID)

                # Process times on-the-fly
                invalid_count = 0
                for idx, time_i in enumerate(time_vector_all_refl):
                    # Skip None or invalid times [ROBUSTNESS]
                    if time_i is None or not isinstance(time_i, dt.datetime):
                        invalid_count += 1
                        continue

                    # O(1) duplicate check using set (cache the set reference)
                    time_refl_ids = refl_ids_per_time[time_i]
                    if refl_id in time_refl_ids:
                        raise ValueError(f"Duplicate reflection ID {refl_id} found for time {time_i} in {gnss_const_name}")

                    # Add to tracking set
                    time_refl_ids.add(refl_id)

                    # Create lightweight reference using namedtuple (much faster than dict)
                    lightweight_ref = LightweightRef(
                        refl_obj=refl_obj,
                        time_index=idx,
                        refl_id=refl_id
                    )

                    # Append the reference to the list for this time
                    time_sorted_dict[time_i].append(lightweight_ref)

                # Report invalid times if any
                if invalid_count > 0:
                    logger.warning(f"{invalid_count} invalid times in reflection object for {gnss_const_name}")

        #########################################################
        # 3: Calculate if the moving target can be detected     #
        #########################################################
        # Iterate through each satellite constellation
        for gnss_name, _ in self.gnss_constellation.items():
            # Iterate through each position in the resampled moving target trajectory
            for idx, (time_tgt_i, pos_tgt_i) in tqdm(
                enumerate(zip(self.timestamps_resampled, self.ship_positions_resampled)),
                desc=f"Detection of moving target using {gnss_name}",
                unit=" time-step ",
                total=len(self.timestamps_resampled),
                leave=False,
            ):
                # Get the time for the current position of the moving target / convert to UTC format
                time_tgt_i = time_tgt_i.replace(tzinfo=dt.UTC)
                ###############################################################################################################
                # Find the closest time in the reflection trajectory matching the current time at the moving target position  #
                ###############################################################################################################
                matching_time_found = False
                time_refl_i = None
                # Iterate through the satellite-constellation times | Make sure times_reflections is not empty [ROBUSTNESS]
                if times_reflections:
                    # Get the closest time
                    matching_idx, min_diff = self._find_closest_time(time_tgt_i, times_reflections_sec)
                    time_refl_i = times_reflections[matching_idx]
                    if min_diff <= self.deltaT_sec:
                        matching_time_found = True

                # Check if the time is in the reflection dictionary (tolerance of 1 second)
                if matching_time_found:
                    # Get the reflection points for this time for the current GNSS constellations
                    refl_points_i = self.reflection_dict_timeSorted[gnss_name][time_refl_i]

                    ######################################################################################################
                    # Check if a reflection moving target at (time_refl_i) is within the "footprint" of the antenna      #
                    ######################################################################################################
                    for reflection in refl_points_i:
                        # Get the reflection object
                        refl_obj = reflection.refl_obj

                        # Get the actuall reflection coordinates for the current time
                        reflection_index = reflection.time_index  # Use the pre-computed index
                        coordinates_reflection = refl_obj.refl_ECI_vec[reflection_index]
                        if not np.all(np.isnan(coordinates_reflection)):
                            lon = refl_obj.refl_LL_vec[reflection_index][1]  # Get longitude (second element)
                            lat = refl_obj.refl_LL_vec[reflection_index][0]  # Get latitude (first element)
                            if np.isnan(lon) or np.isnan(lat):
                                if verbose:
                                    tqdm.write(
                                        f"INFORMATION: Reflection coordinates are NaN for {gnss_name}, satellites ID-GNSS {refl_obj.tx_satID}, and ID-GNSS-R {refl_obj.rx_satID} at time {time_refl_i}"
                                    )
                                continue
                            # Create an ellipse around the reflection point
                            for freq in self.sat_antenna.frequencies:
                                footprint_reflection, spec_refl_point = self._reflection_point_footprint(refl_obj, reflection_index, freq, verbose)

                                ## DEBUG: Plot the footprint and the moving target position
                                # show_reflection_and_target_position(pos_tgt_i, footprint_reflection, spec_refl_point, gnss_name)
                                if footprint_reflection.intersects(pos_tgt_i):
                                    # Moving Target could be detected! Initialize the moving target detection structure
                                    possible_target_detection = MovingTargetDetectionStruct()
                                    possible_target_detection.position_eci = refl_obj.refl_ECI_vec[reflection_index]
                                    possible_target_detection.position_ll = refl_obj.refl_LL_vec[reflection_index]
                                    possible_target_detection.time = time_tgt_i
                                    possible_target_detection.footprint_shape = footprint_reflection
                                    possible_target_detection.spec_refl_point = spec_refl_point  # Specular reflection point
                                    possible_target_detection.refl_obj = refl_obj
                                    possible_target_detection.frequency = freq
                                    possible_target_detection.pos_tgt_i = pos_tgt_i
                                    possible_target_detection.tx_satID = refl_obj.tx_satID
                                    possible_target_detection.rx_satID = refl_obj.rx_satID

                                    # Append the possible target detection to the target detection list
                                    self.target_detect_dict[gnss_name].append(possible_target_detection)
                elif time_refl_i is not None:
                    time_diff_sec = (time_refl_i - time_tgt_i).total_seconds()
                    tqdm.write(
                        f"WARNING: No matching time found for target time {time_tgt_i.strftime('%Y-%m-%d %H:%M:%S')} for the {gnss_name} constellation\n     Closest time is {time_refl_i.strftime('%Y-%m-%d %H:%M:%S')} [deltaT = {time_diff_sec} seconds]\n     -> Make sure skyfield times and moving target times are overlapping."
                    )
                else:
                    tqdm.write(
                        f"WARNING: No reflection times available for the {gnss_name} constellation at target time {time_tgt_i.strftime('%Y-%m-%d %H:%M:%S')}."
                    )

        total = self.footprint_stats["total"]
        rejected = self.footprint_stats["rejected_empty"]
        if total > 0:
            pct = rejected / total * 100.0
            logger.info(f"Footprint computations: {rejected}/{total} rejected as empty ({pct:.1f}%)")

    def show_reflections_and_target_position_at_time_i(
        self,
        time_i,
        gnss_constellation_name=None,
        file_name=None,
        label_reflections=True,
        downscale_factor=1,
        show=False,
        verbose=False,
    ) -> None:
        """
        =========================================================================================================
        Plot all reflection points and the target position at a given time.

        INPUTS:
            time_i:                  [datetime]    Time of the target position
            gnss_constellation_name: [str | None]  Constellation name to plot (default: all loaded)
            file_name:               [str | None]  Output filename; if None, returns an image array
            label_reflections:       [bool]        Annotate reflections with sat IDs (default: True)
            downscale_factor:        [float]       Figure downscale factor when saving (default: 1)
            show:                    [bool]        Display interactively (default: False)
            verbose:                 [bool]        Print warnings (default: False)
        =========================================================================================================
        """
        if gnss_constellation_name is None:
            # If no constellation name is provided, show all constellations
            gnss_constellation_name = list(self.reflection_dict_timeSorted.keys())
        else:
            # Check if the constellation name is valid
            if gnss_constellation_name not in self.reflection_dict_timeSorted:
                raise ValueError(f'ERROR: GNSS constellation "{gnss_constellation_name}" not found in reflection dictionary.')

        # Get the reflection points for the given time
        # Find the closest time in the reflection trajectory matching the current time at the moving target position
        reflections_time_i = {}
        for gnss_const in gnss_constellation_name:
            # Check if the time is in the reflection dictionary (tolerance of 1 second)
            if time_i not in self.reflection_dict_timeSorted[gnss_const]:
                time_i_ = min(
                    self.reflection_dict_timeSorted[gnss_const].keys(),
                    key=lambda k: abs(k.timestamp() - time_i.timestamp()),
                )
            else:
                time_i_ = time_i
            reflections_time_i[gnss_const] = self.reflection_dict_timeSorted[gnss_const].get(time_i_, None)
            if reflections_time_i[gnss_const] is None:
                if verbose:
                    logger.warning(f"No reflections found for time {time_i} in constellation {gnss_const}")
                return

        # Get the target position at time_i
        # Check if the time is in the gdf_time_series
        gdf_time_series_dt = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC) for d in self.gdf_time_series["datetime_s"].values]

        if time_i not in gdf_time_series_dt:
            time_i_dt = min(
                gdf_time_series_dt,
                key=lambda k: abs(k.timestamp() - time_i.timestamp()),
            )
            time_i_str = time_i_dt.strftime("%Y-%m-%d %H:%M:%S")
            pos_tgt_i = self.gdf_time_series.loc[self.gdf_time_series["datetime_s"] == time_i_str]
        else:
            # Get the target position at time_i
            time_i_str = time_i.strftime("%Y-%m-%d %H:%M:%S")
            pos_tgt_i = self.gdf_time_series.loc[self.gdf_time_series["datetime_s"] == time_i_str]

        # Plot the target position and the reflection points with footprints
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        # Set the map extent to your bounding box (order: west, east, south, north)
        
        #ax.set_global()
        buffer_deg = self.bbox_buffer_deg
        min_lon, max_lon, min_lat, max_lat = self.bounding_box
        ax.set_extent([min_lon - buffer_deg, max_lon + buffer_deg, min_lat - buffer_deg, max_lat + buffer_deg], crs=ccrs.PlateCarree())

        ax.set_title(f"Reflections and Target Position at {time_i}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.7)
        ax.stock_img()
        # Add grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        # Plot the trajectory points
        trajectory_x = [point[1] for point in self.moving_target_object.trajectory_points]
        trajectory_y = [point[0] for point in self.moving_target_object.trajectory_points]
        ax.plot(
            trajectory_x,
            trajectory_y,
            "-.",
            color="orange",
            linewidth=1.5,
            alpha=0.7,
            label="Trajectory",
        )
        # Add a legend (handle duplicates properly)
        if label_reflections:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles, strict=False))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
        # Plot the target position
        if not pos_tgt_i.empty:
            pos_tgt_i = pos_tgt_i.iloc[0]
            ax.plot(
                pos_tgt_i.geometry.x,
                pos_tgt_i.geometry.y,
                "x",
                color="orange",
                markersize=8,
                label="Target Position",
            )
            # Add a label for the target position
            ax.text(
                pos_tgt_i.geometry.x,
                pos_tgt_i.geometry.y,
                "Target Position",
                fontsize=8,
                ha="right",
                va="bottom",
                color="orange",
            )
        else:
            if verbose:
                logger.warning(f"No target position found for time {time_i} in the trajectory data.")
            return
        # Plot the reflection points and their footprints
        for gnss_const, reflections in reflections_time_i.items():
            if reflections is not None:
                for reflection in reflections:
                    # Get the reflection object
                    refl_obj = reflection.refl_obj
                    # Get the actual reflection coordinates for the current time
                    reflection_index = reflection.time_index
                    coordinates_reflection = refl_obj.refl_ECI_vec[reflection_index]
                    if not np.all(np.isnan(coordinates_reflection)):
                        lon = refl_obj.refl_LL_vec[reflection_index][1]
                        lat = refl_obj.refl_LL_vec[reflection_index][0]
                        # Skip points outside the map extent
                        if lon < west or lon > east or lat < south or lat > north:
                            continue
                        if np.isnan(lon) or np.isnan(lat):
                            # If the coordinates are NaN, skip this reflection point
                            if verbose:
                                logger.warning(
                                    f"Reflection coordinates are NaN for {gnss_const}, satellites ID-GNSS {refl_obj.tx_satID}, and ID-GNSS-R {refl_obj.rx_satID} at time {time_i}"
                                )
                            continue

                        # Plot the reflection point
                        ax.scatter(
                            lon,
                            lat,
                            c=fn.color_gnss_const(gnss_const),
                            s=30,
                            marker="o",
                            alpha=0.7,
                            label=(f"{gnss_const} Reflection" if reflection == reflections[0] else ""),
                        )
                        # Add a label for the reflection point (if enabled)
                        if label_reflections:
                            ax.text(
                                lon,
                                lat,
                                f"{gnss_const} sat-ID {refl_obj.tx_satID}, GNSS-R sat. {refl_obj.rx_satID}",
                                fontsize=8,
                                ha="right",
                                va="bottom",
                                color=fn.color_gnss_const(gnss_const),
                            )
                        for frequency in self.sat_antenna.frequencies:
                            # Create the footprint (ellipse) around the reflection point
                            footprint_reflection, _ = self._reflection_point_footprint(refl_obj, reflection_index, frequency, verbose)

                            # Safe footprint plotting
                            success = self._plot_footprint(
                                ax, footprint_reflection,
                                color=fn.color_gnss_const(gnss_const),
                                linewidth=1, alpha=0.5,
                                transform=ccrs.PlateCarree()
                            )
                            if not success and verbose:
                                logger.warning(f"Skipped invalid footprint for {gnss_const} at ({lon:.2f}, {lat:.2f})")

        # Add a legend (handle duplicates properly)
        if label_reflections:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles, strict=False))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
        if show:
            plt.show(block=True)

        # Save the figure if the file name is provided
        if file_name is not None:
            file_path = os.path.join(Constants.output_folder_path, file_name)
            # Ensure output directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Downscale the figure size if needed
            if downscale_factor > 1:
                fig.set_size_inches(fig.get_size_inches() / downscale_factor)
            # Save the figure
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return None  # nothing to return when saving to file
        else:
            # Convert figure to numpy array for in-memory animation
            fig.canvas.draw()  # ensure rendering
            buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            buffer = buffer.reshape((h, w, 4))  # ARGB format

            # Convert ARGB → RGB
            img_array = np.roll(buffer, -1, axis=2)[:, :, :3]
            plt.close(fig)
            return img_array
        
    def create_reflection_animation(self, time_step_min=3, output_format="gif", output_filename=None, fps=3):
        """
        =========================================================================================================
        Create an animation showing reflections over time. Supports GIF or video (MP4/AVI) output.

        INPUTS:
            time_step_min:   [float]        Time step between frames [min] (default: 3)
            output_format:   [str]          'gif', 'mp4', or 'avi' (default: 'gif')
            output_filename: [str | None]   Output filename (default: timestamped)
            fps:             [int]          Frames per second (default: 3)
        =========================================================================================================
        """
        logger.info("Creating an animation of the moving target and the reflections")

        # Generate time points
        time_points = []
        current_time = self.moving_target_object.start_time
        end_time = self.moving_target_object.end_time
        while current_time <= end_time:
            time_points.append(current_time)
            current_time += timedelta(minutes=time_step_min)

        logger.info(f"Creating {len(time_points)} frames...")

        # Generate frames in memory
        frames = []
        for time_point in tqdm(time_points):
            try:
                img_array = self.show_reflections_and_target_position_at_time_i(
                    time_i=time_point,
                    file_name=None,  # return image instead of saving
                    show=False,
                    label_reflections=False,
                )
                if img_array is not None:
                    frames.append(img_array)
            except Exception as e:
                logger.warning(f"Skipping frame at {time_point} due to error: {e}")
                continue

        # Create output filename
        if output_filename is None:
            timestamp = self.moving_target_object.start_time.strftime("%Y%m%d_%H%M")
            output_filename = f"animation_{timestamp}.{output_format}"

        output_path = os.path.join(Constants.output_folder_path, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Creating {output_format} at {output_path}...")

        # Save as MP4 or GIF
        if output_format.lower() in ["mp4", "avi"]:
            # Video writer
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        else:
            # GIF
            imageio.mimsave(output_path, frames, fps=fps)

        logger.info(f"Animation created at: {output_path}")

    def _find_closest_time(self, target_time, times_reflections_sec) -> tuple:
        """
        =========================================================================================================
        Find the closest time using vectorized operations.

        INPUTS:
            target_time:            [datetime]     Reference timestamp (UTC-aware or naive UTC)
            times_reflections_sec:  [np.ndarray]   Reference timestamps as float Unix seconds

        OUTPUT:
            result: [tuple]   (matching_idx, time_diff_seconds)
        =========================================================================================================
        """
        target_sec = fn.ensure_utc_aware(target_time).timestamp()
        time_diffs = np.abs(times_reflections_sec - target_sec)
        matching_idx = int(np.argmin(time_diffs))

        return matching_idx, time_diffs[matching_idx]

    def _get_bounding_box(self, buffer_meters=0.0) -> tuple:
        """
        =========================================================================================================
        Calculate the bounding box of the maximal/minimal movement of the moving target.

        INPUTS:
            buffer_meters: [float]   Optional padding around the bounding box [m]

        OUTPUT:
            bbox: [tuple]   (min_lon, max_lon, min_lat, max_lat)
        =========================================================================================================
        """
        # Get the bounding box of the gridded shapefile
        latitudes = [coords[0] for coords in self.moving_target_object.waypoints]
        longitudes = [coords[1] for coords in self.moving_target_object.waypoints]
        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)

        if buffer_meters > 0:
            # Convert buffer from meters to degrees (approximate conversion)
            deg_lat_at_equator = Constants.RADIUS_EARTH_EQUAT * 2 * np.pi / 360  # meters per degree latitude at the equator
            lat_buffer = buffer_meters / deg_lat_at_equator
            lon_buffer = buffer_meters / (deg_lat_at_equator * np.cos(np.radians((min_lat + max_lat) / 2)))

            min_lat -= lat_buffer
            max_lat += lat_buffer
            min_lon -= lon_buffer
            max_lon += lon_buffer

        # Create a bounding box tuple (min_lon, max_lon, min_lat, max_lat)
        return (min_lon, max_lon, min_lat, max_lat)

    def _reflection_point_footprint(self, refl_obj: Reflection | Backscatter, i: int, frequency: float, verbose: bool = False) -> tuple[Polygon, np.ndarray]:
        """
        =========================================================================================================
        Compute the antenna footprint polygon at sample `i` of a reflection object, with cleanup
        for invalid/multi-polygon results.

        INPUTS:
            refl_obj:  [Reflection | BRP]   Reflection object
            i:         [int]                Sample index
            frequency: [float]              Carrier frequency [MHz]
            verbose:   [bool]               Print warnings (default: False)

        OUTPUT:
            result: [tuple]   (footprint_polygon, specular_reflection_point) — empty Polygon on failure
        =========================================================================================================
        """

        def show_footprint(footprint_poly: Polygon) -> None:
            """
            =========================================================================================================
            Debug plot of the footprint polygon on a map.

            INPUTS:
                footprint_poly: [Polygon]   Footprint polygon to display
            =========================================================================================================
            """
            ## DEBUG plot footprint_poly
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})

            # Make sure bounding box is appropriate for your circle
            # If self.bounding_box is too large compared to your circle, adjust it
            # Format should be [lon_min, lon_max, lat_min, lat_max]
            max_lon, min_lon, max_lat, min_lat = self.bounding_box
            max_lon = max(max_lon, footprint_poly.bounds[2]) + 0.1
            min_lon = min(min_lon, footprint_poly.bounds[0]) - 0.1
            max_lat = max(max_lat, footprint_poly.bounds[3]) + 0.1
            min_lat = min(min_lat, footprint_poly.bounds[1]) - 0.1
            new_bounding_box = (min_lon, max_lon, min_lat, max_lat)
            ax.set_extent(new_bounding_box, crs=ccrs.PlateCarree())

            ax.set_title("GNSS-R Reflection and Target Position")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.add_feature(cfeature.LAKES, facecolor="lightblue")

            # Try making the circle more visible
            ax.plot(
                *footprint_poly.exterior.xy,
                color="red",
                linewidth=2,
                label="Footprint Polygon",
                transform=ccrs.PlateCarree(),
            )
            center_point = footprint_poly.centroid
            ax.plot(
                center_point.x,
                center_point.y,
                "o",
                color="blue",
                markersize=8,
                label="Polygon Center",
                transform=ccrs.PlateCarree(),
            )

            # Log the coordinates to check if they're inside the bounding box
            logger.debug(
                f"Circle coords range: Lon [{min(footprint_poly.exterior.xy[0])}-{max(footprint_poly.exterior.xy[0])}], "
                f"Lat [{min(footprint_poly.exterior.xy[1])}-{max(footprint_poly.exterior.xy[1])}]"
            )
            logger.debug(f"Bounding box: {self.bounding_box}")

            # Add grid lines for better reference
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

            ax.legend()
            plt.tight_layout()
            plt.show(block=True)

        def check_nan_coords(polygon) -> bool:
            # Get exterior coordinates
            if isinstance(polygon, Polygon):
                exterior_coords = np.array(polygon.exterior.coords)
                return np.isnan(exterior_coords).any()
            elif isinstance(polygon, MultiPolygon):
                # Check each polygon in the MultiPolygon
                for poly in polygon.geoms:
                    exterior_coords = np.array(poly.exterior.coords)
                    if np.isnan(exterior_coords).any():
                        return True
                return False
            else:
                # Handling of other types if necessary
                if verbose:
                    logger.warning(f"Unsupported geometry type {type(polygon)} for NaN check.")
                return True # Assume NaN if unsupported type

        self.footprint_stats["total"] += 1

        if not np.isfinite(refl_obj.power[i]):
            self.footprint_stats["rejected_empty"] += 1
            return Polygon(), np.array([])

        footprint_points, spec_refl_point = refl_obj._compute_footprint(i, self.sat_antenna, frequency, earth_model=self.earth_model)
        if not footprint_points.is_valid:
            if verbose:
                logger.warning(f"Invalid footprint for reflection index {i}. {explain_validity(footprint_points)}")
            footprint_points = footprint_points.buffer(0)  # Fix invalid geometry if necessary
        if isinstance(footprint_points, MultiPolygon):
            # Try to merge the polygons into one (MultiPolygon -> Polygon)
            merged = unary_union(footprint_points)
            if isinstance(merged, MultiPolygon):
                # If merging didnt work take the convex hull of the MultiPolygon (hull around all polygons)
                merged = merged.convex_hull
            if isinstance(merged, Polygon):
                footprint_points = merged
            else:
                if verbose:
                    logger.warning(f"Merging multipoligons for reflection index {i} not successful. Using largest of the MultiPolygon!")
                # If both method above didnt work take the largest polygon
                footprint_points = max(footprint_points.geoms, key=lambda p: p.area)

        # Robustness: check if the points are nan
        if check_nan_coords(footprint_points):
            if verbose:
                logger.warning(f"Ellipse points contain NaN values for reflection index {i} in {refl_obj.gnss_const}. Skipping footprint creation. The satellite might be too far away or the footprint size is too large.")
            self.footprint_stats["rejected_empty"] += 1
            return Polygon(), spec_refl_point

            ## DEBUG plot footprint_poly
            # show_footprint(footprint_points)

        # Create a shapely polygon from the points
        if footprint_points.is_empty:
            self.footprint_stats["rejected_empty"] += 1
        return footprint_points, spec_refl_point

    def _fix_longitude_wrapping(self, polygon):
        """
        =========================================================================================================
        Fix polygon longitude wrapping and validate size based on latitude.

        INPUTS:
            polygon: [Polygon | MultiPolygon | None]   Polygon to fix

        OUTPUT:
            parts: [list]   Valid polygon(s) ready for plotting, or empty list if invalid
        =========================================================================================================
        """
        if polygon is None or polygon.is_empty:
            return []

        try:
            # Handle MultiPolygon recursively
            if isinstance(polygon, MultiPolygon):
                fixed_parts = []
                for geom in polygon.geoms:
                    fixed_parts.extend(self._fix_longitude_wrapping(geom))
                return fixed_parts

            bounds = polygon.bounds  # (min_lon, min_lat, max_lon, max_lat)
            min_lon, min_lat, max_lon, max_lat = bounds

            # Check for invalid coordinates
            if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
                return []

            # Get latitude-dependent thresholds
            center_lat = abs((min_lat + max_lat) / 2)
            if center_lat > 60:
                max_lon_span, max_lat_span = 40, 20
            elif center_lat > 30:
                max_lon_span, max_lat_span = 25, 15
            else:
                max_lon_span, max_lat_span = 15, 10

            exterior_coords = np.array(polygon.exterior.coords)
            lons, lats = exterior_coords[:, 0], exterior_coords[:, 1]

            # Check for longitude wrapping (large jumps between consecutive points)
            lon_diffs = np.abs(np.diff(lons))
            has_wrapping = np.any(lon_diffs > 180)

            if has_wrapping:
                # Fix wrapped coordinates by unwrapping
                fixed_lons = lons.copy()
                for i in range(1, len(lons)):
                    diff = lons[i] - lons[i-1]
                    if diff > 180:
                        fixed_lons[i:] -= 360
                    elif diff < -180:
                        fixed_lons[i:] += 360

                # Check if unwrapped polygon is reasonable
                if np.max(fixed_lons) - np.min(fixed_lons) < max_lon_span:
                    try:
                        fixed_polygon = Polygon(zip(fixed_lons, lats))
                        if fixed_polygon.is_valid:
                            return [fixed_polygon]
                    except Exception:
                        pass
                return []  # Couldn't fix wrapping

            # Check size without wrapping
            lon_span = max_lon - min_lon
            lat_span = max_lat - min_lat

            if lon_span > max_lon_span or lat_span > max_lat_span:
                return []  # Too large, likely invalid

            return [polygon]  # Valid polygon

        except Exception as e:
            if hasattr(self, 'verbose') and self.verbose:
                logger.warning(f"Error fixing polygon: {e}")
            return []

    def _plot_footprint(self, ax, polygon, color, linewidth=1, alpha=0.5, transform=None):
        """
        =========================================================================================================
        Safely plot footprint polygon(s) with longitude wrapping protection.

        INPUTS:
            ax:        [matplotlib axes]      Target axes
            polygon:   [Polygon | MultiPolygon]   Footprint polygon(s)
            color:     [str]                  Plot color
            linewidth: [float]                Line width (default: 1)
            alpha:     [float]                Transparency (default: 0.5)
            transform: [cartopy projection]   Coordinate transform (optional)

        OUTPUT:
            success: [bool]   True if at least one polygon was plotted
        =========================================================================================================
        """
        fixed_polygons = self._fix_longitude_wrapping(polygon)

        plotted_count = 0
        for poly in fixed_polygons:
            try:
                ax.plot(*poly.exterior.xy, color=color, linewidth=linewidth, 
                       alpha=alpha, transform=transform)
                plotted_count += 1
            except Exception:
                continue

        return plotted_count > 0

    def _polygon_intersects_180_lat(self, polygon, max_lon_span=10, max_lat_span=5) -> bool:
        """
        =========================================================================================================
        Check if a polygon has reasonable bounds (not spanning too much of the map).

        INPUTS:
            polygon:      [Polygon]   Polygon to check
            max_lon_span: [float]     Maximum longitude span [deg] (default: 10)
            max_lat_span: [float]     Maximum latitude span [deg] (default: 5)

        OUTPUT:
            ok: [bool]   True if polygon spans are within thresholds (latitude-adaptive)
        =========================================================================================================
        """
        try:
            bounds = polygon.bounds  # (min_lon, min_lat, max_lon, max_lat)
            lon_span = bounds[2] - bounds[0]  # max_lon - min_lon
            lat_span = bounds[3] - bounds[1]  # max_lat - min_lat

            # Get the center latitude to adjust thresholds
            center_lat = abs((bounds[1] + bounds[3]) / 2)

            # Increase allowable spans at high latitudes
            if center_lat > 60:  # Above 60° latitude
                max_lat_span = 15
                max_lon_span = 30
            elif center_lat > 30:  # Above 30° latitude
                max_lat_span = 10
                max_lon_span = 20

            # Skip polygons that span too much longitude or latitude
            if lon_span > max_lon_span or lat_span > max_lat_span:
                return False

            return True
        except Exception:
            return False

    def _identify_wrapping_polygons(self, plot_bounds, gnss_constellation=None):
        """
        =========================================================================================================
        Identify problematic polygons that would cause horizontal lines or wrap-around artifacts.

        INPUTS:
            plot_bounds:        [tuple]        (min_lon, max_lon, min_lat, max_lat)
            gnss_constellation: [str | None]   Optional comma-separated constellation filter

        OUTPUT:
            problem_indices: [set[tuple]]   Set of (gnss_name, index) tuples to skip when plotting
        =========================================================================================================
        """
        min_lon, max_lon, min_lat, max_lat = plot_bounds

        gnss_const = fn.normalize_gnss_constellation_arg(gnss_constellation, self.gnss_constellation.keys())

        problem_indices = set()

        for gnss_name in gnss_const:
            if gnss_name in self.target_detect_dict:
                for idx, detect_struct in enumerate(self.target_detect_dict[gnss_name]):
                    pos_ll = detect_struct.position_ll

                    # Check if detection point has valid coordinates
                    if pos_ll is None or np.any(np.isnan(pos_ll)):
                        problem_indices.add((gnss_name, idx))
                        continue

                    # Check if detection point is within reasonable bounds
                    if abs(pos_ll[1]) > 180 or abs(pos_ll[0]) > 90:
                        problem_indices.add((gnss_name, idx))
                        continue

                    # Analyze the polygon
#                    polygon = detect_struct.shape
                    polygon = detect_struct.footprint_shape
                    if polygon is None or polygon.is_empty:
                        problem_indices.add((gnss_name, idx))
                        continue

                    bounds = polygon.bounds  # (min_lon, min_lat, max_lon, max_lat)
                    lon_span = bounds[2] - bounds[0]
                    lat_span = bounds[3] - bounds[1]

                    # Check for problematic polygons
                    is_problem = False

                    # 1. Check if polygon is way outside map bounds
                    if (bounds[2] < min_lon - 10 or bounds[0] > max_lon + 10 or 
                        bounds[3] < min_lat - 10 or bounds[1] > max_lat + 10):
                        is_problem = True

                    # 2. Check for huge spans
                    if lon_span > 50 or lat_span > 50:
                        is_problem = True

                    # 3. Check for longitude wrapping around 180/-180
                    if bounds[0] < -170 and bounds[2] > 170:
                        is_problem = True

                    # 4. Check for invalid coordinate ranges
                    if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
                        is_problem = True

                    # 5. Check for large coordinate jumps
                    if hasattr(polygon, 'exterior'):
                        coords = list(polygon.exterior.coords)
                        for i in range(1, len(coords)):
                            lon_jump = abs(coords[i][0] - coords[i-1][0])
                            lat_jump = abs(coords[i][1] - coords[i-1][1])
                            if lon_jump > 100 or lat_jump > 50:
                                is_problem = True
                                break

                    # 6. Test if clipping works properly
                    try:
                        clipped = self._polygon_within_map(polygon, min_lon, max_lon, min_lat, max_lat)
                        if clipped is None:
                            is_problem = True
                        elif hasattr(clipped, 'geoms') and len(clipped.geoms) > 10:
                            is_problem = True
                    except Exception:
                        is_problem = True

                    if is_problem:
                        problem_indices.add((gnss_name, idx))

        return problem_indices

    @staticmethod
    def _polygon_within_map(polygon: Polygon, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> Polygon:
        """
        =========================================================================================================
        Clip a polygon to a bounding box defined by min/max longitude and latitude.

        INPUTS:
            polygon: [Polygon]   Input Shapely polygon
            min_lon: [float]     Minimum longitude of the map extent
            max_lon: [float]     Maximum longitude of the map extent
            min_lat: [float]     Minimum latitude of the map extent
            max_lat: [float]     Maximum latitude of the map extent

        OUTPUT:
            clipped: [Polygon]   The clipped polygon, or an empty geometry if no overlap
        =========================================================================================================
        """
        try:
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

            if polygon.is_empty:
                return polygon

            # Define bounding box as a shapely polygon
            map_bbox = box(min_lon, min_lat, max_lon, max_lat)

            # Clip the input polygon with the map bounding box
            clipped = polygon.intersection(map_bbox)

            if not clipped.is_valid:
                clipped = clipped.buffer(0)

                # Ensure result is either Polygon or MultiPolygon
            if isinstance(clipped, Polygon | MultiPolygon) and not clipped.is_empty:
                return clipped

            return Polygon()  # Return empty Polygon if no usable geometry

        except GEOSException as e:
            logger.error(f"GEOS error while clipping polygon: {e}")
            return Polygon()
        except Exception as e:
            logger.error(f"Error clipping polygon: {e}")
            return Polygon()

    def show_track_and_detection_points(self, gnss_constellation=None, file_name=None, show_figure=False,
                                       delta_lon=None, delta_lat=None, verbose=False, label_detections=False) -> None:
        """
        =========================================================================================================
        Plot the detection points for each GNSS constellation. Filters out problematic polygons that
        cause horizontal lines / wrap-around artifacts.

        INPUTS:
            gnss_constellation: [str | None]   Comma-separated constellation filter (default: all)
            file_name:          [str | None]   Output file name (default: do not save)
            show_figure:        [bool]         Display interactively (default: False)
            delta_lon:          [float | None] Longitude buffer added to bounds [deg]; falls back to self.bbox_buffer_deg
            delta_lat:          [float | None] Latitude buffer added to bounds [deg]; falls back to self.bbox_buffer_deg
            verbose:            [bool]         Print warnings (default: False)
            label_detections:   [bool]         Annotate detections with sat IDs (default: False)
        =========================================================================================================
        """
        if delta_lon is None:
            delta_lon = self.bbox_buffer_deg
        if delta_lat is None:
            delta_lat = self.bbox_buffer_deg

        # Calculate plot bounds once (with buffer for better visualization)
        min_lon, max_lon, min_lat, max_lat = self.bounding_box
        min_lon -= delta_lon
        max_lon += delta_lon
        min_lat -= delta_lat
        max_lat += delta_lat
        plot_bounds = (min_lon, max_lon, min_lat, max_lat)

        # Silently identify problem polygons using the calculated bounds
        problem_indices = self._identify_wrapping_polygons(plot_bounds, gnss_constellation)

        gnss_const = fn.normalize_gnss_constellation_arg(gnss_constellation, self.gnss_constellation.keys())

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        gnss_const_str = ", ".join(gnss_const).replace("[", "").replace("]", "").replace(", ", " and ")
        plt.title(f"Detection Points for GNSS Constellations {gnss_const_str}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        # Add grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Plot trajectory
        trajectory_x = [point[1] for point in self.moving_target_object.trajectory_points]
        trajectory_y = [point[0] for point in self.moving_target_object.trajectory_points]
        ax.plot(trajectory_x, trajectory_y, "b-", linewidth=1.5, alpha=0.7, label="Trajectory")

        # Plot detection points (filtered)
        plotted_count = 0
        skipped_count = 0

        for gnss_name in gnss_const:
            if gnss_name in self.target_detect_dict:
                for idx, detect_struct in enumerate(self.target_detect_dict[gnss_name]):
                    # Skip problematic polygons
                    if (gnss_name, idx) in problem_indices:
                        skipped_count += 1
                        continue

                    pos_ll = detect_struct.position_ll
                    if pos_ll is not None and not np.any(np.isnan(pos_ll)):
                        # Plot the point
                        ax.scatter(
                            pos_ll[1], pos_ll[0],
                            c=fn.color_gnss_const(gnss_name),
                            s=30, marker="o", alpha=0.7,
                            label=f"{gnss_name} Detection" if idx == 0 else "",
                            transform=ccrs.PlateCarree()
                        )

                        # Plot the footprint (only for good polygons)
#                        try:
#                            footprint_reflection = self._polygon_within_map(
#                                detect_struct.footprint_shape, min_lon, max_lon, min_lat, max_lat
#                            )
#
#                            if (footprint_reflection is not None and
#                                not footprint_reflection.is_empty and
#                                footprint_reflection.is_valid):
#
#                                if hasattr(footprint_reflection, 'geoms'):  # MultiPolygon
#                                    for poly in footprint_reflection.geoms:
#                                        if poly.is_valid and not poly.is_empty:
#                                            ax.plot(
#                                                *poly.exterior.xy,
#                                                color=fn.color_gnss_const(gnss_name),
#                                                linewidth=1, alpha=0.5,
#                                                transform=ccrs.PlateCarree()
#                                            )
#                                else:  # Single Polygon
#                                    ax.plot(
#                                        *footprint_reflection.exterior.xy,
#                                        color=fn.color_gnss_const(gnss_name),
#                                        linewidth=1, alpha=0.5,
#                                        transform=ccrs.PlateCarree()
#                                    )
#                        except Exception as e:
#                            print(f"Error plotting footprint for {gnss_name}[{idx}]: {e}")
#                            skipped_count += 1

                        success = self._plot_footprint(
                            ax, detect_struct.footprint_shape,
                            color=fn.color_gnss_const(gnss_name),
                            linewidth=1, alpha=0.5,
                            transform=ccrs.PlateCarree()
                        )
                        if not success and verbose:
                            logger.warning(f"Skipped invalid footprint for {gnss_name}[{idx}]")

                        plotted_count += 1

                        if label_detections:
                            ax.text(
                                pos_ll[1],  # longitude
                                pos_ll[0],  # latitude
                                f"{gnss_name} sat-ID {detect_struct.tx_satID}, GNSS-R sat. {detect_struct.rx_satID}",
                                fontsize=8,
                                ha="right",
                                va="bottom",
                                color=fn.color_gnss_const(gnss_name),
                                transform=ccrs.PlateCarree()
                            )

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        if file_name is not None:
            file_path = os.path.join(Constants.output_folder_path, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved as {file_name}")

        if show_figure:
            plt.show(block=True)

@dataclass
class MovingTargetDetectionStruct:
    position_eci: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, dtype=np.float32))
    position_ll: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    time: dt = None
    footprint_shape: float = np.nan
    refl_obj: Reflection | Backscatter = None  # Reference to the reflection object that caused the detection
    frequency: float = np.nan
