from bisect import bisect_left
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
import datetime as dt
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
import pymap3d
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity
from tqdm import tqdm

import functions.functions as fn
import functions.link_budget as lb
from object_definitions.BackScattPoint_def import BRP as Backscatter
from object_definitions.Constants_def import Constants
from object_definitions.OceanTarget_def import OceanTarget
from object_definitions.Reflection_def import Reflection
from object_definitions.ShipTarget_def import ShipTarget
from object_definitions.ShipTrajectory_def import ShipTrajectory

logger = logging.getLogger(__name__)


class MovingTargetDetection:
    """Find when GNSS-R reflections fall on a moving target and score detectability, per band.

    Drives the moving-target mode. It walks the target trajectory, finds the reflections (specular or
    backscatter) that land on or near the target within a time tolerance, and -- per band -- evaluates the
    received power for the ocean, the ship, and the thermal noise to score detectability (SCR and SINR in
    backscatter mode, or a forward-shadow change in power in specular mode). Detection events and plots are
    produced per band.

    Parameters
    ----------
    sat_const : SatelliteConstellation
        LEO receiver constellation.
    gnss_const : dict
        ``{name: SatelliteConstellation}`` of GNSS transmitter constellations.
    moving_target_object : ShipTrajectory
        Moving target with a populated trajectory shapefile.
    antenna_obj : RxAntenna
        Receiver antenna.
    create_shapefile_for_visualization : bool, optional
        Also write a visualization shapefile (default False).
    deltaT : int, optional
        Time tolerance [s] for matching a reflection to a target position (default 60).
    earth_model : {"spherical", "elliptical"}, optional
        Surface model.
    reflection_mode : {"specular", "backscatter"}, optional
        Forward mirror point or backscatter BRP.
    link_budget_settings : dict or BistaticLinkBudgetSettings or None, optional
        Rx-system link-budget settings.
    bbox_buffer_deg : float, optional
        Padding [deg] around the trajectory bounding box for plots (default 2.0).
    max_central_angle_deg : float, keyword-only
        Maximum Tx-Rx central angle [deg].
    max_incidence_angle_deg : float or None, keyword-only
        Maximum BRP incidence angle [deg]; required for backscatter, may be None for specular.
    ship_dimensions : dict or None, keyword-only
        ``{length_m, beam_m, freeboard_m}``; enables the ship-RCS contribution when set.
    selected_signals : list of dict or None, keyword-only
        Validated ``[{band, constellation, freq_hz, eirp_dbw, chip_rate_hz}]`` plan from the INI bands.

    Attributes
    ----------
    bounding_box : tuple
        ``(min_lon, max_lon, min_lat, max_lat)`` of the trajectory.
    signals_per_constellation : dict
        Per-constellation list of band signals (frequency, EIRP, chip rate).
    target_detect_dict : dict
        Per-constellation detection accumulators.
    """

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
        *,
        max_central_angle_deg,
        max_incidence_angle_deg,
        ship_dimensions: dict | None = None,
        selected_signals: list[dict] | None = None,
    ) -> None:
        """Build the detector, validate the band plan, and prepare the per-constellation accumulators."""
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
        self.link_budget_settings = lb.coerce_link_budget_settings(link_budget_settings)
        self.bbox_buffer_deg = float(bbox_buffer_deg)
        self.max_central_angle_deg = float(max_central_angle_deg)
        if self.reflection_mode == "backscatter":
            if max_incidence_angle_deg is None:
                raise ValueError("max_incidence_angle_deg is required when reflection_mode = 'backscatter'.")
            self.max_incidence_angle_deg = float(max_incidence_angle_deg)
        else:
            self.max_incidence_angle_deg = None

        # Per-(constellation, band) signal plan from the validated INI band selection. Geometry is
        # computed once per Tx/Rx pair; the link budget is evaluated per band, each carrying its own
        # carrier frequency and transmit EIRP. Built once here so a misconfiguration fails fast.
        self.signals_per_constellation = {name: [] for name in self.gnss_constellation}
        for sig in selected_signals or []:
            constellation = sig["constellation"]
            if constellation in self.signals_per_constellation:
                self.signals_per_constellation[constellation].append(
                    {
                        "band": sig["band"],
                        "freq_hz": float(sig["freq_hz"]),
                        "eirp_dbw": float(sig["eirp_dbw"]),
                        "chip_rate_hz": float(sig["chip_rate_hz"]),
                    }
                )
        if not any(self.signals_per_constellation.values()):
            raise ValueError(
                "No usable (constellation, band) signals: every configured band was dropped during "
                "validation, or none belong to the loaded GNSS constellations. Check [ANTENNA].bands."
            )

        self.footprint_stats = {"total": 0, "rejected_empty": 0}  # populated by _reflection_point_footprint, summarised at the end of calculate_moving_target_reflection_intersection
        # Optional ship physical dimensions (length / beam / freeboard) for evaluating the ship's
        # contribution to detection-event Pr. None => ocean-only mode (legacy behaviour).
        self.ship_dimensions = ship_dimensions
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

        def show_reflection_and_target_position(pos_tgt_i: Point, footprint_reflection: Polygon, reflection_point_ll: np.ndarray, gnss_name: str) -> None:
            """
            =========================================================================================================
            Debug plot: one reflection point and the target position at a given time.

            INPUTS:
                pos_tgt_i:            [Point]        Target position as a shapely Point (lon = x, lat = y)
                footprint_reflection: [Polygon]      Footprint polygon at this time step
                reflection_point_ll:  [np.ndarray]   Reflection point [lat, lon] (specular point, or BRP in backscatter mode)
                gnss_name:            [str]          Name of the GNSS constellation
            =========================================================================================================
            """
            # Agg is forced at import for headless runs and cannot show a window; switch to the
            # interactive Qt5Agg backend (PyQt5 is a project dependency) so this debug plot displays.
            # Must happen before the figure is created, since a figure binds to the active backend.
            plt.switch_backend("Qt5Agg")
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

            footprint_label = f"Footprint {self.sat_antenna.footprint_dB_contour:g} dB"  # configured iso-gain contour
            if isinstance(footprint_reflection, Polygon):
                # Handle single Polygon
                ax.plot(
                    *footprint_reflection.exterior.xy,
                    color="red",
                    linewidth=2,
                    label=footprint_label,
                    transform=ccrs.PlateCarree(),
                )
            elif isinstance(footprint_reflection, MultiPolygon):
                # Handle MultiPolygon - plot each constituent polygon
                for i, poly in enumerate(footprint_reflection.geoms):
                    label = footprint_label if i == 0 else ""  # Only label the first polygon
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
                pos_tgt_i.x,
                pos_tgt_i.y,
                "o",
                color="blue",
                markersize=8,
                label="Target Position",
                transform=ccrs.PlateCarree(),
            )
            ax.plot(
                reflection_point_ll[1],
                reflection_point_ll[0],
                ".",
                color="orange",
                markersize=8,
                label="Reflection Point",
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
            logger.debug(f"Reflection point: Lon {reflection_point_ll[0]:.6f}, Lat {reflection_point_ll[1]:.6f}")

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

        # Specular mode reports the ocean forward-glint Pr and the forward-scatter shadow delta-Pr
        # (the change the ship induces by occulting its glint patch); the ship SCR/SINR is a
        # backscatter-mode metric and is not produced here.
        if self.reflection_mode == "specular" and self.ship_dimensions is not None:
            logger.info(
                "Specular (forward-scatter) mode: reporting ocean glint Pr and the forward-scatter shadow "
                "delta-Pr; ship SCR/SINR is a backscatter-mode metric and is not computed here."
            )

        LightweightRef = namedtuple('LightweightRef', ['refl_obj', 'time_index', 'refl_id'])
        ###########################################
        # 1: Calculate all the reflection points  #
        ###########################################
        # Reflection times are the all-visible timestamp grid produced by the visibility pre-pass.
        times_reflections = self._get_reflection_times()
        times_refl_array = np.array(times_reflections)
        times_reflections_sec = np.array([t.timestamp() for t in times_reflections])
        visible_rx_satellites = [
            (sat_no, sat_rx)
            for sat_no, sat_rx in enumerate(self.sat_constellation.satellites)
            if self._satellite_has_visible_propagation(sat_rx, len(times_reflections))
        ]

        # Calculate the total number of reflection points for all GNSS constellations (for the progress bar)
        total_comp_refl = sum(
            len(visible_rx_satellites) * len(gnss_const.satellites)
            for gnss_const in self.gnss_constellation.values()
        )
        reflection_dict = {}  # Dictionary to store the reflection points for each GNSS constellation

        # Loop through all GNSS computations and calculate all the reflection points
        with tqdm(total=total_comp_refl, desc="Reflection calculations", unit="reflection") as pbar:
            # Loop through each GNSS constellations
            for gnss_name, gnss_const in self.gnss_constellation.items():
                if verbose:
                    logger.info("-----------------------------------------------------------------")
                    logger.info(f'Calculating reflections for GNSS constellation  "{gnss_name}"')
                reflections_vec = []

                # Loop through the GNSS-R satellites in the GNSS-R LEO constellation / Rx
                for sat_no, sat_rx in visible_rx_satellites:
                    # Update progress bar description (not advancing)
                    pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {gnss_name}")

                    # Loop through the GNSS satellites in the GNSS constellation / Tx
                    for i, sat_tx in enumerate(gnss_const.satellites):
                        if i % 5 == 0:
                            # Update progress bar description (only every 5th iteration)
                            pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {gnss_name} [{i}/{len(gnss_const.satellites)}]")

                        # Geometry only here; tx_eirp_dbw/frequency_hz are unused with compute_link_budget=False
                        # (the budget is evaluated per band in _compute_ocean_pr_at_event / _compute_ship_pr_at_event).
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
                                tx_eirp_dbw=float("nan"),
                                antenna=self.sat_antenna,
                                link_budget_settings=self.link_budget_settings,
                                compute_link_budget=False,  # deferred to _compute_ocean_pr_at_event per detection event
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
                                tx_eirp_dbw=float("nan"),
                                antenna=self.sat_antenna,
                                link_budget_settings=self.link_budget_settings,
                                compute_link_budget=False,  # deferred to _compute_ocean_pr_at_event per detection event
                            )
                            reflections_vec.append(reflection)
                        else:
                            raise ValueError(f"Invalid reflection mode: {self.reflection_mode}. Must be 'specular' or 'backscatter'.")
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
            for _idx, (time_tgt_i, pos_tgt_i) in tqdm(
                enumerate(zip(self.timestamps_resampled, self.ship_positions_resampled, strict=False)),
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
                            # Create footprint around the reflection point, once per configured band of this
                            # constellation. Geometry is band-independent; only the footprint size, antenna gain,
                            # EIRP, and ocean RCS change with the carrier frequency.
                            for signal in self.signals_per_constellation[gnss_name]:
                                band = signal["band"]
                                frequency_hz = signal["freq_hz"]
                                tx_eirp_dbw = signal["eirp_dbw"]
                                chip_length_m = Constants.SPEED_OF_LIGHT / signal["chip_rate_hz"]  # delta_L per code chip
                                freq_mhz = frequency_hz / 1e6
                                footprint_reflection, reflection_point_ll = self._reflection_point_footprint(refl_obj, reflection_index, freq_mhz, verbose)

                                ## DEBUG: Plot the footprint and the moving target position
                                # show_reflection_and_target_position(pos_tgt_i, footprint_reflection, reflection_point_ll, gnss_name) # TODO this always uses-3db and specular
                                if footprint_reflection.intersects(pos_tgt_i):
                                    # Per-event ocean-clutter link budget, deferred from the reflection
                                    # propagation. A non-finite Pr means the reflection is out of the antenna
                                    # FOV or has no valid RCS; the eager path dropped these via an empty
                                    # footprint, so skip the detection to preserve that behaviour.
                                    pr_ocean_w, rcs_ocean_m2, snr_ocean_rf_db, noise_power_w = self._compute_ocean_pr_at_event(
                                        refl_obj, reflection_index, frequency_hz, tx_eirp_dbw, chip_length_m
                                    )
                                    if not np.isfinite(pr_ocean_w):
                                        continue

                                    # Moving Target could be detected! Initialize the moving target detection structure
                                    possible_target_detection = MovingTargetDetectionStruct()
                                    possible_target_detection.position_eci = refl_obj.refl_ECI_vec[reflection_index]
                                    possible_target_detection.position_ll = refl_obj.refl_LL_vec[reflection_index]
                                    possible_target_detection.time = time_tgt_i
                                    possible_target_detection.footprint_shape = footprint_reflection
                                    possible_target_detection.reflection_point_ll = reflection_point_ll  # specular point, or BRP in backscatter mode
                                    possible_target_detection.refl_obj = refl_obj
                                    possible_target_detection.frequency = freq_mhz
                                    possible_target_detection.band = band
                                    possible_target_detection.pos_tgt_i = pos_tgt_i
                                    possible_target_detection.tx_satID = refl_obj.tx_satID
                                    possible_target_detection.rx_satID = refl_obj.rx_satID

                                    possible_target_detection.pr_ocean_w = pr_ocean_w
                                    possible_target_detection.rcs_ocean_m2 = rcs_ocean_m2
                                    possible_target_detection.snr_ocean_rf_db = snr_ocean_rf_db
                                    possible_target_detection.noise_power_w = noise_power_w
                                    possible_target_detection.detection_sinr_threshold_db = self.link_budget_settings.detection_sinr_threshold_db
                                    possible_target_detection.processing_gain_linear = lb.coherent_processing_gain_linear(self.link_budget_settings)

                                    # Ship backscatter SCR/SINR is a backscatter-mode metric: it divides the ship's
                                    # R_t^2 R_r^2 backscatter Pr by the ocean clutter Pr. In specular (forward-scatter)
                                    # mode the ship signature is instead the change it induces in the coherent forward
                                    # glint, so the ship is not run through the backscatter link budget here; only the
                                    # ocean glint Pr and the geometric ship-in-footprint event are reported.
                                    if self.reflection_mode == "backscatter" and self.ship_dimensions is not None:
                                        ship_position_ll = (pos_tgt_i.y, pos_tgt_i.x)
                                        pr_ship_w, rcs_ship_m2, snr_ship_rf_db, ship_noise_power_w = self._compute_ship_pr_at_event(
                                            refl_obj, reflection_index, frequency_hz, tx_eirp_dbw, time_tgt_i, ship_position_ll,
                                        )
                                        possible_target_detection.pr_ship_w = pr_ship_w
                                        possible_target_detection.rcs_ship_m2 = rcs_ship_m2
                                        possible_target_detection.snr_ship_rf_db = snr_ship_rf_db
                                        if not np.isfinite(noise_power_w) and np.isfinite(ship_noise_power_w):
                                            noise_power_w = ship_noise_power_w
                                            possible_target_detection.noise_power_w = noise_power_w
                                        self._populate_detection_metrics(possible_target_detection)
                                    elif self.reflection_mode == "specular" and self.ship_dimensions is not None:
                                        # Forward-scatter detection: the ship's signature is the change it induces in the
                                        # ocean glint by occulting its waterline-area patch (delta_Pr = -shadow power).
                                        ship_position_ll = (pos_tgt_i.y, pos_tgt_i.x)
                                        pr_shadow_w, _rcs_shadow_m2, shadow_noise_power_w = self._compute_forward_shadow_delta_pr_at_event(
                                            refl_obj, reflection_index, frequency_hz, tx_eirp_dbw, time_tgt_i, ship_position_ll,
                                        )
                                        possible_target_detection.forward_shadow_pr_w = pr_shadow_w
                                        if not np.isfinite(noise_power_w) and np.isfinite(shadow_noise_power_w):
                                            noise_power_w = shadow_noise_power_w
                                            possible_target_detection.noise_power_w = noise_power_w
                                        self._populate_forward_detection_metrics(possible_target_detection)

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

    @staticmethod
    def _populate_detection_metrics(event: "MovingTargetDetectionStruct") -> None:
        """
        Populate derived ship-vs-ocean/noise metrics for one footprint-overlap event.

        The moving-target detectability metric is SINR = P_ship / (P_ocean + P_noise).
        Footprint-overlap candidates are still kept even when this threshold is not met.
        """
        if not (
            event.pr_ocean_w is not None
            and event.pr_ship_w is not None
            and np.isfinite(event.pr_ocean_w)
            and np.isfinite(event.pr_ship_w)
            and event.pr_ocean_w > 0.0
            and event.pr_ship_w > 0.0
        ):
            return

        event.pr_total_w = event.pr_ocean_w + event.pr_ship_w
        event.scr_db = 10.0 * np.log10(event.pr_ship_w / event.pr_ocean_w)

        if not (event.noise_power_w is not None and np.isfinite(event.noise_power_w) and event.noise_power_w > 0.0):
            return

        event.snr_total_rf_db = 10.0 * np.log10(event.pr_total_w / event.noise_power_w)
        # Coherent integration plus the sqrt(M) estimator-sensitivity approximation is expressed as an
        # equivalent noise scale for thresholding. This is not the post-detection mean thermal power.
        gain = event.processing_gain_linear if (event.processing_gain_linear and event.processing_gain_linear > 0.0) else 1.0
        event.processing_gain_db = 10.0 * np.log10(gain)
        event.noise_power_effective_w = event.noise_power_w / gain
        interference_plus_noise_w = event.pr_ocean_w + event.noise_power_effective_w
        event.sinr_ship_db = 10.0 * np.log10(event.pr_ship_w / interference_plus_noise_w)

        if event.detection_sinr_threshold_db is None or not np.isfinite(event.detection_sinr_threshold_db):
            return
        event.detection_margin_db = event.sinr_ship_db - event.detection_sinr_threshold_db
        event.is_sinr_detection = bool(event.detection_margin_db >= 0.0)

    def _get_reflection_times(self) -> list[dt.datetime]:
        """
        Return the visible timestamp grid used for reflection calculations.
        """
        timestamps_resampled = getattr(self, "timestamps_resampled", None)
        if timestamps_resampled is not None and len(timestamps_resampled) > 0:
            return [fn.ensure_utc_aware(t) for t in timestamps_resampled]

        for sat in self.sat_constellation.satellites:
            time_prop_vec = getattr(sat, "time_prop_vec", None)
            if time_prop_vec is not None and len(time_prop_vec) > 0:
                return [t.utc_datetime() for t in time_prop_vec]

        return []

    @staticmethod
    def _satellite_has_visible_propagation(sat, expected_time_count: int) -> bool:
        """
        True when a LEO satellite has propagated state on the all-visible timestamp grid.
        """
        time_prop_vec = getattr(sat, "time_prop_vec", None)
        positions_eci = getattr(sat, "positions_eci", None)
        velocities_eci = getattr(sat, "velocities_eci", None)
        if time_prop_vec is None or positions_eci is None or velocities_eci is None:
            return False
        if len(time_prop_vec) != expected_time_count:
            return False

        positions_eci = np.asarray(positions_eci, dtype=float)
        velocities_eci = np.asarray(velocities_eci, dtype=float)
        if positions_eci.shape != (3, expected_time_count) or velocities_eci.shape != (3, expected_time_count):
            return False

        valid_state = np.isfinite(positions_eci).all(axis=0) & np.isfinite(velocities_eci).all(axis=0)
        return bool(np.any(valid_state))

    def _compute_ship_heading_eci(self, detection_time: dt.datetime) -> np.ndarray | None:
        """
        Estimate the ship's bow direction in ECI at the detection time.

        Uses the two trajectory waypoints that bracket the detection time directly — the
        trajectory's stored timestamps and points define the segment, so no interval lookup
        or ad-hoc sampling window is needed. Times outside the trajectory range fall back to
        the first / last segment (initial / final bearing).
        Returns None if the trajectory is too short or the segment is degenerate.
        """
        timestamps = self.moving_target_object.timestamps
        points = self.moving_target_object.trajectory_points
        if len(timestamps) < 2 or len(points) < 2:
            return None                                       # need >= 2 waypoints to define a heading

        idx = bisect_left(timestamps, detection_time)
        if idx <= 0:
            i_before, i_after = 0, 1                          # before trajectory -> first segment (initial bearing)
        elif idx >= len(timestamps):
            i_before, i_after = len(timestamps) - 2, len(timestamps) - 1   # after trajectory -> last segment
        else:
            i_before, i_after = idx - 1, idx                  # bracketing segment

        pos_before = points[i_before]
        pos_after = points[i_after]
        if pos_before == pos_after:
            return None                                       # degenerate segment

        lat_before, lon_before = pos_before
        lat_after, lon_after = pos_after
        x_before, y_before, z_before = pymap3d.geodetic2ecef(lat_before, lon_before, 0.0)
        x_after, y_after, z_after = pymap3d.geodetic2ecef(lat_after, lon_after, 0.0)
        heading_ecef = np.array([x_after - x_before, y_after - y_before, z_after - z_before], dtype=float)
        mag = np.linalg.norm(heading_ecef)
        if mag < 1.0:                                         # less than 1 m segment -> degenerate
            return None
        heading_ecef /= mag

        # ECEF and ECI share the Earth's centre as origin, so ecef2eci is a pure rotation —
        # applied to a unit direction it returns the rotated unit direction.
        hx, hy, hz = pymap3d.ecef2eci(
            np.array([heading_ecef[0]]),
            np.array([heading_ecef[1]]),
            np.array([heading_ecef[2]]),
            [detection_time],
        )
        return np.array([float(hx[0]), float(hy[0]), float(hz[0])])

    def _compute_ship_pr_at_event(
        self,
        refl_obj: Reflection | Backscatter,
        reflection_index: int,
        frequency_hz: float,
        tx_eirp_dbw: float,
        detection_time: dt.datetime,
        ship_position_ll: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        """
        Compute the ship's contribution to received power at one detection event.

        Builds a single-sample BistaticGeometry at the detected ship position, instantiates
        a ShipTarget with the ship's instantaneous heading, and runs the bistatic link budget.
        Returns (pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w);
        all NaN if heading is unavailable or geometry is invalid.
        """
        nan_result = (float("nan"), float("nan"), float("nan"), float("nan"))
        if self.ship_dimensions is None:
            return nan_result

        heading_eci = self._compute_ship_heading_eci(detection_time)
        if heading_eci is None:
            return nan_result

        ship_lat, ship_lon = ship_position_ll
        ship_x_ecef, ship_y_ecef, ship_z_ecef = pymap3d.geodetic2ecef(ship_lat, ship_lon, 0.0)
        ship_pos_ecef = np.array([[ship_x_ecef, ship_y_ecef, ship_z_ecef]], dtype=float)
        ship_pos_eci = np.column_stack(
            pymap3d.ecef2eci(
                np.array([ship_x_ecef]),
                np.array([ship_y_ecef]),
                np.array([ship_z_ecef]),
                [detection_time],
            )
        )
        ship_surface_normal_eci = lb.ellipsoid_surface_normals_eci(ship_pos_ecef, [detection_time])

        # Per-sample slices from the cached reflection object, evaluated at the ship point.
        tx_pos = refl_obj.gnss_pos_vec[:, reflection_index:reflection_index + 1]      # (3, 1)
        rx_pos = refl_obj.sat_pos_vec[:, reflection_index:reflection_index + 1]       # (3, 1)
        rx_vel = refl_obj.sat_vel_vec[:, reflection_index]
        if not (
            np.isfinite(tx_pos).all()
            and np.isfinite(rx_pos).all()
            and np.isfinite(rx_vel).all()
            and np.isfinite(ship_pos_eci).all()
            and np.isfinite(ship_surface_normal_eci).all()
        ):
            return nan_result

        geometry = lb.build_bistatic_geometry(tx_pos, rx_pos, ship_pos_eci, ship_surface_normal_eci)
        wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz

        ship_target = ShipTarget.from_dict(self.ship_dimensions, heading_eci=heading_eci[np.newaxis, :])
        rcs_result = ship_target.compute_rcs(geometry, wavelength_m)
        rcs_ship_m2 = float(rcs_result.rcs_m2[0])
        if not np.isfinite(rcs_ship_m2):
            return nan_result

        look_dir_center_a = lb.compute_rx_look_direction_antenna_frame(
            rx_pos[:, 0],
            rx_vel,
            refl_obj.refl_ECI_vec[reflection_index],
            self.sat_antenna,
        )
        steering_angles = self.sat_antenna.resolve_steering_angles_for_direction(look_dir_center_a)
        if steering_angles is None:
            return nan_result
        steering_across_track_deg, steering_along_track_deg = steering_angles

        look_dir_ship_a = lb.compute_rx_look_direction_antenna_frame(
            rx_pos[:, 0],
            rx_vel,
            ship_pos_eci[0],
            self.sat_antenna,
        )
        rx_gain_dbi = self.sat_antenna.get_gain_dbi_for_direction(
            look_dir_ship_a,
            frequency_hz / 1e6,
            across_track_deg=steering_across_track_deg,
            along_track_deg=steering_along_track_deg,
        )

        link_budget = lb.compute_bistatic_link_budget(
            range_tx_m=geometry.range_tx_m,
            range_rx_m=geometry.range_rx_m,
            rx_gain_dbi=np.array([rx_gain_dbi]),
            rcs_m2=np.array([rcs_ship_m2]),
            frequency_hz=frequency_hz,
            tx_eirp_dbw=tx_eirp_dbw,
            settings=self.link_budget_settings,
        )
        pr_ship_w = float(link_budget["received_power_w"][0])
        snr_ship_rf_db = float(link_budget["snr_rf_db"][0])
        noise_power_w = float(link_budget["noise_power_w"])
        return pr_ship_w, rcs_ship_m2, snr_ship_rf_db, noise_power_w

    def _compute_forward_shadow_delta_pr_at_event(
        self,
        refl_obj: Reflection | Backscatter,
        reflection_index: int,
        frequency_hz: float,
        tx_eirp_dbw: float,
        detection_time: dt.datetime,
        ship_position_ll: tuple[float, float],
    ) -> tuple[float, float, float]:
        """
        Forward-scatter shadowing (specular mode): the ocean glint power the ship occults.

        Builds the bistatic geometry at the ship position, takes the ocean *incoherent* (diffuse)
        RCS over the ship's waterline area (length x beam), and runs the bistatic link budget. The
        result is the glint power that patch of sea would have returned; the detectable forward
        change is delta_Pr = -pr_shadow. The coherent-glint shadow (calm sea, ship at the specular
        point) is a non-linear field regime deferred as a refinement; for any non-trivial sea state
        the forward return is the diffuse term used here. Returns (pr_shadow_w, ocean_rcs_shadow_m2,
        noise_power_w); all NaN if geometry is invalid.
        """
        nan_result = (float("nan"), float("nan"), float("nan"))
        if self.ship_dimensions is None:
            return nan_result
        ship_area_m2 = float(self.ship_dimensions["length_m"]) * float(self.ship_dimensions["beam_m"])  # waterline plan area

        ship_lat, ship_lon = ship_position_ll
        ship_x_ecef, ship_y_ecef, ship_z_ecef = pymap3d.geodetic2ecef(ship_lat, ship_lon, 0.0)
        ship_pos_ecef = np.array([[ship_x_ecef, ship_y_ecef, ship_z_ecef]], dtype=float)
        ship_pos_eci = np.column_stack(
            pymap3d.ecef2eci(np.array([ship_x_ecef]), np.array([ship_y_ecef]), np.array([ship_z_ecef]), [detection_time])
        )
        surface_normal_eci = lb.ellipsoid_surface_normals_eci(ship_pos_ecef, [detection_time])

        tx_pos = refl_obj.gnss_pos_vec[:, reflection_index:reflection_index + 1]      # (3, 1)
        rx_pos = refl_obj.sat_pos_vec[:, reflection_index:reflection_index + 1]       # (3, 1)
        rx_vel = refl_obj.sat_vel_vec[:, reflection_index]
        if not (
            np.isfinite(tx_pos).all()
            and np.isfinite(rx_pos).all()
            and np.isfinite(rx_vel).all()
            and np.isfinite(ship_pos_eci).all()
            and np.isfinite(surface_normal_eci).all()
        ):
            return nan_result

        geometry = lb.build_bistatic_geometry(tx_pos, rx_pos, ship_pos_eci, surface_normal_eci)
        wavelength_m = Constants.SPEED_OF_LIGHT / frequency_hz

        # Ocean diffuse RCS over the ship-sized patch: nrcs * min(glistening_area, ship_area) = nrcs * ship_area.
        ocean_target = OceanTarget.from_settings(self.link_budget_settings)
        rcs_result = ocean_target.compute_rcs(geometry, wavelength_m, illuminated_area_m2=np.array([ship_area_m2]))
        ocean_rcs_shadow_m2 = float(rcs_result.info["rcs_incoherent_m2"][0])
        if not np.isfinite(ocean_rcs_shadow_m2) or ocean_rcs_shadow_m2 <= 0.0:
            return nan_result

        look_dir_center_a = lb.compute_rx_look_direction_antenna_frame(
            rx_pos[:, 0], rx_vel, refl_obj.refl_ECI_vec[reflection_index], self.sat_antenna
        )
        steering_angles = self.sat_antenna.resolve_steering_angles_for_direction(look_dir_center_a)
        if steering_angles is None:
            return nan_result
        steering_across_track_deg, steering_along_track_deg = steering_angles

        look_dir_ship_a = lb.compute_rx_look_direction_antenna_frame(rx_pos[:, 0], rx_vel, ship_pos_eci[0], self.sat_antenna)
        rx_gain_dbi = self.sat_antenna.get_gain_dbi_for_direction(
            look_dir_ship_a, frequency_hz / 1e6, across_track_deg=steering_across_track_deg, along_track_deg=steering_along_track_deg
        )

        link_budget = lb.compute_bistatic_link_budget(
            range_tx_m=geometry.range_tx_m,
            range_rx_m=geometry.range_rx_m,
            rx_gain_dbi=np.array([rx_gain_dbi]),
            rcs_m2=np.array([ocean_rcs_shadow_m2]),
            frequency_hz=frequency_hz,
            tx_eirp_dbw=tx_eirp_dbw,
            settings=self.link_budget_settings,
        )
        pr_shadow_w = float(link_budget["received_power_w"][0])
        noise_power_w = float(link_budget["noise_power_w"])
        return pr_shadow_w, ocean_rcs_shadow_m2, noise_power_w

    @staticmethod
    def _populate_forward_detection_metrics(event: "MovingTargetDetectionStruct") -> None:
        """
        Derive the forward-scatter shadow metrics from the occulted ocean glint power.

        forward_delta_pr_w is the signed change in received power (negative: the ship removes glint);
        the fraction and over-noise figures express it against the footprint glint and the thermal floor.
        """
        shadow = event.forward_shadow_pr_w
        if not (shadow is not None and np.isfinite(shadow) and shadow > 0.0):
            return
        event.forward_delta_pr_w = -shadow                                # ship occults glint -> negative change
        if event.pr_ocean_w is not None and np.isfinite(event.pr_ocean_w) and event.pr_ocean_w > 0.0:
            event.forward_delta_pr_fraction_db = 10.0 * np.log10(shadow / event.pr_ocean_w)
        if event.noise_power_w is not None and np.isfinite(event.noise_power_w) and event.noise_power_w > 0.0:
            # Compare the shadow against the same detection-equivalent noise scale as the SINR metric.
            gain = event.processing_gain_linear if (event.processing_gain_linear and event.processing_gain_linear > 0.0) else 1.0
            event.processing_gain_db = 10.0 * np.log10(gain)
            event.noise_power_effective_w = event.noise_power_w / gain
            event.forward_delta_pr_over_noise_db = 10.0 * np.log10(shadow / event.noise_power_effective_w)

    def _compute_ocean_pr_at_event(
        self,
        refl_obj: Reflection | Backscatter,
        reflection_index: int,
        frequency_hz: float,
        tx_eirp_dbw: float,
        chip_length_m: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Compute the ocean-clutter received power and RCS at one reflection sample.

        Re-runs the shared link budget on a single-sample slice of the reflection object, so the
        result is identical to the eager full-series value at `reflection_index`. `chip_length_m`
        (c / chip_rate) is only consumed when the delay-Doppler cell is enabled. Returns
        (pr_ocean_w, rcs_ocean_m2, snr_ocean_rf_db, noise_power_w).
        """
        idx = reflection_index
        if self.earth_model == "elliptical" and refl_obj.refl_ECEF_vec is not None:
            surface_normals_eci = lb.ellipsoid_surface_normals_eci(refl_obj.refl_ECEF_vec[idx:idx + 1, :], [refl_obj.time_vec[idx]])
        else:
            surface_normals_eci = None

        link_budget = lb.compute_reflection_link_budget(
            tx_positions_eci=refl_obj.gnss_pos_vec[:, idx:idx + 1],
            rx_positions_eci=refl_obj.sat_pos_vec[:, idx:idx + 1],
            rx_velocities_eci=refl_obj.sat_vel_vec[:, idx:idx + 1],
            reflection_positions_eci=refl_obj.refl_ECI_vec[idx:idx + 1, :],
            incidence_deg=refl_obj.refl_ang_vec[idx:idx + 1],
            antenna=self.sat_antenna,
            frequency_hz=frequency_hz,
            tx_eirp_dbw=tx_eirp_dbw,
            settings=self.link_budget_settings,
            surface_normals_eci=surface_normals_eci,
            tx_velocities_eci=refl_obj.gnss_vel_vec[:, idx:idx + 1],
            chip_length_m=chip_length_m,
        )
        pr_ocean_w = float(link_budget["received_power_w"][0])
        rcs_ocean_m2 = float(link_budget["rcs_m2"][0])
        snr_ocean_rf_db = float(link_budget["snr_rf_db"][0])
        noise_power_w = float(link_budget["noise_power_w"])
        return pr_ocean_w, rcs_ocean_m2, snr_ocean_rf_db, noise_power_w

    def show_link_budget_contributions(self, file_name: str | None = None, show_figure: bool = False, band: str | None = None) -> bool:
        """
        =========================================================================================================
        Visualise per-footprint-overlap ocean clutter, ship signal, noise, and SINR link-budget metrics.

        Saves four separate figures, each the base file_name with a per-metric suffix:
            - <stem>_power<ext>:     Pr_ship_dBm vs Pr_ocean_dBm scatter, SCR = 0 dB diagonal, thermal floor.
            - <stem>_scr_hist<ext>:  SCR_dB histogram per GNSS constellation.
            - <stem>_sinr_hist<ext>: ship SINR histogram per GNSS constellation, with the SINR threshold.
            - <stem>_margin<ext>:    per-event SINR margin relative to that threshold.

        Also prints a per-constellation summary (event counts, SCR min/median/max, SINR min/median/max,
        and SINR-qualified counts) to stdout.

        INPUTS:
            file_name:   [str | None]   Base output filename in Constants.output_folder_path; each metric
                                        is saved as <stem>_<metric><ext>. None => don't save.
            show_figure: [bool]         Display interactively (default: False)
            band:        [str | None]   Restrict to one band (default: all bands combined)

        RETURNS:
            [bool]   True if plots were produced, False if skipped (no events with finite ship Pr, e.g.
                     specular mode where the ship backscatter budget is not evaluated).
        =========================================================================================================
        """
        def _power_w_to_dbm(power_w: float) -> float:
            return float(10.0 * np.log10(power_w) + 30.0)

        def _finite_values(events: list, attr_name: str) -> list[float]:
            values = []
            for event in events:
                value = getattr(event, attr_name)
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            return values

        # Collect footprint-overlap events that have a finite ship-Pr and ocean-Pr pair.
        events_by_const: dict[str, list] = {}
        for gnss_name, events in self.target_detect_dict.items():
            kept = []
            for e in events:
                if band is not None and e.band != band:
                    continue
                if (
                    e.pr_ocean_w is not None and np.isfinite(e.pr_ocean_w) and e.pr_ocean_w > 0.0
                    and e.pr_ship_w is not None and np.isfinite(e.pr_ship_w) and e.pr_ship_w > 0.0
                ):
                    kept.append(e)
            events_by_const[gnss_name] = kept

        if not any(events_by_const.values()):
            logger.warning("No footprint-overlap events with finite ship Pr -- skipping link budget contribution plot.")
            return False

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        const_color = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(self.gnss_constellation.keys())
        }

        # Each metric is saved as its own figure: the base file_name gets a per-panel suffix.
        def _panel_file(suffix: str) -> str | None:
            if file_name is None:
                return None
            stem, ext = os.path.splitext(file_name)
            return f"{stem}_{suffix}{ext}"

        def _finalize(fig, suffix: str) -> None:
            if band is not None:
                owner = Constants.get_constellation_for_band(band)
                fig.suptitle(f"{owner} -- Band {band}" if owner else f"Band {band}", fontweight="bold")
            fig.tight_layout()
            panel_file = _panel_file(suffix)
            if panel_file is not None:
                fig.savefig(os.path.join(Constants.output_folder_path, panel_file), dpi=300, bbox_inches="tight")
                logger.info(f"Figure saved as {panel_file}")
            if show_figure:
                plt.show(block=True)
            plt.close(fig)

        # Received-power scatter: Pr_ship_dBm vs Pr_ocean_dBm.
        fig, ax_power = plt.subplots(figsize=(8, 8))
        all_dbm = []
        all_noise_dbm = []
        for gnss_name, events in events_by_const.items():
            if not events:
                continue
            pr_ocean_dbm = np.array([_power_w_to_dbm(e.pr_ocean_w) for e in events])
            pr_ship_dbm = np.array([_power_w_to_dbm(e.pr_ship_w) for e in events])
            ax_power.scatter(
                pr_ocean_dbm,
                pr_ship_dbm,
                color=const_color[gnss_name],
                label=gnss_name,
                alpha=0.7,
                s=60,
                edgecolors="black",
            )
            all_dbm.extend(pr_ocean_dbm.tolist() + pr_ship_dbm.tolist())
            all_noise_dbm.extend(
                _power_w_to_dbm(e.noise_power_w)
                for e in events
                if e.noise_power_w is not None and np.isfinite(e.noise_power_w) and e.noise_power_w > 0.0
            )

        if all_noise_dbm:
            noise_floor_dbm = float(np.median(all_noise_dbm))
            all_dbm.append(noise_floor_dbm)
            ax_power.axvline(noise_floor_dbm, color="0.35", linestyle=":", linewidth=1.5, label="Thermal floor (pre-correlation)")
            ax_power.axhline(noise_floor_dbm, color="0.35", linestyle=":", linewidth=1.5)

        lo, hi = min(all_dbm), max(all_dbm)
        padding = max((hi - lo) * 0.05, 1.0)
        lo -= padding
        hi += padding
        ax_power.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="SCR = 0 dB")
        ax_power.set_xlim(lo, hi)
        ax_power.set_ylim(lo, hi)
        ax_power.set_xlabel("Pr ocean clutter [dBm]")
        ax_power.set_ylabel("Pr ship signal [dBm]")
        ax_power.set_title("Received Power per Footprint-Overlap Event")
        ax_power.legend(loc="best")
        ax_power.grid(True, alpha=0.3)
        ax_power.set_aspect("equal", adjustable="box")
        _finalize(fig, "power")

        # SCR histogram.
        fig, ax_scr_hist = plt.subplots(figsize=(8, 6))
        all_scrs = []
        for events in events_by_const.values():
            all_scrs.extend(e.scr_db for e in events if e.scr_db is not None and np.isfinite(e.scr_db))
        scr_lo = min(all_scrs + [-5.0])
        scr_hi = max(all_scrs + [5.0])
        bin_count = max(10, min(30, len(all_scrs)))
        bins = np.linspace(scr_lo - 5.0, scr_hi + 5.0, bin_count + 1)

        for gnss_name, events in events_by_const.items():
            scrs = np.array([e.scr_db for e in events if e.scr_db is not None and np.isfinite(e.scr_db)])
            if scrs.size == 0:
                continue
            ax_scr_hist.hist(
                scrs, bins=bins, color=const_color[gnss_name], alpha=0.5,
                label=f"{gnss_name} (n={scrs.size})", edgecolor="black",
            )
        ax_scr_hist.axvline(0.0, color="k", linestyle="--", alpha=0.5, label="SCR = 0 dB")
        ax_scr_hist.set_xlabel("SCR = 10 log10(Pr_ship / Pr_ocean) [dB]")
        ax_scr_hist.set_ylabel("Number of footprint-overlap events")
        ax_scr_hist.set_title("Signal-to-Clutter Ratio Distribution")
        ax_scr_hist.legend(loc="best")
        ax_scr_hist.grid(True, alpha=0.3)
        _finalize(fig, "scr_hist")

        # SINR histogram.
        fig, ax_sinr_hist = plt.subplots(figsize=(8, 6))
        all_sinrs = []
        all_thresholds = []
        for events in events_by_const.values():
            all_sinrs.extend(_finite_values(events, "sinr_ship_db"))
            all_thresholds.extend(_finite_values(events, "detection_sinr_threshold_db"))

        if all_sinrs:
            sinr_lo = min(all_sinrs + [-5.0])
            sinr_hi = max(all_sinrs + [5.0])
            sinr_bin_count = max(10, min(30, len(all_sinrs)))
            sinr_bins = np.linspace(sinr_lo - 5.0, sinr_hi + 5.0, sinr_bin_count + 1)
            for gnss_name, events in events_by_const.items():
                sinrs = np.array(_finite_values(events, "sinr_ship_db"))
                if sinrs.size == 0:
                    continue
                ax_sinr_hist.hist(
                    sinrs, bins=sinr_bins, color=const_color[gnss_name], alpha=0.5,
                    label=f"{gnss_name} (n={sinrs.size})", edgecolor="black",
                )
            if all_thresholds:
                threshold_db = float(np.median(all_thresholds))
                ax_sinr_hist.axvline(
                    threshold_db,
                    color="k",
                    linestyle="--",
                    alpha=0.6,
                    label=f"Threshold {threshold_db:.1f} dB",
                )
            ax_sinr_hist.legend(loc="best")
        else:
            ax_sinr_hist.text(0.5, 0.5, "No finite SINR values", transform=ax_sinr_hist.transAxes, ha="center", va="center")
        ax_sinr_hist.set_xlabel("Ship SINR = 10 log10(Pr_ship / (Pr_ocean + noise)) [dB]")
        ax_sinr_hist.set_ylabel("Number of footprint-overlap events")
        ax_sinr_hist.set_title("Ship SINR Distribution")
        ax_sinr_hist.grid(True, alpha=0.3)
        _finalize(fig, "sinr_hist")

        # SINR detection margin per event. The margin is SINR - threshold, so 0 dB is the qualification boundary.
        fig, ax_margin = plt.subplots(figsize=(8, 6))
        event_offset = 0
        plotted_margin_label = set()
        has_margins = False
        for gnss_name, events in events_by_const.items():
            margin_pairs = [
                (event_offset + idx, e)
                for idx, e in enumerate(events)
                if e.detection_margin_db is not None and np.isfinite(e.detection_margin_db)
            ]
            event_offset += len(events)
            if not margin_pairs:
                continue

            x_values = np.array([idx for idx, _ in margin_pairs], dtype=int)
            margin_events = [e for _, e in margin_pairs]
            margins = np.array([e.detection_margin_db for e in margin_events], dtype=float)
            pass_mask = np.array([e.is_sinr_detection is True for e in margin_events], dtype=bool)
            fail_mask = np.array([e.is_sinr_detection is False for e in margin_events], dtype=bool)

            if np.any(pass_mask):
                label = gnss_name if gnss_name not in plotted_margin_label else None
                ax_margin.scatter(
                    x_values[pass_mask],
                    margins[pass_mask],
                    color=const_color[gnss_name],
                    marker="o",
                    s=48,
                    alpha=0.75,
                    edgecolors="black",
                    label=label,
                )
            if np.any(fail_mask):
                # Below-threshold points are left unlabeled: the dashed SINR-threshold line makes them obvious.
                ax_margin.scatter(
                    x_values[fail_mask],
                    margins[fail_mask],
                    color=const_color[gnss_name],
                    marker="x",
                    s=60,
                    alpha=0.85,
                )
            plotted_margin_label.add(gnss_name)
            has_margins = True

        ax_margin.axhline(0.0, color="k", linestyle="--", alpha=0.6, label="SINR threshold")
        if not has_margins:
            ax_margin.text(0.5, 0.5, "No finite SINR margins", transform=ax_margin.transAxes, ha="center", va="center")
        ax_margin.set_xlabel("Footprint-overlap event index")
        ax_margin.set_ylabel("SINR margin [dB]")
        ax_margin.set_title("Ship SINR Qualification Margin")
        ax_margin.legend(loc="best")
        ax_margin.grid(True, alpha=0.3)
        _finalize(fig, "margin")

        # Console summary.
        print()
        print("=" * 64)
        header = "LINK BUDGET CONTRIBUTION SUMMARY (ship vs ocean/noise per footprint overlap)"
        if band is not None:
            header += f" -- band {band}"
        print(header)
        print("=" * 64)
        for gnss_name in self.gnss_constellation:
            all_events = self.target_detect_dict.get(gnss_name, [])
            if band is not None:
                all_events = [e for e in all_events if e.band == band]
            total = len(all_events)
            if band is not None and total == 0:
                continue  # this constellation has no signal in this band -- omit the noise row
            kept = events_by_const.get(gnss_name, [])
            n_finite = len(kept)
            if n_finite == 0:
                print(f"  {gnss_name:8s}: {total:4d} footprint overlaps, 0 with finite ship Pr")
                continue
            scrs = np.array([e.scr_db for e in kept if e.scr_db is not None and np.isfinite(e.scr_db)])
            ship_dominant = int(np.sum(scrs > 0))
            ocean_dominant = int(np.sum(scrs <= 0))
            print(f"  {gnss_name:8s}: {total:4d} footprint overlaps, {n_finite:3d} with finite ship Pr")
            if scrs.size > 0:
                print(
                    f"    SCR [dB]    min={scrs.min():7.2f}  median={np.median(scrs):7.2f}  max={scrs.max():7.2f}"
                )
                print(
                    f"    ship-dominant (SCR > 0): {ship_dominant:3d} / {n_finite:<3d}    "
                    f"ocean-dominant: {ocean_dominant:3d} / {n_finite:<3d}"
                )
            sinrs = np.array([e.sinr_ship_db for e in kept if e.sinr_ship_db is not None and np.isfinite(e.sinr_ship_db)])
            if sinrs.size > 0:
                threshold = kept[0].detection_sinr_threshold_db
                n_sinr_detected = sum(1 for e in kept if e.is_sinr_detection is True)
                print(
                    f"    SINR [dB]   min={sinrs.min():7.2f}  median={np.median(sinrs):7.2f}  max={sinrs.max():7.2f}"
                )
                print(
                    f"    SINR-qualified (threshold {threshold:.2f} dB): {n_sinr_detected:3d} / {n_finite:<3d}"
                )
        print("=" * 64)
        return True

    def show_forward_shadow_summary(self, band: str | None = None) -> bool:
        """
        Print the forward-scatter shadow delta-Pr per constellation (specular mode).

        Reports, per GNSS constellation, the change the ship induces in the ocean glint by occulting
        its waterline-area patch: delta-Pr relative to the footprint glint and relative to the thermal
        floor. Returns True if any constellation had a finite shadow delta-Pr, False otherwise.
        """
        rows = []
        for gnss_name in self.gnss_constellation:
            events = self.target_detect_dict.get(gnss_name, [])
            if band is not None:
                events = [e for e in events if e.band == band]
            fractions = [e.forward_delta_pr_fraction_db for e in events if e.forward_delta_pr_fraction_db is not None and np.isfinite(e.forward_delta_pr_fraction_db)]
            over_noise = [e.forward_delta_pr_over_noise_db for e in events if e.forward_delta_pr_over_noise_db is not None and np.isfinite(e.forward_delta_pr_over_noise_db)]
            if fractions:
                rows.append((gnss_name, len(events), np.array(fractions), np.array(over_noise)))
        if not rows:
            return False

        print()
        print("=" * 64)
        header = "FORWARD-SCATTER SHADOW SUMMARY (ship glint occultation, specular mode)"
        if band is not None:
            header += f" -- band {band}"
        print(header)
        print("=" * 64)
        for gnss_name, n_events, fr, on in rows:
            print(f"  {gnss_name:8s}: {n_events:4d} footprint overlaps, {fr.size:3d} with finite shadow delta-Pr")
            print(f"    delta-Pr / footprint glint [dB]  min={fr.min():7.2f}  median={np.median(fr):7.2f}  max={fr.max():7.2f}")
            if on.size > 0:
                n_above = int(np.sum(on > 0.0))
                print(f"    delta-Pr / noise [dB]            min={on.min():7.2f}  median={np.median(on):7.2f}  max={on.max():7.2f}  (above noise: {n_above}/{on.size})")
        print("=" * 64)
        return True

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
            gnss_constellation_names = list(self.reflection_dict_timeSorted.keys())
        elif isinstance(gnss_constellation_name, str):
            gnss_constellation_names = [gnss_constellation_name]
        else:
            gnss_constellation_names = list(gnss_constellation_name)

        for gnss_const in gnss_constellation_names:
            # Check if the constellation name is valid
            if gnss_const not in self.reflection_dict_timeSorted:
                raise ValueError(f'ERROR: GNSS constellation "{gnss_const}" not found in reflection dictionary.')

        # Get the reflection points for the given time
        # Find the closest time in the reflection trajectory matching the current time at the moving target position
        reflections_time_i = {}
        for gnss_const in gnss_constellation_names:
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
        west, east = min_lon - buffer_deg, max_lon + buffer_deg
        south, north = min_lat - buffer_deg, max_lat + buffer_deg
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

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

    def create_reflection_animation(self, output_filename=None, fps=3):
        """
        =========================================================================================================
        Create a GIF animation showing reflections over time.

        INPUTS:
            output_filename: [str | None]   Output filename (default: timestamped)
            fps:             [int]          Frames per second (default: 3)
        =========================================================================================================
        """
        logger.info("Creating an animation of the moving target and the reflections")

        # Frame times: the native reflection-sample grid (simulation step), restricted to timestamps
        # that have at least one reflection inside the map extent. Specular points sweep the ground at
        # several km/s, so sampling coarser than the simulation step makes them appear to teleport/blink
        # instead of travel; timestamps with no in-view reflection (dead time) are skipped.
        buffer_deg = self.bbox_buffer_deg
        min_lon, max_lon, min_lat, max_lat = self.bounding_box
        west, east = min_lon - buffer_deg, max_lon + buffer_deg
        south, north = min_lat - buffer_deg, max_lat + buffer_deg

        # Only animate constellations that have a selected band: one with no band contributes
        # reflection geometry but no detection, so it must not appear in the animation.
        shown_constellations = [
            name for name, sigs in self.signals_per_constellation.items()
            if sigs and self.reflection_dict_timeSorted.get(name)
        ]
        if not shown_constellations:
            logger.warning("No constellation has both a selected band and reflections; skipping animation.")
            return
        shown_refl_dicts = [self.reflection_dict_timeSorted[name] for name in shown_constellations]

        all_refl_times = sorted({t for refls in shown_refl_dicts for t in refls})

        time_points = []
        for t in all_refl_times:
            in_view = False
            for refls in shown_refl_dicts:
                for ref in refls.get(t, ()):
                    lat, lon = ref.refl_obj.refl_LL_vec[ref.time_index]   # reflection point at this sample
                    if np.isfinite(lat) and np.isfinite(lon) and west <= lon <= east and south <= lat <= north:
                        in_view = True
                        break
                if in_view:
                    break
            if in_view:
                time_points.append(t)

        if not time_points:
            logger.warning("No in-view reflection timestamps found; skipping animation.")
            return

        logger.info(f"Creating {len(time_points)} frames...")

        # Generate frames in memory
        frames = []
        for time_point in tqdm(time_points):
            try:
                img_array = self.show_reflections_and_target_position_at_time_i(
                    time_i=time_point,
                    gnss_constellation_name=shown_constellations,
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
            output_filename = f"animation_{timestamp}.gif"

        stem, ext = os.path.splitext(output_filename)            # tag filename with reflection mode (specular / backscatter)
        output_filename = f"{stem}_{self.reflection_mode}{ext}"

        output_path = os.path.join(Constants.output_folder_path, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Creating gif at {output_path}...")
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
            result: [tuple]   (footprint_polygon, reflection_point_ll) — empty Polygon on failure
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

        if np.isnan(refl_obj.refl_ECI_vec[i]).any():
            self.footprint_stats["rejected_empty"] += 1
            return Polygon(), np.array([])

        footprint_points, reflection_point_ll = refl_obj._compute_footprint(i, self.sat_antenna, frequency, earth_model=self.earth_model)
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
            return Polygon(), reflection_point_ll

            ## DEBUG plot footprint_poly
            # show_footprint(footprint_points)

        # Create a shapely polygon from the points
        if footprint_points.is_empty:
            self.footprint_stats["rejected_empty"] += 1
        return footprint_points, reflection_point_ll

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
                        fixed_polygon = Polygon(zip(fixed_lons, lats, strict=False))
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
                                       delta_lon=None, delta_lat=None, verbose=False, label_detections=False,
                                       band: str | None = None) -> None:
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
            band:               [str | None]   Restrict to one band (default: all bands combined)
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
        title = f"Footprint Overlap for GNSS Constellations {gnss_const_str}"
        if band is not None:
            title += f" (band {band})"
        plt.title(title)
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
        labeled_constellations = set()  # one legend entry per constellation, set on its first plotted point

        for gnss_name in gnss_const:
            if gnss_name in self.target_detect_dict:
                for idx, detect_struct in enumerate(self.target_detect_dict[gnss_name]):
                    # Restrict to one band when requested (idx stays aligned with _identify_wrapping_polygons)
                    if band is not None and detect_struct.band != band:
                        continue
                    # Skip problematic polygons
                    if (gnss_name, idx) in problem_indices:
                        skipped_count += 1
                        continue

                    pos_ll = detect_struct.position_ll
                    if pos_ll is not None and not np.any(np.isnan(pos_ll)):
                        # Label only the first plotted point of each constellation so the color legend is complete
                        point_label = f"{gnss_name} footprint overlap" if gnss_name not in labeled_constellations else ""
                        labeled_constellations.add(gnss_name)
                        # Plot the point
                        ax.scatter(
                            pos_ll[1], pos_ll[0],
                            c=fn.color_gnss_const(gnss_name),
                            s=30, marker="o", alpha=0.7,
                            label=point_label,
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
    frequency: float = np.nan              # carrier frequency [MHz]
    band: str | None = None                # constellation-specific band name (e.g. L1, E1, B1I)
    # Per-event link budget split (populated when ship_dimensions is supplied to MovingTargetDetection).
    pr_ocean_w: float | None = None        # ocean clutter received power [W]
    pr_ship_w: float | None = None         # ship signal received power [W]
    pr_total_w: float | None = None        # pr_ocean_w + pr_ship_w [W]
    scr_db: float | None = None            # signal-to-clutter ratio = 10 log10(pr_ship / pr_ocean) [dB]
    noise_power_w: float | None = None     # raw thermal noise power in configured noise bandwidth (k T B) [W]
    processing_gain_linear: float = 1.0    # detection-equivalent sensitivity factor: B T_coh sqrt(M)
    processing_gain_db: float | None = None           # 10 log10(processing_gain_linear) [dB]
    noise_power_effective_w: float | None = None      # threshold-equivalent noise scale; not mean post-detection noise [W]
    snr_ocean_rf_db: float | None = None   # ocean clutter RF SNR = 10 log10(pr_ocean / noise) [dB]
    snr_ship_rf_db: float | None = None    # ship RF SNR = 10 log10(pr_ship / noise) [dB]
    snr_total_rf_db: float | None = None   # total RF SNR = 10 log10((pr_ocean + pr_ship) / noise) [dB]
    sinr_ship_db: float | None = None      # ship SINR = 10 log10(pr_ship / (pr_ocean + noise)) [dB]
    detection_sinr_threshold_db: float | None = None  # configured ship SINR threshold [dB]
    detection_margin_db: float | None = None          # sinr_ship_db - detection_sinr_threshold_db [dB]
    is_sinr_detection: bool | None = None             # True when detection_margin_db >= 0
    rcs_ocean_m2: float | None = None      # ocean RCS at the reflection point [m^2]
    rcs_ship_m2: float | None = None       # ship RCS (combined plates + dihedrals) [m^2]
    # Forward-scatter (specular mode): the ship shadows its waterline-area patch of the ocean glint.
    forward_shadow_pr_w: float | None = None             # ocean glint power the ship occults [W]
    forward_delta_pr_w: float | None = None              # signed change in received power = -forward_shadow_pr_w [W]
    forward_delta_pr_fraction_db: float | None = None    # 10 log10(forward_shadow_pr / pr_ocean footprint) [dB]
    forward_delta_pr_over_noise_db: float | None = None  # 10 log10(forward_shadow_pr / noise) [dB]
