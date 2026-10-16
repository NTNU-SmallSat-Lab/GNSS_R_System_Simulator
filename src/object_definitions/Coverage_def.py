from dataclasses import dataclass, field
import math
import os
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

import functions.functions as fn
from object_definitions.BackScattPoint_def import BRP
from object_definitions.Constants_def import Constants
from object_definitions.Reflection_def import Reflection
from object_definitions.Static_classes_def import Plotting


def backscatter_swath_half_width_m(range_rx_m, range_tx_m, max_bistatic_bksc_angle_deg):
    """Ground half-width of the near-backscatter acceptance region around the BRP [m].

    A surface point displaced by d perpendicular to the shared line of sight opens the bistatic angle
    by beta ~= d * (1/R_r - 1/R_t) (Tx and Rx sit on the SAME side of the surface, hence the
    difference of curvatures, not the sum), so the beta <= beta_max region has cross-LOS half-width
    beta_max / (1/R_r - 1/R_t). This is the conservative (smaller) semi-axis of the acceptance
    ellipse; the in-plane semi-axis is wider by 1/cos(incidence) and is deliberately not used, so
    the swath never overstates coverage.
    """
    beta_max_deg = float(max_bistatic_bksc_angle_deg)
    if not math.isfinite(beta_max_deg) or not (0.0 < beta_max_deg < 90.0):
        raise ValueError(f"max_bistatic_bksc_angle_deg must be finite and in (0, 90) deg; got {beta_max_deg}")
    range_rx_m = np.asarray(range_rx_m, dtype=float)
    range_tx_m = np.asarray(range_tx_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        curvature_rad_m = 1.0 / range_rx_m - 1.0 / range_tx_m       # d(beta)/d(cross-LOS offset) [rad/m]
        half_width_m = np.radians(beta_max_deg) / curvature_rad_m   # beta_max / (1/R_r - 1/R_t)
    return np.where(np.isfinite(half_width_m) & (half_width_m > 0.0), half_width_m, np.nan)


class Coverage:
    """Count contiguous reflection-track visits per grid cell over a region (the area-coverage mode).

    Builds (or loads) a gridded version of an input shapefile, then -- per GNSS constellation -- connects
    each contiguous run of valid samples into a track and counts one visit per intersected cell and run.
    In ``specular`` mode the track is the specular-point line; in ``backscatter`` mode it is the BRP
    track buffered by the near-backscatter swath (all cells whose bistatic angle to some (Tx, Rx) pair
    stays below ``max_bistatic_bksc_angle_deg``).
    Coverage is a **geometric** metric: the band's frequency and EIRP only feed the antenna-field-of-view
    finiteness gate, not a detection threshold, so the result is per constellation, not per band.

    Parameters
    ----------
    sat_const : SatelliteConstellation
        LEO receiver constellation.
    gnss_const : dict
        ``{name: SatelliteConstellation}`` of GNSS transmitter constellations.
    shapefile : str
        Path to the region shapefile (the grid is generated alongside it).
    grid_size_deg : float or None, optional
        Cell size in degrees (equiangular grid). Mutually exclusive with ``grid_size_m``.
    grid_size_m : float or None, optional
        Cell size in metres (equidistant grid).
    overwrite_shapefile : bool, optional
        Regenerate the gridded shapefile even if it exists.
    earth_model : {"spherical", "elliptical"}, optional
        Surface model for the reflection geometry.
    link_budget_settings : dict or BistaticLinkBudgetSettings or None, optional
        Rx-system link-budget settings (used only for the finiteness gate).
    max_central_angle_deg : float, keyword-only
        Maximum Tx-Rx central angle [deg].
    selected_signals : list of dict or None, keyword-only
        Validated INI band plan; deduplicated to one entry per constellation.
    reflection_type : {"specular", "backscatter"}, keyword-only
        Reflection geometry the coverage counts (default "specular").
    max_incidence_angle_deg : float or None, keyword-only
        Maximum BRP incidence angle [deg]; required for backscatter.
    max_bistatic_bksc_angle_deg : float or None, keyword-only
        Maximum bistatic angle at the surface for a cell to count as near-backscatter-covered [deg];
        sets the swath half-width around the BRP track. Required for backscatter, unused for specular.
    """

    def __init__(
        self,
        sat_const,
        gnss_const,
        shapefile,
        grid_size_deg=None,
        grid_size_m=None,
        overwrite_shapefile=False,
        earth_model="spherical",
        link_budget_settings=None,
        *,
        max_central_angle_deg,
        selected_signals: list[dict] | None = None,
        reflection_type: str = "specular",
        max_incidence_angle_deg: float | None = None,
        max_bistatic_bksc_angle_deg: float | None = None,
    ) -> None:
        """Build (or load) the gridded shapefile and prepare the per-constellation coverage vectors."""
        # Path to the input shapefile (outline of the area of interest)
        self.shp_input_file_path = os.path.join(Constants.base_dir, shapefile)
        # GeoDataFrame for the input shapefile (outline of the area of interest)
        self.gdf = gpd.read_file(self.shp_input_file_path)
        if grid_size_m is None:
            self.grid_type = "equiangular"  # Type of grid (equiangular or equidistant)
            self.gridded_shapefile_path = self.shp_input_file_path.replace(
                ".shp", "_gridded_deg.shp"
            )  # Path to the shapefile with grid-cells (fishnet), This file is generated by self.create_coverage_shapefile_deg()
        else:
            self.grid_type = "equidistant"  # Type of grid (equiangular or equidistant)
            self.gridded_shapefile_path = self.shp_input_file_path.replace(
                ".shp", "_gridded_m.shp"
            )  # Path to the shapefile with grid-cells (fishnet), This file is generated by self.create_coverage_shapefile_m()
        self.grid_gdf = None  # GeoDataFrame for the grid-celled shapefile (fishnet) (actual "file" in memory of the file self.gridded_shapefile_path)

        self.grid_size_deg = grid_size_deg  # Size of the grid cells in degrees latitude / longitude (if grid_size_m is None)
        self.grid_size_m = grid_size_m  # Size of the grid cells in meters
        self.bounding_box = None   # Bounding box for filtering points (min_lon, max_lon, min_lat, max_lat) (initialized in )
        self.reflection_dict = {}  # Reflection dictionary for storing reflection vector containing reflection points for each GNSS constellations
        self.coverage_dict = {}  # Coverage vector for storing coverage information over each grid-cell in the gridded shapefile

        self.sat_constellation = sat_const  # Satellite constellation object
        self.gnss_constellation = gnss_const  # Dictionary with GNSS constellation objects
        self.link_budget_settings = link_budget_settings
        self.max_central_angle_deg = float(max_central_angle_deg)
        # Area coverage is a GEOMETRIC metric (specular-reflection revisits per cell), so it is computed
        # per CONSTELLATION, not per band -- the band's frequency/EIRP only feed the antenna-FOV finiteness
        # gate, not a detection threshold (see TODO.md "Area coverage: geometric now, detectable later").
        # [ANTENNA].bands selects which constellations participate; dedupe to the first selected band per
        # constellation for the representative freq/EIRP. Empty when constructed without a plan (e.g. tests).
        self.coverage_constellations = []
        _seen_constellations = set()
        for s in selected_signals or []:
            constellation = s["constellation"]
            if constellation in self.gnss_constellation and constellation not in _seen_constellations:
                _seen_constellations.add(constellation)
                self.coverage_constellations.append(
                    {"constellation": constellation, "freq_hz": s["freq_hz"], "eirp_dbw": s["eirp_dbw"]}
                )
        if earth_model not in ["spherical", "elliptical"]:
            raise ValueError("earth_model must be either 'spherical' or 'elliptical'")
        else:
            self.earth_model = earth_model

        if reflection_type not in ["specular", "backscatter"]:
            raise ValueError("reflection_type must be either 'specular' or 'backscatter'")
        self.reflection_type = reflection_type
        if reflection_type == "backscatter":
            if max_incidence_angle_deg is None:
                raise ValueError("max_incidence_angle_deg is required for backscatter area coverage.")
            if max_bistatic_bksc_angle_deg is None:
                raise ValueError(
                    "[COVERAGE].max_bistatic_bksc_angle_deg is required for backscatter area coverage: "
                    "it sets the near-backscatter acceptance half-angle (the swath width around the BRP track)."
                )
            self.max_incidence_angle_deg = float(max_incidence_angle_deg)
            beta_max = float(max_bistatic_bksc_angle_deg)
            if not math.isfinite(beta_max) or not (0.0 < beta_max < 90.0):
                raise ValueError(f"max_bistatic_bksc_angle_deg must be finite and in (0, 90) deg; got {beta_max}")
            self.max_bistatic_bksc_angle_deg = beta_max
        else:
            if max_bistatic_bksc_angle_deg is not None:
                warnings.warn(
                    "max_bistatic_bksc_angle_deg is only used for backscatter area coverage; "
                    "it is ignored for reflection_type = 'specular'.",
                    stacklevel=2,
                )
            self.max_incidence_angle_deg = None
            self.max_bistatic_bksc_angle_deg = None

        self.initialize_coverage(overwrite_shapefile)

    def initialize_coverage(self, overwrite_shapefile) -> None:
        #############################
        # Loading "Gridded Shapefile"
        #############################
        shapefile_exists = os.path.exists(self.gridded_shapefile_path)
        if shapefile_exists and not overwrite_shapefile:
            print(f"Grided shapefile already exists at: {self.gridded_shapefile_path} \n        Loading existing gridded shapefile.")
        else:
            if shapefile_exists:
                print("Overwriting existing gridded shapefile.")
                os.remove(self.gridded_shapefile_path)
            self._create_gridded_shapefile()

        # Check if the gridded shapefile was created successfully and load it if it exists
        if os.path.exists(self.gridded_shapefile_path):
            self.grid_gdf = gpd.read_file(self.gridded_shapefile_path)

        # One empty per-cell coverage vector per constellation (filled per band at compute time).
        for gnss_name in self.gnss_constellation:
            self.coverage_dict[gnss_name] = self._fresh_coverage_vector()

        self.bounding_box = self._get_bounding_box()

    def _fresh_coverage_vector(self) -> list:
        """Build an empty per-cell CoverageStruct vector with each cell's centroid and boundary set."""
        vector = [CoverageStruct() for _ in range(len(self.grid_gdf))]
        for idx in range(len(self.grid_gdf)):
            centroid = self.grid_gdf.loc[idx].geometry.centroid
            vector[idx].position = (centroid.x, centroid.y)
            vector[idx].shape = self.grid_gdf.loc[[idx]].boundary
        return vector

    def _get_bounding_box(self) -> tuple:
        """
        =========================================================================================================
        Get the bounding box of the coverage area, padded by 5 degrees on each side.

        OUTPUT:
            bbox: [tuple | None]   (min_lon, max_lon, min_lat, max_lat) — None if grid is equiangular
        =========================================================================================================
        """
        delta_lat = 5
        delta_lon = 5
        if self.grid_size_m is not None:
            return (math.floor(self.grid_gdf.bounds.minx.min() - delta_lon),
                    math.ceil(self.grid_gdf.bounds.maxx.max() + delta_lon),
                    math.floor(self.grid_gdf.bounds.miny.min() - delta_lat),
                    math.ceil(self.grid_gdf.bounds.maxy.max() + delta_lat))
        else:
            return None

    def _calculate_coverage(self, times, constellation) -> None:
        """
        =========================================================================================================
        Filter reflection points to those inside the bounding box, then accumulate per-cell coverage
        (times, satellite IDs, incidence angles, received power) into `self.coverage_dict[constellation]`.

        INPUTS:
            times:         [list[datetime]]   Time vector of the reflection samples
            constellation: [str]              Constellation whose reflections to accumulate (the current signal)
        =========================================================================================================
        """
        if self.reflection_dict is None:
            print("Reflection dictionary is empty. Coverage is zero.")
            return

        # Pre-compute KD-Tree for grid centroids
        #        kdtree = self._precompute_grid_kdtree()

        # Accumulate only the current signal's constellation. reflection_dict may also hold other
        # constellations from earlier signals (kept for the whole-earth reflection plots), whose
        # coverage is already computed -- re-processing them here would double-count.
        for gnss_name in [constellation]:
            refl_vector = self.reflection_dict[gnss_name]  # Reflection vector for the current GNSS constellation

            for i in range(len(refl_vector)):
                validity_mask, ref_points_in_box = self.validity_mask(refl_vector[i], self.bounding_box)
                # Assembly of points in the bounding box for one GNSS-R satellite and one GNSS satellite
                refl_clusters = self._get_reflection_clusters(refl_vector[i], validity_mask)
                for cluster in refl_clusters:
                    intersecting_rectangles = []
                    if len(cluster) == 0:
                        continue

                    # Extract valid times and positions for this cluster. The track geometry is the
                    # specular/BRP polyline; in backscatter mode it is buffered by the near-backscatter
                    # acceptance swath (a cell counts while its bistatic angle to the pair stays below
                    # max_bistatic_bksc_angle_deg). _cluster_track_geometry is shared with the plotting.
                    lon_lat = ref_points_in_box[cluster][:, [1, 0]]
                    track_geometry = self._cluster_track_geometry(refl_vector[i], cluster, ref_points_in_box)
                    for idx, polygon in self.grid_gdf.iterrows():
                        intersection = polygon["geometry"].intersection(track_geometry)
                        if not intersection.is_empty:
                            intersecting_rectangles.append(idx)

                    if len(intersecting_rectangles) > 0:
                        # Save results
                        with warnings.catch_warnings():  # TODO: This could be improved! resp. better documented
                            warnings.simplefilter("ignore", UserWarning)
                            centroids_array = np.array([(centroid.x, centroid.y) for centroid in self.grid_gdf.loc[intersecting_rectangles].centroid])
                        distances = cdist(centroids_array, lon_lat)
                        nearest_indices = np.argmin(distances, axis=1)
                        times_at_nearest_points = [times[k] for k in cluster[nearest_indices]]
                        angle_at_nearest_points = refl_vector[i].refl_ang_vec[cluster[nearest_indices]]
                        power_at_nearest_points = refl_vector[i].power[cluster[nearest_indices]]

                        # Save the nearest points
                        for rect_idx, time, angle, power in zip(
                            intersecting_rectangles,
                            times_at_nearest_points,
                            angle_at_nearest_points,
                            power_at_nearest_points,
                            strict=False,
                        ):
                            if not np.isfinite(power):
                                continue
                            # Assign time to coverage_vec
                            self.coverage_dict[gnss_name][rect_idx].times.append(time)
                            self.coverage_dict[gnss_name][rect_idx].tx_satID.append(refl_vector[i].tx_satID)
                            self.coverage_dict[gnss_name][rect_idx].rx_satID.append(refl_vector[i].rx_satID)
                            self.coverage_dict[gnss_name][rect_idx].angles.append(angle)
                            self.coverage_dict[gnss_name][rect_idx].power.append(power)

    def _cluster_swath_half_width_deg(self, reflection_obj, cluster, valid_position) -> float:
        """Near-backscatter swath half-width for one track cluster, in degrees at its mean latitude."""
        refl_eci = np.asarray(reflection_obj.refl_ECI_vec, dtype=float)[cluster]      # (k, 3)
        rx_eci = np.asarray(reflection_obj.sat_pos_vec, dtype=float)[:, cluster].T    # (k, 3)
        tx_eci = np.asarray(reflection_obj.gnss_pos_vec, dtype=float)[:, cluster].T   # (k, 3)
        range_rx_m = np.linalg.norm(rx_eci - refl_eci, axis=1)                        # surface -> Rx slant range
        range_tx_m = np.linalg.norm(tx_eci - refl_eci, axis=1)                        # surface -> Tx slant range
        half_width_m = backscatter_swath_half_width_m(range_rx_m, range_tx_m, self.max_bistatic_bksc_angle_deg)
        if not np.isfinite(half_width_m).any():
            return float("nan")
        half_width_min_m = float(np.nanmin(half_width_m))                             # conservative: narrowest sample
        mean_lat_deg = float(np.nanmean(valid_position[:, 0]))
        return float(fn.delta_lat_from_distance(mean_lat_deg, half_width_min_m))      # smaller degree equivalent

    def _cluster_track_geometry(self, reflection_obj, cluster, ref_points_in_box):
        """Coverage geometry for one in-box track cluster: the specular/BRP polyline, buffered by the
        near-backscatter swath half-width in backscatter mode. Shared by the coverage counting and the
        map plotting, so the drawn swath is exactly the region that is counted."""
        valid_position = ref_points_in_box[cluster]                          # (k, 2) = (lat, lon)
        lon_lat = valid_position[:, [1, 0]]
        geom = Point(lon_lat[0]) if len(cluster) == 1 else LineString(lon_lat)
        if self.reflection_type == "backscatter":
            swath_deg = self._cluster_swath_half_width_deg(reflection_obj, cluster, valid_position)
            if np.isfinite(swath_deg) and swath_deg > 0.0:
                geom = geom.buffer(swath_deg)                                # near-backscatter acceptance swath
        return geom

    def compute_coverage_for_constellation(self, constellation, freq_hz, eirp_dbw, verbose=False):
        """
        =========================================================================================================
        Compute reflection points and per-cell GEOMETRIC coverage for one GNSS constellation.

        Geometry is solved per (LEO, GNSS) pair for `constellation`; the link budget at `freq_hz` /
        `eirp_dbw` is used only as the antenna-FOV finiteness gate (coverage is geometric, not
        detectability-gated -- see TODO.md). Resets this constellation's coverage vector; reflections
        accumulate across calls so the whole-earth reflection plots show every computed constellation.

        INPUTS:
            constellation: [str]     GNSS constellation to cover
            freq_hz:       [float]   Representative carrier frequency [Hz] (antenna-FOV gate only)
            eirp_dbw:      [float]   Representative transmit EIRP [dBW]
            verbose:       [bool]    Print progress (default: False)
        =========================================================================================================
        """
        times = [time.utc_datetime() for time in self.sat_constellation.times_skyfield]
        gnss_const = self.gnss_constellation[constellation]

        # Fresh coverage accumulator for this constellation. Reflections from earlier signals are kept
        # in reflection_dict so the end-of-run whole-earth reflection plots show every computed constellation.
        self.coverage_dict[constellation] = self._fresh_coverage_vector()

        total_computations = len(self.sat_constellation.satellites) * len(gnss_const.satellites)
        with tqdm(total=total_computations, desc=f"Reflections {constellation}", unit="reflection") as pbar:
            if verbose:
                print(f'\nCalculating reflections for {constellation} ({freq_hz / 1e6:.2f} MHz)')
            reflections_vec = []
            for sat_no, sat in enumerate(self.sat_constellation.satellites):
                pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {constellation}")
                antenna_obj = sat.gnss_r_antenna if hasattr(sat, "gnss_r_antenna") else None

                for i, gnss_sat in enumerate(gnss_const.satellites):
                    if i % 5 == 0:
                        pbar.set_description(f"GNSS-R Sat-{sat_no + 1} with {constellation} [{i}/{len(gnss_const.satellites)}]")
                    if self.reflection_type == "backscatter":
                        refl_gnss = BRP(
                            gnss_sat.satID,
                            sat.satID,
                            earth_model=self.earth_model,
                            max_central_angle_deg=self.max_central_angle_deg,
                            max_incidence_angle_deg=self.max_incidence_angle_deg,
                            gnss_const=constellation,
                        )
                        refl_gnss.compute_brp_coordinates(
                            gnss_sat,
                            sat,
                            fn.to_utc_datetime_array(times),
                            tx_eirp_dbw=eirp_dbw,
                            antenna=antenna_obj,
                            frequency_hz=freq_hz,
                            link_budget_settings=self.link_budget_settings,
                        )
                    else:
                        refl_gnss = Reflection(
                            gnss_sat.satID,
                            sat.satID,
                            earth_model=self.earth_model,
                            max_central_angle_deg=self.max_central_angle_deg,
                            gnss_const=constellation,
                        )
                        refl_gnss.compute_reflection_coordinates(
                            gnss_sat,
                            sat,
                            times,
                            tx_eirp_dbw=eirp_dbw,
                            antenna=antenna_obj,
                            frequency_hz=freq_hz,
                            link_budget_settings=self.link_budget_settings,
                        )
                    reflections_vec.append(refl_gnss)
                    pbar.update(1)
            self.reflection_dict[constellation] = reflections_vec
        self._calculate_coverage(times, constellation)

    def calculate_statistics(self, gnss_constellation=None):
        """
        =========================================================================================================
        Calculate statistics from coverage data and print them.

        INPUTS:
            gnss_constellation: [str | None]   Comma-separated list of GNSS constellations to summarize,
                                               e.g. "GPS, GALILEO" — defaults to all loaded constellations
        =========================================================================================================
        """
        # Process the gnss_constellation input
        gnss_const = fn.normalize_gnss_constellation_arg(gnss_constellation, self.gnss_constellation.keys())

        print(f"\nStatistics for GNSS constellations: {', '.join(gnss_const)}")
        print("=" * 50)

        # Calculate statistics for each GNSS constellation
        for gnss_name in gnss_const:
            if gnss_name in self.coverage_dict:
                # Average signal power - collect all power values across all grid cells
                all_powers = []
                for idx in range(len(self.grid_gdf)):
                    all_powers.extend(self.coverage_dict[gnss_name][idx].power)

                if all_powers:
                    average_power = np.mean(all_powers)
                    min_power = np.min(all_powers)
                    max_power = np.max(all_powers)
                    print(f"\nGNSS Constellation: {gnss_name}")
                    print(f"Average Power: {average_power:.2f}")
                    print(f"Min Power: {min_power:.2f}")
                    print(f"Max Power: {max_power:.2f}")
                else:
                    print(f"\nGNSS Constellation: {gnss_name}")
                    print("No power data available")

                # Coverage percentage
                intersection_count = {idx: len(self.coverage_dict[gnss_name][idx].times) for idx in range(len(self.grid_gdf))}
                non_zero_count = sum(1 for value in intersection_count.values() if value > 0)
                coverage_percentage = non_zero_count / len(intersection_count) * 100
                print(f"Coverage percentage: {coverage_percentage:.2f}%")
                print(f"Areas covered: {non_zero_count} out of {len(intersection_count)} grid cells")
            else:
                print(f"\nGNSS Constellation: {gnss_name}")
                print("No data available for this constellation")

        # Calculate combined statistics for all selected constellations
        if len(gnss_const) > 1:
            print("\nCombined Statistics for All Selected Constellations")
            print("-" * 50)

            # Combined coverage percentage
            combined_intersection_count = dict.fromkeys(range(len(self.grid_gdf)), 0)
            for gnss_name in gnss_const:
                if gnss_name in self.coverage_dict:
                    for idx in range(len(self.grid_gdf)):
                        if len(self.coverage_dict[gnss_name][idx].times) > 0:
                            combined_intersection_count[idx] = 1  # Mark as covered

            combined_coverage_count = sum(combined_intersection_count.values())
            combined_coverage_percentage = combined_coverage_count / len(combined_intersection_count) * 100
            print(f"Combined coverage percentage: {combined_coverage_percentage:.2f}%")
            print(f"Total areas covered: {combined_coverage_count} out of {len(combined_intersection_count)} grid cells")

            # Combined hit statistics
            total_hits = 0
            for gnss_name in gnss_const:
                if gnss_name in self.coverage_dict:
                    for idx in range(len(self.grid_gdf)):
                        total_hits += len(self.coverage_dict[gnss_name][idx].times)

            print(f"Total number of reflections: {total_hits}")
            if combined_coverage_count > 0:
                print(f"Average reflections per covered cell: {total_hits / combined_coverage_count:.2f}")

    def plot_reflection_points(self):
        """
        =========================================================================================================
        Plot reflection points for all GNSS constellations and save to OUTPUT/.
        =========================================================================================================
        """
        for gnss_costl_name, refl_vec in self.reflection_dict.items():
            plot_name = f"Reflection_points_{gnss_costl_name}.png"
            gnss_constellation = self.gnss_constellation[gnss_costl_name]
            Plotting.plot_reflection_traces_gnss_const(
                self.sat_constellation,
                gnss_constellation,
                refl_vec,
                file_name=plot_name,
            )

        # Only the constellations actually computed (those with a selected band) are in reflection_dict.
        present = {name: self.gnss_constellation[name] for name in self.reflection_dict}
        const_names_str = "_".join(present.keys())
        Plotting.plot_reflection_traces_total(
            self.sat_constellation,
            present,
            self.reflection_dict,
            file_name=f"Reflection_traces_{const_names_str}.png",
        )

    def plot_coverage_shapefile(self, gnss_constellation=None, file_name="Coverage1.png", title=None):
        """
        =========================================================================================================
        Plot the coverage area of the GNSS constellation(s) and the reflection vectors.

        INPUTS:
            gnss_constellation: [str | None]   Comma-separated list, e.g. "GPS, GLONASS" (default: all)
            file_name:          [str]          Output file name (default: "Coverage1.png")
            title:              [str | None]   Plot title (optional)
        =========================================================================================================
        """
        [min_lon, max_lon, min_lat, max_lat] = self.bounding_box

        # If a GNSS constellation is provided, plot only the coverage area of that constellation, otherwise plot all constellations
        gnss_const = fn.normalize_gnss_constellation_arg(gnss_constellation, self.gnss_constellation.keys())

        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_aspect("equal", adjustable="box")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_global()
        ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5])  # Add padding to the extent
        #        coverage_obj.grid_gdf.plot(ax=ax, edgecolor='lightgray', facecolor='lightgray', linestyle='-', alpha=0.5, label="Grid Background")
        self.gdf.plot(
            ax=ax,
            edgecolor="lightgray",
            facecolor="lightgray",
            linestyle="-",
            alpha=0.5,
            label="Coverage Area",
            aspect=None,
        )
        # Draw only the in-box, contiguous track segments used by the coverage counting (the full
        # refl_LL_vec would add out-of-region / pole-wrapping fragments unrelated to this map).
        for gnss_constel in gnss_const:
            color = fn.color_gnss_const(gnss_constel)
            for refl in self.reflection_dict[gnss_constel]:
                validity_mask, ref_points_in_box = self.validity_mask(refl, self.bounding_box)
                for cluster in self._get_reflection_clusters(refl, validity_mask):
                    if len(cluster) < 2:
                        continue
                    segment = ref_points_in_box[cluster]  # (k, 2) = (lat, lon), finite within the box
                    # Backscatter: fill the near-backscatter acceptance swath (the counted region) at
                    # 90% transparency, so the coloured cells line up with the drawn track.
                    if self.reflection_type == "backscatter":
                        swath = self._cluster_track_geometry(refl, cluster, ref_points_in_box)
                        if swath.geom_type in ("Polygon", "MultiPolygon"):
                            ax.add_geometries([swath], crs=ccrs.PlateCarree(), facecolor=color, edgecolor="none", alpha=0.1)
                    ax.plot(
                        segment[:, 1],
                        segment[:, 0],
                        color=color,
                        linewidth=0.8,
                        transform=ccrs.PlateCarree(),
                    )

        file_path = os.path.join(Constants.output_folder_path, file_name)
        if title is None:
            plural = "s" if len(gnss_const) > 1 else ""
            title_string = f"Coverage Area with Reflection Vectors for {', '.join(gnss_const)} constellation{plural}"
        else:
            title_string = title
        plt.title(title_string)
        plt.savefig(file_path, dpi=600)

    def plot_coverage_gridded_shapefile(self, gnss_constellation=None, file_name="Coverage_grid.png", title=None):
        """
        =========================================================================================================
        Plot the coverage area of the GNSS constellation(s) and the reflection vectors using the
        gridded shapefile.

        INPUTS:
            gnss_constellation: [str | None]   Comma-separated list, e.g. "GPS, GLONASS" (default: all)
            file_name:          [str]          Output file name (default: "Coverage_grid.png")
            title:              [str | None]   Plot title (optional)
        =========================================================================================================
        """

        def _get_color(count):
            if count == 1:
                return "aqua", 0.3  # Color for 1-time intersection
            elif count == 2:
                return "blue", 0.3  # Color for 2-time intersection
            elif count > 2:
                return "orange", 0.3
            else:
                return "none", 0  # If the cell isn't intersected at all

        # If a GNSS constellation is provided, plot only the coverage area of that constellation, otherwise plot all constellations
        gnss_const = fn.normalize_gnss_constellation_arg(gnss_constellation, self.gnss_constellation.keys())

        # Create intersection count dictionary for each grid cell
        intersection_count = dict.fromkeys(self.grid_gdf.index, 0)

        for gnss_name in gnss_const:
            if gnss_name in self.coverage_dict:
                for i in range(len(self.grid_gdf)):
                    # Count the number of times each grid cell is intersected
                    intersection_count[i] += len(self.coverage_dict[gnss_name][i].times)

        # Define color based on the count
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_aspect("equal", adjustable="box")

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_global()

        # Map extent: the region of interest plus a margin. The counting bounding box is padded by 5 deg,
        # which would also draw many tracks that never reach the region; lines outside the axes are clipped.
        pad_deg = 1.5
        min_x, min_y, max_x, max_y = self.grid_gdf.total_bounds
        ax.set_extent([min_x - pad_deg, max_x + pad_deg, min_y - pad_deg, max_y + pad_deg])

        # Plot the grid background
        self.grid_gdf.plot(
            ax=ax,
            edgecolor="lightgray",
            facecolor="lightgray",
            linestyle="-",
            alpha=0.5,
            label="Grid Background",
            aspect=None,
            zorder=1,
        )

        # Group cells by intersection count and plot each group in a single batch call —
        # a per-cell .plot() loop over thousands of cells is the main bottleneck here.
        counts = np.array([intersection_count[i] for i in self.grid_gdf.index])
        for mask, fill_color in [
            (counts == 1, "aqua"),
            (counts == 2, "blue"),
            (counts > 2, "orange"),
        ]:
            if mask.any():
                self.grid_gdf[mask].plot(ax=ax, color=fill_color, edgecolor="red", alpha=0.3, aspect=None, zorder=3)

        # Plot reflection tracks -- only the in-box, contiguous segments that the coverage counting
        # uses, so the drawn tracks correspond to the coloured cells. Plotting the full refl_LL_vec
        # would draw out-of-region and pole-wrapping segments (e.g. a BRP track that lies over the
        # equator or the south pole for this window) clipped onto the map, which are unrelated to the
        # coverage counted here.
        # Specular point tracks are thin lines and numerous; only the parts within 1 deg of the region can colour a
        # cell, so the rest (e.g. over land) is not drawn. Backscatter swaths are few and drawn in full.
        roi_zone = None
        if self.reflection_type == "specular":
            cells = self.grid_gdf.geometry
            roi_zone = (cells.union_all() if hasattr(cells, "union_all") else cells.unary_union).buffer(1.0)

        plotted_consts = []
        for gnss_name in gnss_const:
            if gnss_name not in self.reflection_dict:
                continue
            color = fn.color_gnss_const(gnss_name)
            plotted_consts.append(gnss_name)
            for refl in self.reflection_dict[gnss_name]:
                validity_mask, ref_points_in_box = self.validity_mask(refl, self.bounding_box)
                for cluster in self._get_reflection_clusters(refl, validity_mask):
                    if len(cluster) < 2:
                        continue
                    segment = ref_points_in_box[cluster]  # (k, 2) = (lat, lon), finite within the box
                    # Backscatter: fill the near-backscatter acceptance swath (the counted region) at
                    # 90% transparency, so the coloured cells line up with the drawn track.
                    if self.reflection_type == "backscatter":
                        swath = self._cluster_track_geometry(refl, cluster, ref_points_in_box)
                        if swath.geom_type in ("Polygon", "MultiPolygon"):
                            ax.add_geometries([swath], crs=ccrs.PlateCarree(), facecolor=color, edgecolor="none", alpha=0.1, zorder=2)
                    if roi_zone is None:
                        parts = [LineString(segment[:, [1, 0]])]
                    else:
                        clipped = LineString(segment[:, [1, 0]]).intersection(roi_zone)
                        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                    for part in parts:
                        if part.is_empty or part.geom_type != "LineString":
                            continue
                        xs, ys = part.xy
                        ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.75, transform=ccrs.PlateCarree(), zorder=2)

        # Add labels and gridlines
        if title is None:
            title_string = "Coverage of grid cells by reflection lines"
        else:
            title_string = title
        plt.title(title_string)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Legend: cell visit counts, uncovered cells, and the track colour of each constellation
        # (plus the acceptance swath in backscatter mode), placed outside the map so it hides no data.
        legend_elements = [
            Patch(facecolor="lightgray", edgecolor="lightgray", alpha=0.5, label="not visited"),
            Patch(facecolor="aqua", edgecolor="aqua", alpha=0.3, label="1 visit"),
            Patch(facecolor="blue", edgecolor="blue", alpha=0.3, label="2 visits"),
            Patch(facecolor="orange", edgecolor="orange", alpha=0.3, label="3+ visits"),
        ]
        track_name = "specular point" if self.reflection_type == "specular" else "BRP"
        for gnss_name in plotted_consts:
            legend_elements.append(Line2D([0], [0], color=fn.color_gnss_const(gnss_name), linewidth=1.5, label=f"{gnss_name} {track_name} track"))
            if self.reflection_type == "backscatter":
                legend_elements.append(Patch(facecolor=fn.color_gnss_const(gnss_name), alpha=0.25, label=f"{gnss_name} acceptance swath"))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True)

        # Keep Cartopy map proportions independent of the latitude of each coverage group.
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        file_path = os.path.join(Constants.output_folder_path, file_name)
        plt.savefig(file_path, dpi=200, bbox_inches="tight")

        print(f"Coverage plot saved to {file_path}")

        # Close the plot to free up memory
        plt.close()

    def _create_gridded_shapefile(self):
        """
        =========================================================================================================
        Dispatch to the equidistant or equiangular shapefile generator based on `grid_size_m`.
        =========================================================================================================
        """
        if self.grid_size_m is None:
            print(f'Creating new gridded shapefile with equal "latitude / longitude" spacing at: {self.gridded_shapefile_path}')
            self._create_coverage_shapefile_deg()
            self.grid_type = "equiangular"
        else:
            print(f'Creating new gridded shapefile with equal "meters" spacing at: {self.gridded_shapefile_path}')
            self._create_coverage_shapefile_m()
            self.grid_type = "equidistant"

    def _create_coverage_shapefile_m(self):
        """
        =========================================================================================================
        Create a gridded shapefile with equal-sized cells in meters.
        Each cell has sides of approximately `grid_size_m` meters, adjusted for latitude to maintain
        equal physical dimensions.
        =========================================================================================================
        """
        # Check if grid_size_m is None, if so skip this function
        if self.grid_size_m is None:
            print("Grid size in meters is not specified. Skipping the creation of the gridded shapefile.")
            return

        # Get the bounds of the shapefile (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = self.gdf.total_bounds

        # Generate the gridded shapefile (list of polygons)
        grid_polygons = []

        current_lat = min_lat  # Start from the minimum latitude
        while current_lat < max_lat:
            delta_lat = fn.delta_lat_from_distance(current_lat, self.grid_size_m)
            current_lon = min_lon  # Start from the minimum longitude
            while current_lon < max_lon:
                delta_lon = fn.delta_lon_from_distance(current_lat, self.grid_size_m)
                # Create a polygon for the current grid cell
                cell_polygon = Polygon(
                    [
                        (current_lon, current_lat),
                        (current_lon + delta_lon, current_lat),
                        (current_lon + delta_lon, current_lat + delta_lat),
                        (current_lon, current_lat + delta_lat),
                    ]
                )
                grid_polygons.append(cell_polygon)
                current_lon += delta_lon  # Move to the next longitude
            current_lat += delta_lat  # Move to the next latitude

        grid_gdf = gpd.GeoDataFrame({"geometry": grid_polygons}, crs=self.gdf.crs)
        # Filter out grid cells that do not intersect with the input shapefile
        grid_gdf = grid_gdf[grid_gdf.intersects(self.gdf.union_all())]

        # Save the grid as a shapefile
        grid_gdf.to_file(self.gridded_shapefile_path, driver="ESRI Shapefile")
        self.grid_gdf = grid_gdf  # Store the grid GeoDataFrame in the class

    def _create_coverage_shapefile_deg(self):
        """
        =========================================================================================================
        Create a gridded shapefile with equiangular cells (constant degrees in latitude/longitude).
        Cell size is taken from `self.grid_size_deg`.
        =========================================================================================================
        """
        # Robustness check, if grid_size_deg is None skip this function
        if self.grid_size_deg is None:
            print("Grid size is not specified. Skipping the creation of the gridded shapefile.")
            return

        # Get the bounds of the shapefile (min_lon, miny, maxx, maxy)
        min_lon, min_lat, max_lon, max_lat = self.gdf.total_bounds

        # Generate the gridded shapefile (list of polygons)
        grid_polygons = []
        for x0 in np.arange(min_lon, max_lon, self.grid_size_deg):
            for y0 in np.arange(min_lat, max_lat, self.grid_size_deg):
                x1 = x0 + self.grid_size_deg
                y1 = y0 + self.grid_size_deg
                grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

        # Create a GeoDataFrame for the gridded shapefile (fishnet)
        grid_gdf = gpd.GeoDataFrame({"geometry": grid_polygons}, crs=self.gdf.crs)
        # Filter out grid cells that do not intersect with the input shapefile
        grid_gdf = grid_gdf[grid_gdf.intersects(self.gdf.union_all())]

        # Save the grid as a shapefile
        grid_gdf.to_file(self.gridded_shapefile_path, driver="ESRI Shapefile")
        self.grid_gdf = grid_gdf  # Store the grid GeoDataFrame in the class

    def save_gridded_shapefile_plot(self):
        """
        =========================================================================================================
        Save the gridded shapefile and plot the fishnet overlay on the original shapefile.
        =========================================================================================================
        """
        print(f"Gridded shapefile saved as: {self.gridded_shapefile_path}")

        # Plotting the results
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal", adjustable="box")

        # Plot the original shapefile
        self.grid_gdf.plot(
            ax=ax,
            color="lightblue",
            edgecolor="black",
            alpha=0.5,
            label="Original Shapefile",
            aspect=None,
        )

        # Plot the gridded shapefile grid
        self.grid_gdf.plot(
            ax=ax,
            edgecolor="red",
            facecolor="none",
            linestyle="-",
            label="Fishnet Grid",
            aspect=None,
        )

        # Add title and legend
        ax.set_title("Fishnet Grid Overlayed on Original Shapefile")
        plt.savefig("rectangles_from_grid_points.png")

    def _precompute_grid_kdtree(self):
        """
        =========================================================================================================
        Pre-compute a KD-Tree for efficient spatial queries on grid centroids (cached on `self`).
        =========================================================================================================
        """
        if not hasattr(self, "_grid_kdtree") or self._grid_kdtree is None:
            centroids = np.array([(centroid.x, centroid.y) for centroid in self.grid_gdf.centroid])
            self._grid_kdtree = cKDTree(centroids)
            self._grid_centroids = centroids
        return self._grid_kdtree

    @staticmethod
    def validity_mask(reflection_obj, bounding_box):
        """
        =========================================================================================================
        Create a validity mask for the reflection points based on the bounding box.

        INPUTS:
            reflection_obj: [Reflection]   Reflection object containing the reflection points
            bounding_box:   [tuple]        (min_lon, max_lon, min_lat, max_lat)

        OUTPUT:
            result: [tuple]   (validity_mask, refl_in_bounding_box)
                              - validity_mask:        Boolean array, True where the point is inside the box
                              - refl_in_bounding_box: Reflection coordinates with NaNs for outside-box points
        =========================================================================================================
        """
        refl_LL_vec = np.array(reflection_obj.refl_LL_vec)

        finite = np.isfinite(refl_LL_vec).all(axis=1)
        if finite.any():
            if bounding_box is None:
                validity_mask = finite
                refl_in_bounding_box = refl_LL_vec.copy()
                refl_in_bounding_box[~validity_mask] = np.nan
                return validity_mask, refl_in_bounding_box

            min_lon, max_lon, min_lat, max_lat = bounding_box

            # Create boolean masks for latitude and longitude
            lat_mask = (min_lat <= refl_LL_vec[:, 0]) & (refl_LL_vec[:, 0] <= max_lat)
            lon_mask = (min_lon <= refl_LL_vec[:, 1]) & (refl_LL_vec[:, 1] <= max_lon)

            # Combine masks
            validity_mask = finite & lat_mask & lon_mask

            # Create refl_in_bounding_box with NaN for invalid points
            refl_in_bounding_box = refl_LL_vec.copy()
            refl_in_bounding_box[~validity_mask] = np.nan
        else:
            # There are only NaN values in the refl_LL_vec
            validity_mask = np.zeros(len(reflection_obj.refl_LL_vec), dtype=bool)
            refl_in_bounding_box = np.full_like(reflection_obj.refl_LL_vec, np.nan)

        return validity_mask, refl_in_bounding_box

    @staticmethod
    def _get_reflection_clusters(reflection_obj, validity_mask):
        """
        =========================================================================================================
        Get clusters of valid reflection points based on the validity mask.

        INPUTS:
            reflection_obj: [Reflection]    Reflection object containing the reflection points
            validity_mask:  [np.ndarray]    Boolean mask indicating valid reflection points

        OUTPUT:
            clusters: [list]   List of arrays of indices corresponding to contiguous valid points
        =========================================================================================================
        """
        validity_mask_bool = np.asarray(validity_mask, dtype=bool)
        if validity_mask_bool.shape != (len(reflection_obj.refl_LL_vec),):
            raise ValueError("validity_mask length must match reflection_obj.refl_LL_vec")

        valid_indices = np.flatnonzero(validity_mask_bool)
        if valid_indices.size == 0:
            return []
        split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
        return list(np.split(valid_indices, split_points))


@dataclass
class CoverageStruct:
    position: list = field(default_factory=list)
    tx_satID: list = field(default_factory=list)
    rx_satID: list = field(default_factory=list)
    times: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    shape: list = field(default_factory=list)
    power: list = field(default_factory=list)
