import json
import os
import warnings

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS", UserWarning)
from bisect import bisect_left
from datetime import datetime, timedelta
from math import atan2, cos, degrees, radians, sin, sqrt
import pickle
import random

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, cKDTree  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402
import shapely.ops as ops  # noqa: E402
from tqdm import tqdm  # noqa: E402

import functions.functions as fn  # noqa: E402
from object_definitions.Constants_def import Constants  # noqa: E402


# from Coverage_def   import Coverage
class ShipTrajectory:
    """
    =========================================================================================================
    ShipTrajectory class:
        - Represent the trajectory of a ship and generate a shapefile for a certain time period.
        - Includes land avoidance for realistic maritime routing.

        ATTRIBUTES:
            ship_id:                [str]        Unique identifier for the ship
            start_time:             [datetime]   Start time of the trajectory
            end_time:               [datetime]   End time of the trajectory
            start_position:         [tuple]      Starting position (latitude, longitude)
            end_position:           [tuple]      Ending position (latitude, longitude)
            ship_speed:             [float]      Ship speed [m/s]
            ocean_polygons:         [object]     Loaded coastline data
            water_buffer:           [float]      Safety distance from shore [m]
            fast_method:            [bool]       Flag for simplified path generation
            mesh_connection_radius: [float]      Connection radius for grid points [m]
            interval_minutes:       [int]        Time interval [min] for trajectory generation
            grid_resolution:        [float]      Grid resolution [m] for the navigation mesh
            delta_m:                [float]      Bounding box margin [m] added to min/max lon/lat
            trajectory_points:      [list]       List of trajectory points
            timestamps:             [list]       List of timestamps for trajectory points
            waypoints:              [list]       List of waypoints for the ship's route
            route_distance:         [float]      Total distance of the route [m]
            shapefile_path:         [dict]       Dict of shapefile paths per trajectory type
    =========================================================================================================
    """

    def __init__(
        self,
        ship_id: str,
        ship_speed_knot: float,
        start_position: tuple,
        start_time: datetime,
        end_position: tuple,
        interval_minutes: int = 10,
        grid_resolution_km: float = 50,
        delta_boudingBox_km: float = 500,
        fast_path_gen_method: bool = True,
        water_buffer_km: float = 25.0,
        trajectory_type: list = None,
        save_graph: bool = False,
    ) -> None:
        self.ship_id = ship_id  # Unique identifier for the ship
        self.ship_speed = ship_speed_knot * 0.51444  # Ship speed in knots (will be converted to to [m/s])
        self.start_position = start_position  # Ship start positon (latitude, longitude), if the start position is on land, the nearest water point will be used
        self.start_time = fn.ensure_utc_aware(start_time)  # Start time of the ship journey
        self.end_position = end_position  # Ship end position  (latitude, longitude), if the end position is on land, the nearest water point will be used
        self.interval_minutes = interval_minutes  # Time interval in minutes for trajectory generation
        self.grid_resolution = (
            grid_resolution_km * 1e3
        )  # Grid resolution in meters for navigation mesh [this has a BIG IMPACT ON THE PERFORMANCE of the path generation]
        self.delta_m = (
            delta_boudingBox_km * 1e3
        )  # Additional margin in meters for the bounding box (default: 500 km | kept big to avoid being 'trapped' in a bay / corner / fjord)
        self.fast_method = (
            fast_path_gen_method  # Flag [boolean] to indicate if simplified path generation is used (faster computation / longer, more simple ship route)
        )
        self.mesh_connection_radius = self.grid_resolution * 2  # Connection radius for grid points in meters
        self.water_buffer = water_buffer_km * 1e3  # Safety distance from shore in meters (default: 22.2 km = 12 nm (Territorial Sea))

        self.ocean_polygons = None  # Loaded coastline data (geopandas object), will be loaded from shapefile
        self.end_time = None  # End time of the ship journey Will be set after route planning

        # Load coastline data if provided
        if Constants.earth_coastline_path and os.path.exists(Constants.earth_coastline_path):
            self.load_coastline_data()
        else:
            warnings.warn("No valid coastline data provided. Update path in Constants_def.py [earth_coastline_path]")

        if self.ocean_polygons is None:
            raise ValueError("No valid coastline data loaded. Cannot proceed with land avoidance.")

        # Initialize trajectory data
        self.trajectory_points = (
            []
        )  # List of trajectory points (latitude, longitude), generated with "self.generate_trajectory_equal_time()" or "self.generate_trajectory_equal_distance()"
        self.timestamps = (
            []
        )  # List of timestamps for trajectory points (datetime objects), generated with "self.generate_trajectory_equal_time()" or "self.generate_trajectory_equal_distance()"
        self.waypoints = []  # List of waypoints (latitude, longitude), generated with "self.plan_route()"
        self.route_distance = None  # Total route distance in meters, calculated with "self.plan_route()"
        self.shapefile_path = {}  # Dictionary to store shapefile paths for different trajectory types (points, lines, waypoints)

        self.initialize(trajectory_type, save_graph)

    def initialize(self, trajectory_type: list, save_graph: bool) -> None:
        """
        =========================================================================================================
        Initialize the trajectory data, loading from disk if available or planning a new route.

        INPUTS:
            trajectory_type: [list]   Trajectory shapefile types to generate ("points", "line", "waypoints")
            save_graph:      [bool]   If True, persist the navigation graph for inspection
        =========================================================================================================
        """
        if os.path.exists(os.path.join(Constants.autogen_ship_shapefiles, f"{self.ship_id}_trajectory.shp")) and os.path.exists(
            os.path.join(
                Constants.autogen_ship_shapefiles,
                f"{self.ship_id}_trajectory_metadata.json",
            )
        ):
            # Load existing shapefile
            self.load_class_from_file()
        else:
            self.plan_route(save_graph)  # Default connection radius for grid points
            self.generate_trajectory_equal_time()
            self.create_trajectory_files()
            if trajectory_type.__contains__("points"):
                self.create_shapefile(trajectory_type="points")
            if trajectory_type.__contains__("line"):
                self.create_shapefile(trajectory_type="line")
            if trajectory_type.__contains__("waypoints"):
                self.create_shapefile(trajectory_type="waypoints")

    def get_travel_duration_hours(self) -> float:
        """
        =========================================================================================================
        Calculate the travel duration in hours from the ship's start and end time.

        OUTPUT:
            travel_duration: [float]   Duration [hours]
        =========================================================================================================
        """
        if self.start_time and self.end_time:
            #            end_time_dt     = datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')
            #            start_time_dt   = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
            travel_duration = (self.end_time - self.start_time).total_seconds() / Constants.SEC_PER_HOUR
            return travel_duration
        else:
            raise ValueError("Start time or end time not set. Cannot calculate travel duration.")

    def load_class_from_file(self) -> None:
        """
        =========================================================================================================
        Load existing trajectory shapefile + metadata JSON for this ship from disk.
        =========================================================================================================
        """
        # Load the shapefile using geopandas
        try:
            self.shapefile_path["time_series"] = os.path.join(Constants.autogen_ship_shapefiles, f"{self.ship_id}_trajectory.shp")
            shapefile_time_series = gpd.read_file(self.shapefile_path["time_series"])
            trajectory_points = shapefile_time_series.geometry.values.tolist()
            self.trajectory_points = [(np.float64(point.y), np.float64(point.x)) for point in trajectory_points]
            self.timestamps = [
                fn.ensure_utc_aware(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")) for dt_str in shapefile_time_series["datetime_s"].values.tolist()
            ]
            geometry = shapefile_time_series["geometry"].values.tolist()
            self.end_position = geometry[-1].coords[0]  # Last point in the trajectory
            self.start_position = geometry[0].coords[0]  # First point in the trajectory

            # Load metadata from JSON file
            with open(
                os.path.join(
                    Constants.autogen_ship_shapefiles,
                    f"{self.ship_id}_trajectory_metadata.json",
                ),
            ) as f:
                metadata = json.load(f)
                self.ship_id = metadata["ship_id"]
                self.start_position = metadata["start_position"]
                self.end_position = metadata["end_position"]
                start_time_str = metadata["start_time"]
                self.start_time = fn.ensure_utc_aware(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S"))
                end_time_str = metadata["end_time"]
                self.end_time = fn.ensure_utc_aware(datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S"))
                self.ship_speed = metadata["ship_speed_m_s"]
                self.route_distance = metadata["route_distance_m"]
                self.fast_method = metadata["fast_method"]
                self.mesh_connection_radius = metadata["mesh_connection_radius"]
                self.water_buffer = metadata["water_buffer"]
                self.waypoints = metadata["waypoints"]
                self.grid_resolution = metadata["grid_resolution"]
                self.delta_m = metadata["delta_m"]
            print(f"Loaded trajectory data for ship {self.ship_id} from file")

        except Exception as e:
            print(f"Error loading shapefile: {e}")

    def load_coastline_data(self) -> bool:
        """
        =========================================================================================================
        Load coastline data from the shapefile referenced by `Constants.earth_coastline_path`.

        OUTPUT:
            success: [bool]   True if the data was loaded
        =========================================================================================================
        """
        try:
            ocean_data = gpd.read_file(Constants.earth_coastline_path)
            print(f"Loaded coastline data with {len(ocean_data)} features")

            # If there are multiple polygons, merge them into a single geometry
            self.ocean_polygons = ops.unary_union(ocean_data.geometry)
            print("Coastline data processed successfully")
            return True
        except Exception as e:
            print(f"Error loading coastline data: {e}")
            print("Proceeding without land avoidance")
            self.ocean_polygons = None
            return False

    def _calculate_haversine_distance(self, pos1: tuple, pos2: tuple) -> float:
        """
        =========================================================================================================
        Calculate the haversine distance between two points.

        INPUTS:
            pos1: [tuple]   First position (latitude, longitude)
            pos2: [tuple]   Second position (latitude, longitude)

        OUTPUT:
            distance: [float]   Distance [m]
        =========================================================================================================
        """

        lat1_rad, lon1_rad = radians(pos1[0]), radians(pos1[1])
        lat2_rad, lon2_rad = radians(pos2[0]), radians(pos2[1])

        # Haversine formula [https://en.wikipedia.org/wiki/Haversine_formula]
        delta_phi = lat2_rad - lat1_rad
        delta_lambda = lon2_rad - lon1_rad

        a = np.sin(delta_phi / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.asin(sqrt(a))

        distance = Constants.RADIUS_EARTH_EQUAT * c  # in meters

        return float(distance)

    def _calculate_haversine_vectorized(self, pos1_array: np.ndarray, pos2_array: np.ndarray) -> np.ndarray:
        """
        =========================================================================================================
        Vectorized Haversine distance calculation for multiple point pairs.

        INPUTS:
            pos1_array: [np.ndarray, shape (N, 2)]   First positions [lat, lon]
            pos2_array: [np.ndarray, shape (N, 2)]   Second positions [lat, lon]

        OUTPUT:
            distances: [np.ndarray, shape (N,)]   Distances [m]
        =========================================================================================================
        """
        # Convert to radians
        lat1_rad = np.radians(pos1_array[:, 0])
        lon1_rad = np.radians(pos1_array[:, 1])
        lat2_rad = np.radians(pos2_array[:, 0])
        lon2_rad = np.radians(pos2_array[:, 1])

        # Vectorized Haversine formula
        delta_phi = lat2_rad - lat1_rad
        delta_lambda = lon2_rad - lon1_rad

        a = np.sin(delta_phi / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        distances = Constants.RADIUS_EARTH_EQUAT * c
        return distances

    def _calculate_segment_distances(self):
        """
        =========================================================================================================
        Calculate all waypoint segment distances at once using vectorized Haversine.

        OUTPUT:
            distances: [np.ndarray | list]   Per-segment distances [m] (empty list if <=1 waypoint)
        =========================================================================================================
        """
        if len(self.waypoints) <= 1:
            return []

        waypoints_array = np.array(self.waypoints)
        pos1_array = waypoints_array[:-1]  # All except last
        pos2_array = waypoints_array[1:]  # All except first

        return self._calculate_haversine_vectorized(pos1_array, pos2_array)

    def calculate_bearing(self, pos1: tuple, pos2: tuple) -> float:
        """
        =========================================================================================================
        Calculate the bearing (heading) from pos1 to pos2.

        INPUTS:
            pos1: [tuple]   First position (latitude, longitude)
            pos2: [tuple]   Second position (latitude, longitude)

        OUTPUT:
            bearing: [float]   Bearing in degrees (0-360)
        =========================================================================================================
        """
        lat1_rad, lon1_rad = radians(pos1[0]), radians(pos1[1])
        lat2_rad, lon2_rad = radians(pos2[0]), radians(pos2[1])

        dlon_rad = lon2_rad - lon1_rad

        y = sin(dlon_rad) * cos(lat2_rad)
        x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon_rad)

        bearing = degrees(atan2(y, x))

        # Convert to 0-360 degrees
        bearing = (bearing + 360) % 360

        return bearing

    def intersects_land(self, point1: tuple, point2: tuple) -> bool:
        """
        =========================================================================================================
        Check if the line between two points intersects with land.

        INPUTS:
            point1: [tuple]   First point (latitude, longitude)
            point2: [tuple]   Second point (latitude, longitude)

        OUTPUT:
            intersects: [bool]   True if the line crosses land
        =========================================================================================================
        """
        if self.ocean_polygons is None:
            warnings.warn("No ocean data available")
            return False  # No land data available

        # Create a line between the two points
        # Note: LineString expects (lon, lat) order
        line = LineString([(point1[1], point1[0]), (point2[1], point2[0])])

        # With ocean data, we need to check if:
        # 1. Both endpoints are in ocean (safe)
        # 2. Line doesn't cross ocean boundaries (doesn't exit and re-enter ocean)

        point1_in_ocean = self.ocean_polygons.contains(Point(point1[1], point1[0]))
        point2_in_ocean = self.ocean_polygons.contains(Point(point2[1], point2[0]))

        if point1_in_ocean and point2_in_ocean:
            # Line should be fully contained within ocean (or at most touch boundary)
            return not self.ocean_polygons.contains(line)

        # If one or both points are on land, the line intersects land
        return True

    def is_point_in_ocean(self, point: tuple) -> bool:
        """
        =========================================================================================================
        Check if a point is in the ocean.

        INPUTS:
            point: [tuple]   Point to check (latitude, longitude)

        OUTPUT:
            in_ocean: [bool]   True if the point is contained in the ocean polygon
        =========================================================================================================
        """
        if self.ocean_polygons is None:
            warnings.warn("No ocean data available")
            return False  # No land data available

        # Create a Point (lon, lat order for shapely)
        point_geom = Point(point[1], point[0])

        # Check if the point is on land
        return self.ocean_polygons.contains(point_geom)

    def get_safe_water_point(self, point: tuple) -> tuple:
        """
        =========================================================================================================
        If a point is on land, find the nearest water point with `self.water_buffer` margin.

        INPUTS:
            point: [tuple]   Original point (latitude, longitude)

        OUTPUT:
            safe_point: [tuple]   Safe water point (latitude, longitude)
        =========================================================================================================
        """
        if self.ocean_polygons is None:
            warnings.warn("No ocean data available")
            return None

        # Create a Point (lon, lat order for shapely)
        point_geom = Point(point[1], point[0])

        # When using ocean data, point is on land if it's NOT contained in the ocean
        is_on_land = not self.ocean_polygons.contains(point_geom)

        if not is_on_land:
            # Already in water
            return point

        # Find nearest point on the boundary
        boundary = self.ocean_polygons.boundary
        nearest_boundary = ops.nearest_points(point_geom, boundary)[1]

        # Add a buffer to make sure we're in the water
        vector_x = nearest_boundary.x - point_geom.x
        vector_y = nearest_boundary.y - point_geom.y

        # Normalize and extend
        length = sqrt(vector_x**2 + vector_y**2)
        if length > 0:
            norm_x = vector_x / length
            norm_y = vector_y / length

            # Move "self.water_buffer" beyond the coastline
            safe_x = nearest_boundary.x + norm_x * self.water_buffer
            safe_y = nearest_boundary.y + norm_y * self.water_buffer

            return (safe_y, safe_x)  # Back to (lat, lon) order

        return (
            point[0] + self.water_buffer,
            point[1],
        )  # Fallback to original point (this should not happen)

    def plan_route(self, save_graph: bool = False) -> None:
        """
        =========================================================================================================
        Plan a route that avoids land. Uses a simplified path-finding approach over a navigation mesh.

        Updates `self.waypoints` and `self.route_distance` in place.

        INPUTS:
            save_graph: [bool]   If True, persist the navigation graph to disk for inspection
        =========================================================================================================
        """
        # If no land data, just return direct path
        if self.ocean_polygons is None:
            warnings.warn("No ocean data available")

        # Ensure start and end points are in water
        start_safe = self.get_safe_water_point(self.start_position)
        end_safe = self.get_safe_water_point(self.end_position)

        # Check if direct path is possible
        if not self.intersects_land(start_safe, end_safe):
            print("Direct path possible - no land obstacles")
            self.waypoints = [start_safe, end_safe]
            self.route_distance = self._calculate_haversine_distance(start_safe, end_safe)
            return

        print("Planning route to avoid land")

        # Define the bounding box with margin
        min_lat_noBuffer = min(start_safe[0], end_safe[0])
        max_lat_noBuffer = max(start_safe[0], end_safe[0])
        min_lon_noBuffer = min(start_safe[1], end_safe[1])
        max_lon_noBuffer = max(start_safe[1], end_safe[1])

        min_lat = min_lat_noBuffer - fn.delta_lat_from_distance(min_lat_noBuffer, self.delta_m)
        max_lat = max_lat_noBuffer + fn.delta_lat_from_distance(max_lat_noBuffer, self.delta_m)
        min_lon = min_lon_noBuffer - fn.delta_lon_from_distance(
            max_lat_noBuffer, self.delta_m
        )  # Use max latitude for delta_lon because it results in a bigger delta_lon
        max_lon = max_lon_noBuffer + fn.delta_lon_from_distance(
            max_lat_noBuffer, self.delta_m
        )  # Use max latitude for delta_lon because it results in a bigger delta_lon

        # Create initial navigation graph with adaptive mesh
        G = nx.Graph()
        G.add_node("start", pos=start_safe)
        G.add_node("end", pos=end_safe)

        # Create grid and add water points to graph
        valid_points = self._create_grid_points(min_lat, max_lat, min_lon, max_lon)

        # Connect points in the graph
        self._connect_grid_points(G, valid_points, start_safe, end_safe, save_graph)

        # Check if path is possible with the initial grid
        if not nx.has_path(G, "start", "end"):
            # If no path exists, try to bridge disconnected components
            G = self._bridge_components(G, valid_points)

        # DEBUG: PLOT GRAPH
        # fig, ax = plt.subplots(figsize=(10, 10))
        # gpd.GeoSeries(self.ocean_polygons).plot(ax=ax, color='lightblue', edgecolor='darkblue', alpha=0.5)
        # for point in valid_points:
        #     ax.plot(point[1], point[0], 'go')  # Green points for valid water points
        # for edge in G.edges(data=True):
        #     p1 = G.nodes[edge[0]]["pos"]
        #     p2 = G.nodes[edge[1]]["pos"]
        #     ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-', alpha=0.5)  # Red lines for edges
        # ax.plot(start_safe[1], start_safe[0], 'ro', markersize=10)  # Red point for start
        # ax.plot(end_safe[1], end_safe[0], 'ro', markersize=10)  # Red point for end
        # ax.set_title("Navigation Mesh with Valid Water Points")
        # ax.set_xlabel("Longitude")
        # ax.set_ylabel("Latitude")
        # plt.show(block=True)
        # fig.savefig(os.path.join(Constants.autogen_ship_shapefiles, f'{self.ship_id}_graph.png'), dpi=300, bbox_inches='tight')
        # # Save the figure as a pickle file for later use
        # with open(os.path.join(Constants.autogen_ship_shapefiles, f'{self.ship_id}_navGraphFig.pickle'), 'wb') as f:
        #     pickle.dump(fig, f)
        # # Save the graph as a pickle file for later use
        # with open(os.path.join(Constants.autogen_ship_shapefiles, f'{self.ship_id}_graph.pickle'), 'wb') as f:
        #     pickle.dump(G, f)

        # Find the shortest path
        try:
            path = nx.shortest_path(G, "start", "end", weight="weight")

            # Extract and simplify waypoints
            waypoints = [G.nodes[node]["pos"] for node in path]
            simplified_waypoints = self.simplify_path(waypoints)

            # Calculate total route distance
            total_distance = 0
            for i in range(len(simplified_waypoints) - 1):
                total_distance += self._calculate_haversine_distance(simplified_waypoints[i], simplified_waypoints[i + 1])

            self.waypoints = simplified_waypoints
            self.route_distance = total_distance

            # Update end_time based on actual route distance
            if self.route_distance:
                # Convert distance to kilometers and speed to km/h for time calculation
                estimated_travel_time_seconds = self.route_distance / self.ship_speed
                self.end_time = self.start_time + timedelta(seconds=estimated_travel_time_seconds)

            print(f"Route planned with {len(simplified_waypoints)} waypoints, total distance: {total_distance:.1f} km")
            return

        except nx.NetworkXNoPath:
            print("WARNING: No path found. Using direct route with potential land crossings.")
            self.waypoints = None
            return

    def _create_grid_points(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> list:
        """
        =========================================================================================================
        Create a grid of points within a bounding box and return only those in the ocean.

        INPUTS:
            min_lat: [float]   Bounding box minimum latitude [deg]
            max_lat: [float]   Bounding box maximum latitude [deg]
            min_lon: [float]   Bounding box minimum longitude [deg]
            max_lon: [float]   Bounding box maximum longitude [deg]

        OUTPUT:
            valid_points: [list[tuple]]   In-ocean grid points [(lat, lon), ...]
        =========================================================================================================
        """
        valid_points = []
        print(f"Creating navigation mesh with resolution of {self.grid_resolution * 1e-3} km")

        # Loop over latitude bands with adaptive longitude spacing
        current_lat = min_lat
        while current_lat <= max_lat:
            # Calculate the longitude increment based on current latitude
            lon_increment = fn.delta_lon_from_distance(current_lat, self.grid_resolution)

            # Create points along this latitude band
            current_lon = min_lon
            while current_lon <= max_lon:
                if self.is_point_in_ocean((current_lat, current_lon)):
                    valid_points.append((current_lat, current_lon))

                current_lon += lon_increment

            # Move to next latitude band
            lat_increment = fn.delta_lat_from_distance(current_lat, self.grid_resolution)
            current_lat += lat_increment

        print(f"Created navigation mesh with {len(valid_points)} valid water points")
        return valid_points

    def _bridge_components(self, G: nx.Graph, valid_points: list) -> nx.Graph:
        """
        =========================================================================================================
        Attempt to bridge disconnected components using Delaunay triangulation to identify
        potential passages.

        INPUTS:
            G:            [nx.Graph]     Navigation graph
            valid_points: [list[tuple]]  In-ocean grid points

        OUTPUT:
            G: [nx.Graph]   Graph with bridge edges added (in-place return)
        =========================================================================================================
        """
        # Find connected components
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} connected components")

        # Find which components contain start and end
        start_component = end_component = None
        for i, component in enumerate(components):
            if "start" in component:
                start_component = i
            if "end" in component:
                end_component = i

        # If start and end are already connected, nothing to do
        if start_component == end_component or start_component is None or end_component is None:
            return G

        print("Start and end are in different components. Attempting to bridge components...")

        # Use Delaunay triangulation to find potential passages
        try:
            # Extract points from each component
            start_comp_nodes = [n for n in components[start_component] if n != "start"]
            end_comp_nodes = [n for n in components[end_component] if n != "end"]

            # Limit to reasonable number for computation
            max_points = 200
            if len(start_comp_nodes) > max_points:
                start_comp_nodes = random.sample(start_comp_nodes, max_points)
            if len(end_comp_nodes) > max_points:
                end_comp_nodes = random.sample(end_comp_nodes, max_points)

            # Get positions
            start_comp_points = [G.nodes[n]["pos"] for n in start_comp_nodes]
            end_comp_points = [G.nodes[n]["pos"] for n in end_comp_nodes]

            # Combine points for triangulation
            all_points = start_comp_points + end_comp_points
            all_points_array = np.array(all_points)

            # Create Delaunay triangulation
            tri = Delaunay(all_points_array)

            # Extract edges that might connect components
            potential_bridges = []

            for simplex in tri.simplices:
                for i in range(3):
                    p1_idx = simplex[i]
                    p2_idx = simplex[(i + 1) % 3]

                    # Skip if both points are from same component
                    if (p1_idx < len(start_comp_points) and p2_idx < len(start_comp_points)) or (
                        p1_idx >= len(start_comp_points) and p2_idx >= len(start_comp_points)
                    ):
                        continue

                    p1 = all_points[p1_idx]
                    p2 = all_points[p2_idx]

                    # Calculate distance
                    dist = self._calculate_haversine_distance(p1, p2)

                    # Check if this connection crosses land
                    if not self.intersects_land(p1, p2):
                        potential_bridges.append((p1, p2, dist))

            # Sort bridges by distance (shorter first)
            potential_bridges.sort(key=lambda x: x[2])

            # Add the shortest 10 bridges
            bridges_added = 0
            for p1, p2, dist in potential_bridges[:10]:
                p1_id = f"{p1[0]:.4f}_{p1[1]:.4f}"
                p2_id = f"{p2[0]:.4f}_{p2[1]:.4f}"

                G.add_edge(p1_id, p2_id, weight=dist)
                bridges_added += 1

                # Check if we've connected the components
                if nx.has_path(G, "start", "end"):
                    break

            print(f"Added {bridges_added} bridges between components")

            # If still not connected, try adding midpoints between closest points
            if not nx.has_path(G, "start", "end") and potential_bridges:
                print("Adding midpoints in narrow passages...")

                for p1, p2, dist in potential_bridges[:5]:
                    # Create midpoint
                    mid_lat = (p1[0] + p2[0]) / 2
                    mid_lon = (p1[1] + p2[1]) / 2
                    midpoint = (mid_lat, mid_lon)

                    # Skip if midpoint is on land
                    if not self.is_point_in_ocean(midpoint):
                        continue

                    # Add midpoint to graph
                    mid_id = f"{mid_lat:.4f}_{mid_lon:.4f}"
                    G.add_node(mid_id, pos=midpoint)

                    # Connect to endpoints
                    p1_id = f"{p1[0]:.4f}_{p1[1]:.4f}"
                    p2_id = f"{p2[0]:.4f}_{p2[1]:.4f}"

                    G.add_edge(p1_id, mid_id, weight=dist / 2)
                    G.add_edge(mid_id, p2_id, weight=dist / 2)

                    # Add more connections from midpoint to nearby points
                    for point in valid_points:
                        point_id = f"{point[0]:.4f}_{point[1]:.4f}"
                        if point_id != mid_id:
                            mid_dist = self._calculate_haversine_distance(midpoint, point)
                            if mid_dist < self.grid_resolution and not self.intersects_land(midpoint, point):
                                G.add_edge(mid_id, point_id, weight=mid_dist)

        except Exception as e:
            print(f"Error in component bridging: {e}")
        return G

    def _connect_grid_points(self, G: nx.Graph, valid_points: list, start_pos: tuple, end_pos: tuple, save_graph: bool) -> None:
        """
        =========================================================================================================
        Connect points in the navigation graph with an improved approach focusing on creating a
        path between start and end positions. Modifies `G` in place.

        INPUTS:
            G:            [nx.Graph]     Navigation graph to add connections to
            valid_points: [list[tuple]]  Valid water points (lat, lon)
            start_pos:    [tuple]        Starting position (lat, lon)
            end_pos:      [tuple]        Ending position (lat, lon)
            save_graph:   [bool]         If True, persist the graph to disk
        =========================================================================================================
        """
        # Pre-compute points IDs and all nodes to add
        point_ids = {}
        nodes_to_add = []
        for point in valid_points:
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"
            point_ids[point] = point_id
            nodes_to_add.append((point_id, {"pos": point}))

        # Add all nodes in a single batch operation
        G.add_nodes_from(nodes_to_add)

        # Build KD-Tree for efficient nearest neighbor search
        points_array = np.array(valid_points)
        kdtree = cKDTree(points_array)

        # 1. Connect start and end to nearby points (more aggressively)
        start_connections = 0
        end_connections = 0

        # First pass - try with normal radius
        if self.fast_method == True:
            radius_in_degrees = self.mesh_connection_radius / 111000

            # Find points within the connection radius
            start_indices = kdtree.query_ball_point(start_pos, radius_in_degrees)
            end_indices = kdtree.query_ball_point([end_pos[0], end_pos[1]], radius_in_degrees)

            # Connect start to nearby points
            for idx in start_indices:
                point = valid_points[idx]
                point_id = point_ids[point]

                dist_to_start = self._calculate_haversine_distance(start_pos, point)  # Fixed: was using end_pos
                if dist_to_start < self.mesh_connection_radius and not self.intersects_land(start_pos, point):
                    G.add_edge("start", point_id, weight=dist_to_start)  # Fixed: was using "end"
                    start_connections += 1

            # Connect end to nearby points
            for idx in end_indices:
                point = valid_points[idx]
                point_id = point_ids[point]

                dist_to_end = self._calculate_haversine_distance(end_pos, point)
                if dist_to_end < self.mesh_connection_radius and not self.intersects_land(end_pos, point):
                    G.add_edge(point_id, "end", weight=dist_to_end)
                    end_connections += 1

            print(f"Connected start to {start_connections} points and end to {end_connections} points")

        # If start or end has no connections, try harder with closest points
        if start_connections == 0:
            distances, indices = kdtree.query([start_pos[0], start_pos[1]], k=min(20, len(valid_points)))
            for i in range(len(indices)):
                point = valid_points[indices[i]]
                if not self.intersects_land(start_pos, point):
                    dist = self._calculate_haversine_distance(start_pos, point)
                    point_id = point_ids[point]
                    G.add_edge("start", point_id, weight=dist)
                    print(f"Added edge from start to a point at distance {1e-3 * dist:.2f} km")
                    start_connections += 1
                    if start_connections >= 3:
                        break

        if end_connections == 0:
            distances, indices = kdtree.query([end_pos[0], end_pos[1]], k=min(20, len(valid_points)))
            for i in range(len(indices)):
                point = valid_points[indices[i]]
                if not self.intersects_land(end_pos, point):
                    dist = self._calculate_haversine_distance(end_pos, point)
                    point_id = point_ids[point]
                    G.add_edge(point_id, "end", weight=dist)
                    print(f"Added edge from end to point at distance {1e-3 * dist:.2f} km")
                    end_connections += 1
                    if end_connections >= 3:
                        break

        # 2. Connect grid points efficiently
        connected_to_start = set(G.neighbors("start"))
        connected_to_end = set(G.neighbors("end"))

        # Use batch edge addition for priority connections
        priority_edges = []

        # Connect from start connections to end connections (high priority)
        for start_node in tqdm(connected_to_start, desc="Finding route start-point to end-point"):
            start_position = G.nodes[start_node]["pos"]
            for end_node in connected_to_end:
                end_position = G.nodes[end_node]["pos"]

                dist = self._calculate_haversine_distance(start_position, end_position)
                if dist < self.mesh_connection_radius and not self.intersects_land(start_position, end_position):
                    priority_edges.append((start_node, end_node, {"weight": dist}))

        # Add all priority edges in batch
        G.add_edges_from(priority_edges)
        print(f"Added {len(priority_edges)} priority connections from start to end regions")

        # If path exists after priority connections, we can stop
        if nx.has_path(G, "start", "end"):
            print("Path found using priority connections")
            return

        # 3. Connect normal grid points with CORRECTED batch processing
        print("Adding regular grid connections...")
        radius_in_degrees = self.mesh_connection_radius / 111000

        # STEP 1: COLLECT all point pairs (don't process distances yet)
        all_pairs = []

        for i, p1 in enumerate(tqdm(valid_points, desc="Finding point pairs")):
            p1_id = point_ids[p1]
            nearby_indices = kdtree.query_ball_point([p1[0], p1[1]], radius_in_degrees)

            for idx in nearby_indices:
                p2 = valid_points[idx]
                p2_id = point_ids[p2]

                # Skip self-connections
                if p1_id == p2_id:
                    continue

                # Only process each pair once (prevent duplicates)
                if p1_id > p2_id and not G.has_edge(p1_id, p2_id):
                    all_pairs.append((i, idx, p1_id, p2_id))  # Store for batch processing

        # STEP 2: PROCESS all collected pairs in batches
        if all_pairs:
            print(f"Calculating {len(all_pairs)} distances in batches...")
            batch_size = 1000

            for batch_start in tqdm(range(0, len(all_pairs), batch_size), desc="Calculating distances"):
                batch_end = min(batch_start + batch_size, len(all_pairs))
                batch_pairs = all_pairs[batch_start:batch_end]

                # Extract point coordinates for this batch
                p1_coords = np.array([valid_points[pair[0]] for pair in batch_pairs])
                p2_coords = np.array([valid_points[pair[1]] for pair in batch_pairs])

                # Calculate all distances at once - VECTORIZED
                distances = self._calculate_haversine_vectorized(p1_coords, p2_coords)

                # Add edges for valid distances
                batch_edges = []
                for (i, idx, p1_id, p2_id), dist in zip(batch_pairs, distances, strict=False):
                    if dist < self.mesh_connection_radius and not self.intersects_land(valid_points[i], valid_points[idx]):
                        batch_edges.append((p1_id, p2_id, {"weight": dist}))

                # Add all edges in this batch at once
                G.add_edges_from(batch_edges)

        print("Added regular grid connections")

        # Save the graph
        if save_graph == True:
            print("Saving graph to file")
            if not os.path.exists(Constants.autogen_ship_shapefiles):
                os.makedirs(Constants.autogen_ship_shapefiles)
            with open(
                os.path.join(Constants.autogen_ship_shapefiles, f"{self.ship_id}_graph.pickle"),
                "wb",
            ) as f:
                pickle.dump(G, f)

    def simplify_path(self, waypoints: list) -> list:
        """
        =========================================================================================================
        Simplify a path by removing unnecessary points (Douglas-Peucker-style filtering).

        INPUTS:
            waypoints: [list]   List of waypoints to simplify

        OUTPUT:
            simplified: [list]   Simplified list of waypoints
        =========================================================================================================
        """
        if len(waypoints) <= 2:
            return waypoints

        # Always keep start and end points
        result = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            # Check if we can skip this point without crossing land
            if self.intersects_land(result[-1], waypoints[i + 1]):
                result.append(waypoints[i])

        result.append(waypoints[-1])
        return result

    def generate_trajectory_equal_dist(self, interval_meters: float = 100) -> list:
        """
        =========================================================================================================
        Generate points along the trajectory at regular distance intervals.
        Uses waypoints for land avoidance if available.

        INPUTS:
            interval_meters: [float]   Distance interval between points [m] (defaults to 100 if non-positive)

        OUTPUT:
            trajectory: [list]   List of (timestamp, (latitude, longitude)) tuples
        =========================================================================================================
        """
        # Ensure valid interval (minimum 100 meters)
        if interval_meters <= 0:
            warnings.warn("interval_meters must be positive. Using default value of 100 meters.")
            interval_meters = 100

        # Clear existing trajectory data
        self.trajectory_points = []
        self.timestamps = []

        # Plan route if we have coastline data and no waypoints yet
        if self.ocean_polygons is not None and not self.waypoints:
            self.plan_route()
        elif not self.waypoints:
            # Direct route if no land data
            self.waypoints = [self.start_position, self.end_position]
            self.route_distance = self._calculate_haversine_distance(self.start_position, self.end_position)

        # Calculate total route distance and time
        total_time = (self.end_time - self.start_time).total_seconds()

        if len(self.waypoints) > 1:
            # Calculate segment lengths
            segment_distances = self._calculate_segment_distances()

            total_distance = sum(segment_distances)

            # Segment start times based on proportional distances
            segment_times = [self.start_time]
            for i in range(len(segment_distances)):
                segment_fraction = segment_distances[i] / total_distance
                segment_time = segment_fraction * total_time
                next_time = segment_times[-1] + timedelta(seconds=segment_time)
                segment_times.append(next_time)

            # Generate trajectory for each segment based on distance intervals
            for i in tqdm(range(len(self.waypoints) - 1), desc="Generating trajectory points"):
                start_pos = self.waypoints[i]
                end_pos = self.waypoints[i + 1]
                start_t = segment_times[i]
                end_t = segment_times[i + 1]

                # Get segment distance
                segment_distance = segment_distances[i]

                # Calculate number of points needed for this segment
                num_points = max(1, int(segment_distance / interval_meters))

                # Generate evenly spaced points along the segment
                for j in range(num_points):
                    # Calculate progress along this segment (0 to 1)
                    progress = j / num_points

                    # Interpolate position
                    lat = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                    lon = start_pos[1] + progress * (end_pos[1] - start_pos[1])

                    # Interpolate time
                    point_time = start_t + timedelta(seconds=progress * (end_t - start_t).total_seconds())

                    # Add to trajectory
                    self.trajectory_points.append((lat, lon))
                    self.timestamps.append(point_time)

            # Make sure to include the end point
            if not self.trajectory_points or self.trajectory_points[-1] != self.waypoints[-1]:
                self.trajectory_points.append(self.waypoints[-1])
                self.timestamps.append(self.end_time)

        return list(zip(self.timestamps, self.trajectory_points, strict=False))

    def generate_trajectory_equal_time(self) -> None:
        """
        =========================================================================================================
        Generate points along the trajectory at regular time intervals (uses `self.interval_minutes`).
        Uses waypoints for land avoidance if available.

        Updates `self.trajectory_points` and `self.timestamps` in place.
        =========================================================================================================
        """
        # Clear existing trajectory data
        self.trajectory_points = []
        self.timestamps = []

        # Plan route if we have coastline data and no waypoints yet
        if self.ocean_polygons is not None and not self.waypoints:
            self.plan_route()
        elif not self.waypoints:
            # Direct route if no land data
            self.waypoints = [self.start_position, self.end_position]
            self.route_distance = self._calculate_haversine_distance(self.start_position, self.end_position)

        if self.interval_minutes == 0:
            warnings.warn("interval_minutes must be positive. Using default value of 10 minutes.")
            self.interval_minutes = 10

        # Normal case with interpolation
        time_step = timedelta(minutes=self.interval_minutes)
        total_time = (self.end_time - self.start_time).total_seconds()

        if len(self.waypoints) > 1:
            # Calculate segment lengths
            # Calculate segment lengths - VECTORIZED VERSION
            segment_distances = self._calculate_segment_distances()
            total_distance = sum(segment_distances)

            # Segment start times based on proportional distances
            segment_times = [self.start_time]
            for i in range(len(segment_distances)):
                # Calculate segment time based on distance fraction
                segment_fraction = segment_distances[i] / total_distance
                segment_time = segment_fraction * total_time
                next_time = segment_times[-1] + timedelta(seconds=segment_time)
                segment_times.append(next_time)

            # Generate trajectory for each segment
            for i in range(len(self.waypoints) - 1):
                # Get start and end positions and times for this segment
                start_pos = self.waypoints[i]
                end_pos = self.waypoints[i + 1]
                start_t = segment_times[i]
                end_t = segment_times[i + 1]
                current_time = start_t
                while current_time <= end_t:
                    # Calculate progress within this segment
                    progress = min(
                        1.0,
                        (current_time - start_t).total_seconds() / max(1, (end_t - start_t).total_seconds()),
                    )

                    # Interpolate position
                    lat = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                    lon = start_pos[1] + progress * (end_pos[1] - start_pos[1])

                    # Add to trajectory
                    self.trajectory_points.append((lat, lon))
                    self.timestamps.append(current_time)

                    # Increment time
                    current_time += time_step

            # Make sure to include the end point
            if not self.trajectory_points or self.trajectory_points[-1] != self.waypoints[-1]:
                self.trajectory_points.append(self.waypoints[-1])
                self.timestamps.append(self.end_time)

    def create_shapefile(self, output_filename: str = None, trajectory_type: str = "points", crs: str = "EPSG:4326") -> None:
        """
        =========================================================================================================
        Create a shapefile from the trajectory. Updates `self.shapefile_path[trajectory_type]` in place.

        INPUTS:
            output_filename: [str | None]   Output filename (defaults to a per-ship name)
            trajectory_type: [str]          'points', 'line', or 'waypoints'
            crs:             [str]          Coordinate reference system (default: EPSG:4326)

        NOTE: Used for visualization in GIS software; not consumed by the GNSS simulation itself.
        =========================================================================================================
        """
        # Generate trajectory if it hasn't been generated yet
        if not self.trajectory_points:
            self.generate_trajectory_equal_time()

        # Create GeoDataFrame
        if trajectory_type == "points":
            # Create points for each position
            geometry = [Point(lon, lat) for lat, lon in self.trajectory_points]
            data = {
                "ship_id": [self.ship_id] * len(self.trajectory_points),
                "timestamp": self.timestamps,
                "latitude": [pos[0] for pos in self.trajectory_points],
                "longitude": [pos[1] for pos in self.trajectory_points],
                "geometry": geometry,
            }
            gdf = gpd.GeoDataFrame(data, crs=crs)

        elif trajectory_type == "line":
            # Create a single LineString for the whole trajectory
            line_coords = [(lon, lat) for lat, lon in self.trajectory_points]
            line = LineString(line_coords)
            data = {
                "ship_id": [self.ship_id],
                "start_time": [self.start_time],
                "end_time": [self.end_time],
                "speed_knots": [self.ship_speed],
                "distance_nm": [self.route_distance if self.route_distance else None],
                "geometry": [line],
            }
            gdf = gpd.GeoDataFrame(data, crs=crs)

        elif trajectory_type == "waypoints":
            # Create points for each waypoint
            geometry = [Point(lon, lat) for lat, lon in self.waypoints]
            data = {
                "ship_id": [self.ship_id] * len(self.waypoints),
                "point_type": (["start"] + ["waypoint"] * (len(self.waypoints) - 2) + ["end"] if len(self.waypoints) > 1 else ["single"]),
                "latitude": [pos[0] for pos in self.waypoints],
                "longitude": [pos[1] for pos in self.waypoints],
                "geometry": geometry,
            }
            gdf = gpd.GeoDataFrame(data, crs=crs)

        else:
            raise ValueError(f"Invalid trajectory_type: {trajectory_type}. Must be 'points', 'line', or 'waypoints'.")

        # Save to shapefile
        if output_filename is None:
            output_filename = f"{self.ship_id}_{trajectory_type}.shp"
        output_path = os.path.join(Constants.autogen_ship_shapefiles, output_filename)
        if Constants.autogen_ship_shapefiles and not os.path.exists(Constants.autogen_ship_shapefiles):
            os.makedirs(Constants.autogen_ship_shapefiles)

        gdf.to_file(output_path)
        print(f"Created shapefile at {output_path}")
        self.shapefile_path[trajectory_type] = output_path

    def create_trajectory_files(self) -> None:
        """
        =========================================================================================================
        Create the time-series shapefile of ship positions and the matching metadata JSON.

        Stores the resulting paths in `self.shapefile_path["time_series"]` and
        `self.shapefile_path["time_series_metadata"]`.
        =========================================================================================================
        """
        # Generate trajectory with specified interval
        self.generate_trajectory_equal_time()

        # Create trajectory list from class attributes
        trajectory = list(zip(self.timestamps, self.trajectory_points, strict=False))

        # Create GeoDataFrame
        geometry = [Point(lon, lat) for lat, lon in self.trajectory_points]
        data = {
            "ship_id": [self.ship_id] * len(trajectory),
            "timestamp": self.timestamps,  # TODO remove this
            "datetime_str": [t.strftime("%Y-%m-%d %H:%M:%S") for t in self.timestamps],
            "latitude": [pos[0] for pos in self.trajectory_points],
            "longitude": [pos[1] for pos in self.trajectory_points],
            "geometry": geometry,
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        # Save to shapefile
        file_name = f"{self.ship_id}_trajectory.shp"
        if not file_name.endswith(".shp"):
            # Robustness: This should not be needed!
            file_name += ".shp"

        output_path = os.path.join(Constants.autogen_ship_shapefiles, file_name)
        if Constants.autogen_ship_shapefiles and not os.path.exists(Constants.autogen_ship_shapefiles):
            os.makedirs(Constants.autogen_ship_shapefiles)
        gdf.to_file(output_path)
        print(f"Created time series shapefile at {output_path}")
        self.shapefile_path["time_series"] = output_path

        ##########################################################################
        # Save additional metadata to a separate JSON file
        metadata_file_name = os.path.splitext(file_name)[0] + "_metadata.json"
        metadata_output_path = os.path.join(Constants.autogen_ship_shapefiles, metadata_file_name)

        # Prepare metadata dictionary
        metadata = {
            "ship_id": self.ship_id,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ship_speed_m_s": self.ship_speed,
            "route_distance_m": self.route_distance,
            "time_interval_minutes": self.interval_minutes,
            "fast_method": self.fast_method,
            "mesh_connection_radius": self.mesh_connection_radius,
            "water_buffer": self.water_buffer,
            "waypoints": self.waypoints,
            "grid_resolution": self.grid_resolution,
            "delta_m": self.delta_m,
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add waypoints if available
        if self.waypoints:
            metadata["waypoints"] = [(round(wp[0], 6), round(wp[1], 6)) for wp in self.waypoints]
            metadata["waypoint_count"] = len(self.waypoints)

        # Save metadata to JSON
        with open(metadata_output_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Created metadata JSON at {metadata_output_path}")
        self.shapefile_path["time_series_metadata"] = metadata_output_path

    def plot_route(self, show_trajectory: bool = True, show_waypoints: bool = True, title: str = None, show: bool = False) -> None:
        """
        =========================================================================================================
        Plot the ship's route on a map and save it to OUTPUT/.

        INPUTS:
            show_trajectory: [bool]   Show the full trajectory (default: True)
            show_waypoints:  [bool]   Show the waypoints (default: True)
            title:           [str]    Plot title (defaults to "Ship Route: {ship_id}")
            show:            [bool]   Display the figure interactively (default: False)
        =========================================================================================================
        """
        # Generate trajectory if needed
        if not self.trajectory_points and show_trajectory:
            self.generate_trajectory_equal_time()

        # Ensure waypoints if needed
        if not self.waypoints and show_waypoints:
            if self.ocean_polygons is not None:
                self.plan_route()
            else:
                self.waypoints = [self.start_position, self.end_position]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot land if available
        if self.ocean_polygons is not None:
            gpd.GeoSeries(self.ocean_polygons).plot(ax=ax, color="lightgreen", edgecolor="darkblue", alpha=0.5)

        # Plot trajectory
        if show_trajectory and self.trajectory_points:
            traj_x = [point[1] for point in self.trajectory_points]
            traj_y = [point[0] for point in self.trajectory_points]
            ax.plot(traj_x, traj_y, "b-", linewidth=1.5, alpha=0.7, label="Ship Trajectory")

        # Plot waypoints
        if show_waypoints and self.waypoints:
            way_x = [point[1] for point in self.waypoints]
            way_y = [point[0] for point in self.waypoints]
            ax.scatter(way_x, way_y, c="red", s=50, zorder=5, label="Waypoints")

        # Plot start and end points
        start_x, start_y = self.start_position[1], self.start_position[0]
        end_x, end_y = self.end_position[1], self.end_position[0]

        ax.scatter(start_x, start_y, c="green", s=100, marker="^", zorder=10, label="Start")
        ax.scatter(end_x, end_y, c="purple", s=100, marker="x", zorder=10, label="End")

        # Add labels and legend
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Ship Route: {self.ship_id}")
        ax.legend()

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if title == None:
            title_str = f"Ship Route: {self.ship_id}"
            file_name = f"ship_route_{self.ship_id}.png"
        else:
            title_str = title
            file_name = f"ship_route_{title}.png"
        plt.title(title_str)
        save_path = os.path.join(Constants.output_folder_path, f"{file_name}")
        if save_path:
            plt.savefig(save_path, dpi=600)
            print(f"Route plot saved to {save_path}")

        # Show plot
        if show == True:
            plt.show()

    def get_position_at_time(self, timestamp: datetime) -> tuple:
        """
        =========================================================================================================
        Get the ship's position at a specific timestamp.

        INPUTS:
            timestamp: [datetime]   The timestamp to look up

        OUTPUT:
            position: [tuple]   (latitude, longitude) at the given timestamp, or None on error
        =========================================================================================================
        """
        # Make sure timestamps are not none
        if not self.timestamps:
            print("Error: No timestamps available. Please generate the trajectory first.")
            return None
        # Make sure timestamps are sorted
        if not all(self.timestamps[i] <= self.timestamps[i + 1] for i in range(len(self.timestamps) - 1)):
            print("Error: Timestamps are not sorted. Please generate the trajectory first.")
            return None
        if not isinstance(timestamp, datetime):
            print("Error: Timestamp must be a datetime object.")
            return None
        # Get the closest timestamp
        idx = bisect_left(self.timestamps, timestamp)
        if idx == 0:
            return self.trajectory_points[0]
        elif idx == len(self.timestamps):
            return self.trajectory_points[-1]
        else:
            # Check which timestamp is closer
            if abs(self.timestamps[idx] - timestamp) < abs(self.timestamps[idx - 1] - timestamp):
                return self.trajectory_points[idx]
            else:
                return self.trajectory_points[idx - 1]

        # Find the closest point in time
        closest_point = min(self.timestamps, key=lambda t: abs(t - timestamp))
        index = self.timestamps.index(closest_point)
        return self.trajectory_points[index]


# Example usage:
if __name__ == "__main__":
    helgoland = (54.199416, 7.892937)
    Murmansk = (69.345572, 33.646450)
    oslo = (59.9139, 10.7522)
    stavanger = (58.9700, 5.7331)

    # Create ship trajectory
    ship = ShipTrajectory(
        ship_id="SHIP001",
        ship_speed_knot=15.0,  # 15 knots
        start_position=helgoland,
        start_time=datetime.now(),
        end_position=Murmansk,
        fast_path_gen_method=True,
    )

    # Plan and plot route
    ship.plot_route()
