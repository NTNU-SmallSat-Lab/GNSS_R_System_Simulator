import os
import warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS', UserWarning)
import numpy             as np
import pandas            as pd
import geopandas         as gpd
import matplotlib.pyplot as plt
import networkx          as nx
import shapely.ops       as ops
import random

from tqdm              import tqdm
from datetime          import datetime, timedelta
from math              import radians, cos, sin, asin, sqrt, atan2, degrees
from shapely.geometry  import Point, LineString, Polygon
from scipy.spatial     import Delaunay

from Constants_def import Constants

class ShipTrajectory:
    """
    ShipTrajectory class to represent the trajectory of a ship
    and generate a shapefile for a certain time period.
    Includes land avoidance for realistic maritime routing.

    Attributes:
        ship_id (str): Unique identifier for the ship.
        start_time (datetime): Start time of the trajectory.
        end_time (datetime): End time of the trajectory.
        start_position (tuple): Starting position of the ship (latitude, longitude).
        end_position (tuple): Ending position of the ship (latitude, longitude).
        ship_speed (float): Speed of the ship in knots.
        coastline_path (str): Path to the coastline shapefile.
    """

    def __init__(self, ship_id, ship_speed, start_position, start_time, end_position, end_time=None, coastline_path=None):
        self.ship_id        = ship_id
        self.start_time     = start_time
        self.start_position = start_position  # (latitude, longitude)
        self.end_position   = end_position    # (latitude, longitude)
        self.ship_speed     = ship_speed      # in knots
        self.coastline_path = coastline_path
        self.ocean_polygons = None            # Loaded with "self.load_coastline_data()"
        self.water_buffer   = 0.05            # degrees (~5km safety distance from shore)

        # Load coastline data if provided
        if coastline_path and os.path.exists(coastline_path):
            self.load_coastline_data(coastline_path)

        if self.ocean_polygons is not None:
            pass
            
#        # Calculate end_time if not provided
#        if end_time is None:
#            # We'll adjust this later after path planning
#            direct_distance       = self.calculate_shortest_distance(start_position, end_position)
#            estimated_travel_time = direct_distance / self.ship_speed
#            # Add 50% time buffer for non-direct routes
#            self.end_time = start_time + timedelta(hours=estimated_travel_time * 1.5)
#        else:
#            self.end_time = end_time

        # Initialize trajectory data
        self.trajectory_points = []
        self.timestamps        = []
        self.waypoints         = []
        self.route_distance    = None

        self.initialize()

    def initialize(self):
        """
        Initialize the trajectory data.
        """
        # TODO check if the shapefile already exists and if so, load it instead of creating a new one
        self.plan_route()

    def load_coastline_data(self, coastline_path):
        """
        Load coastline data from shapefile.

        INPUT:
            coastline_path (str): Path to the coastline shapefile
        """
        try:
            ocean_data = gpd.read_file(coastline_path)
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

    def calculate_shortest_distance(self, pos1, pos2):
        """
        Calculate the great-circle distance between two points in nautical miles
        using the Haversine formula.

        INPUT:
            pos1 (tuple): First position (latitude, longitude)
            pos2 (tuple): Second position (latitude, longitude)

        OUTPUT:
            float: Distance in nautical miles
        """
        lat1, lon1 = radians(pos1[0]), radians(pos1[1])
        lat2, lon2 = radians(pos2[0]), radians(pos2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Earth radius in km
        r = Constants.Radius_earth * 1e-3

        return c * r
    
    def calculate_bearing(self, pos1, pos2):
        """
        Calculate the bearing (heading) from pos1 to pos2.
        
        INPUT:
            pos1 (tuple): First position (latitude, longitude)
            pos2 (tuple): Second position (latitude, longitude)
            
        OUTPUT:
            float: Bearing in degrees (0-360)
        """
        lat1, lon1 = radians(pos1[0]), radians(pos1[1])
        lat2, lon2 = radians(pos2[0]), radians(pos2[1])
        
        dlon = lon2 - lon1
        
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = degrees(atan2(y, x))
        
        # Convert to 0-360 degrees
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def get_position_at_distance(self, start_pos, bearing, distance):
        """
        Calculate the position after moving a certain distance along a bearing.
        
        INPUT:
            start_pos (tuple): Starting position (latitude, longitude)
            bearing (float): Bearing in degrees
            distance (float): Distance in nautical miles
            
        OUTPUT:
            tuple: New position (latitude, longitude)
        """
        lat1, lon1 = radians(start_pos[0]), radians(start_pos[1])
        bearing_rad = radians(bearing)
        
        # Earth radius in nautical miles
        r = 3440.07
        
        # Angular distance in radians
        d = distance / r
        
        lat2 = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing_rad))
        lon2 = lon1 + atan2(sin(bearing_rad) * sin(d) * cos(lat1),
                            cos(d) - sin(lat1) * sin(lat2))
        
        # Convert back to degrees
        lat2, lon2 = degrees(lat2), degrees(lon2)
        
        return (lat2, lon2)
    
    def intersects_land(self, point1, point2):
        """
        Check if the line between two points intersects with land.
        
        INPUT:
            point1 (tuple): First point (latitude, longitude)
            point2 (tuple): Second point (latitude, longitude)
            
        OUTPUT:
            bool: True if the line intersects land, False otherwise
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
    
    def is_point_in_ocean(self, point):
        """
        Check if a point is on land.
        
        INPUT:
            point (tuple): Point to check (latitude, longitude)
            
        OUTPUT:
            bool: True if the point is on land, False otherwise
        """
        if self.ocean_polygons is None:
            warnings.warn("No ocean data available")
            return False  # No land data available
        
        # Create a Point (lon, lat order for shapely)
        point_geom = Point(point[1], point[0])
        
        # Check if the point is on land
        return self.ocean_polygons.contains(point_geom)
    
    def get_safe_water_point(self, point):
        """
        If a point is on land (not in the ocean), find the nearest water point.
        
        INPUT:
            point (tuple): Original point (latitude, longitude)
            
        OUTPUT:
            tuple: Safe water point (latitude, longitude)
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
        boundary         = self.ocean_polygons.boundary
        nearest_boundary = ops.nearest_points(point_geom, boundary)[1]
        
        # Add a buffer to make sure we're in the water
        vector_x = nearest_boundary.x - point_geom.x
        vector_y = nearest_boundary.y - point_geom.y
        
        # Normalize and extend
        length = sqrt(vector_x**2 + vector_y**2)
        if length > 0:
            norm_x = vector_x / length
            norm_y = vector_y / length
            
            # Move 5km (~0.05 degrees) beyond the coastline
            safe_x = nearest_boundary.x + norm_x * self.water_buffer
            safe_y = nearest_boundary.y + norm_y * self.water_buffer
            
            return (safe_y, safe_x)  # Back to (lat, lon) order
        
        return (point[0] + self.water_buffer, point[1])  # Fallback to original point (this should not happen)
    
    def plan_route(self, resolution=2.0, delta_deg = 8.0, connection_radius_factor=2.5, start_end_radius_factor=1000.0):
        """
        Plan a route that avoids land.
        Uses a simplified path finding approach with a mesh of navigable points.
        
        INPUT:
            resolution (float): Grid resolution in degrees for navigation mesh
            delta_deg (float):  Additional margin in degrees for the bounding box
            connection_radius_factor (float): Factor to determine connection radius for grid points
            start_end_radius_factor (float): Factor to determine connection radius for start/end points
            
        OUTPUT:
            list: List of waypoints (latitude, longitude)
        """
            # If no land data, just return direct path
        if self.ocean_polygons is None:
            return [self.start_position, self.end_position]
        
        # Ensure start and end points are in water
        start_safe = self.get_safe_water_point(self.start_position)
        end_safe   = self.get_safe_water_point(self.end_position)
    
        # Check if direct path is possible
        if not self.intersects_land(start_safe, end_safe):
            print("Direct path possible - no land obstacles")
            self.waypoints      = [start_safe, end_safe]
            self.route_distance = self.calculate_shortest_distance(start_safe, end_safe)
            return self.waypoints
    
        print("Planning route to avoid land")
    
        # Define the bounding box with margin
        min_lat = min(start_safe[0], end_safe[0]) - delta_deg
        max_lat = max(start_safe[0], end_safe[0]) + delta_deg
        min_lon = min(start_safe[1], end_safe[1]) - delta_deg
        max_lon = max(start_safe[1], end_safe[1]) + delta_deg

        # Create initial navigation graph with adaptive mesh
        G = nx.Graph()
        G.add_node("start", pos=start_safe)
        G.add_node("end",   pos=end_safe)

        # Create grid and add water points to graph
        valid_points = self._create_grid_points(min_lat, max_lat, min_lon, max_lon, resolution)

        # Connect points in the graph
#        self._connect_grid_points(G, valid_points, start_safe, end_safe, 
#                                  resolution * connection_radius_factor, 
#                                  resolution * start_end_radius_factor)
        self._connect_grid_points_improved(G, valid_points, start_safe, end_safe,
                                           resolution * connection_radius_factor, 
                                           resolution * start_end_radius_factor)

        # Check if path is possible with the initial grid
        if not nx.has_path(G, "start", "end"):
            # If no path exists, try to bridge disconnected components
            G = self._bridge_components(G, valid_points, resolution)

        #DEBUG: PLOT GRAPH
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
        # plt.show()

        # Find the shortest path
        try:
            path = nx.shortest_path(G, "start", "end", weight="weight")

            # Extract and simplify waypoints
            waypoints            = [G.nodes[node]["pos"] for node in path]
            simplified_waypoints = self.simplify_path(waypoints)

            # Calculate total route distance
            total_distance = 0
            for i in range(len(simplified_waypoints) - 1):
                total_distance += self.calculate_shortest_distance(simplified_waypoints[i], simplified_waypoints[i+1])
            
            self.waypoints      = simplified_waypoints
            self.route_distance = total_distance

            # Update end_time based on actual route distance
            if self.route_distance:
                # Convert distance to kilometers and speed to km/h for time calculation
                estimated_travel_time = self.route_distance / (self.ship_speed * 1.852)  # 1.852 km/h per knot
                self.end_time         = self.start_time + timedelta(hours=estimated_travel_time)

            print(f"Route planned with {len(simplified_waypoints)} waypoints, total distance: {total_distance:.1f} km")
            return simplified_waypoints

        except nx.NetworkXNoPath:
            print("No path found. Using direct route with potential land crossings.")
            self.waypoints      = [start_safe, end_safe]
            self.route_distance = self.calculate_shortest_distance(start_safe, end_safe)
            return self.waypoints

    def _create_grid_points(self, min_lat, max_lat, min_lon, max_lon, resolution):
        """
        Create a grid of points and return those in water.
        """
        valid_points = []

        # Regular grid
        lat_points = np.arange(min_lat, max_lat, resolution)
        lon_points = np.arange(min_lon, max_lon, resolution)

        # Add points that are in water
        for lat in tqdm(lat_points, desc="Creating grid points"):
            for lon in lon_points:
                if self.is_point_in_ocean((lat, lon)):
                    valid_points.append((lat, lon))

        print(f"Created navigation mesh with {len(valid_points)} valid water points")
        return valid_points

    def _connect_grid_points(self, G, valid_points, start_pos, end_pos, connection_radius, start_end_radius):
        """
        Connect points in the navigation graph.
        """
        # Add all valid points to the graph
        for point in valid_points:
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"
            G.add_node(point_id, pos=point)

        # Connect start and end to nearby points
        for point in tqdm(valid_points, desc="Connecting to start/end"):
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"

            # Connect to start if possible
            dist_to_start = self.calculate_shortest_distance(start_pos, point)
            if dist_to_start < start_end_radius and not self.intersects_land(start_pos, point):
                G.add_edge("start", point_id, weight=dist_to_start)

            # Connect to end if possible
            dist_to_end = self.calculate_shortest_distance(end_pos, point)
            if dist_to_end < start_end_radius and not self.intersects_land(end_pos, point):
                G.add_edge(point_id, "end", weight=dist_to_end)

        # Connect grid points to each other
        edge_count = 0
        for i, p1 in enumerate(tqdm(valid_points, desc="Creating edges")):
            p1_id = f"{p1[0]:.4f}_{p1[1]:.4f}"

            # Only check nearby points for efficiency
            for j in range(i + 1, len(valid_points)):
                p2 = valid_points[j]

                # Quick distance check to avoid unnecessary calculations
                if abs(p1[0] - p2[0]) > connection_radius/60 or abs(p1[1] - p2[1]) > connection_radius/60:
                    continue

                p2_id = f"{p2[0]:.4f}_{p2[1]:.4f}"

                # Calculate actual distance
                dist = self.calculate_shortest_distance(p1, p2)

                # Connect if close enough and no land intersection
                if dist < connection_radius and not self.intersects_land(p1, p2):
                    G.add_edge(p1_id, p2_id, weight=dist)
                    edge_count += 1

        print(f"Created {edge_count} edges in the navigation mesh")

    def _connect_grid_points2(self, G, valid_points, start_pos, end_pos, connection_radius, start_end_radius):
        """
        Connect points in the navigation graph using a different approach.
        """
        # Add Edges from Start to all valid points
        for point in tqdm(valid_points, desc="Connecting to start/end"):
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"

            # Connect to start if possible
            dist_to_start = self.calculate_shortest_distance(start_pos, point)
            if dist_to_start < start_end_radius and not self.intersects_land(start_pos, point):
                G.add_edge("start", point_id, weight=dist_to_start)

            # Connect to end if possible
            dist_to_end = self.calculate_shortest_distance(end_pos, point)
            if dist_to_end < start_end_radius and not self.intersects_land(end_pos, point):
                G.add_edge(point_id, "end", weight=dist_to_end)

        # Add Edges from valid points to each other if possible
        edge_count = 0
        for i, p1 in enumerate(tqdm(valid_points, desc="Creating edges")):
            p1_id = f"{p1[0]:.4f}_{p1[1]:.4f}"
            # Only check nearby points for efficiency
            for j in range(i + 1, len(valid_points)):
                p2 = valid_points[j]

                # Quick distance check to avoid unnecessary calculations
                if abs(p1[0] - p2[0]) > connection_radius/60 or abs(p1[1] - p2[1]) > connection_radius/60:
                    continue

                p2_id = f"{p2[0]:.4f}_{p2[1]:.4f}"

                # Calculate actual distance
                dist = self.calculate_shortest_distance(p1, p2)

                # Connect if close enough and no land intersection
                if dist < connection_radius and not self.intersects_land(p1, p2):
                    G.add_edge(p1_id, p2_id, weight=dist)
                    edge_count += 1
        

    def _bridge_components(self, G, valid_points, resolution):
        """
        Attempt to bridge disconnected components using Delaunay triangulation
        to identify potential passages.
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
                    p2_idx = simplex[(i+1)%3]
                    
                    # Skip if both points are from same component
                    if (p1_idx < len(start_comp_points) and p2_idx < len(start_comp_points)) or \
                       (p1_idx >= len(start_comp_points) and p2_idx >= len(start_comp_points)):
                        continue
                    
                    p1 = all_points[p1_idx]
                    p2 = all_points[p2_idx]
                    
                    # Calculate distance
                    dist = self.calculate_shortest_distance(p1, p2)
                    
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
                    
                    G.add_edge(p1_id, mid_id, weight=dist/2)
                    G.add_edge(mid_id, p2_id, weight=dist/2)
                    
                    # Add more connections from midpoint to nearby points
                    for point in valid_points:
                        point_id = f"{point[0]:.4f}_{point[1]:.4f}"
                        if point_id != mid_id:
                            mid_dist = self.calculate_shortest_distance(midpoint, point)
                            if mid_dist < resolution * 3 and not self.intersects_land(midpoint, point):
                                G.add_edge(mid_id, point_id, weight=mid_dist)
        
        except Exception as e:
            print(f"Error in component bridging: {e}")
        return G

    def _connect_grid_points_improved(self, G, valid_points, start_pos, end_pos, connection_radius, start_end_radius):
        """
        Connect points in the navigation graph with an improved approach focusing on
        creating a path between start and end positions.

        Parameters:
        -----------
        G : networkx.Graph
            The graph to add connections to
        valid_points : list
            List of valid water points (lat, lon)
        start_pos : tuple
            Starting position (lat, lon)
        end_pos : tuple
            Ending position (lat, lon)
        connection_radius : float
            Maximum distance to connect regular grid points
        start_end_radius : float
            Maximum distance to connect start/end to grid points
        """
        # Add all valid points to the graph
        for point in valid_points:
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"
            G.add_node(point_id, pos=point)

        # 1. Connect start and end to nearby points (more aggressively)
        start_connections = 0
        end_connections   = 0

        # First pass - try with normal radius
        for point in tqdm(valid_points, desc="Connecting to start/end"):
            point_id = f"{point[0]:.4f}_{point[1]:.4f}"

            # Connect to start if possible
            dist_to_start = self.calculate_shortest_distance(start_pos, point)
            if dist_to_start < start_end_radius and not self.intersects_land(start_pos, point):
                G.add_edge("start", point_id, weight=dist_to_start)
                start_connections += 1
            
            # Connect to end if possible
            dist_to_end = self.calculate_shortest_distance(end_pos, point)
            if dist_to_end < start_end_radius and not self.intersects_land(end_pos, point):
                G.add_edge(point_id, "end", weight=dist_to_end)
                end_connections += 1
        
        print(f"Connected start to {start_connections} points and end to {end_connections} points")
        
        # If start or end has no connections, try harder with closest points
        if start_connections == 0:
            closest_to_start = sorted(valid_points, key=lambda p: self.calculate_shortest_distance(start_pos, p))
            for point in closest_to_start[:10]:  # Try the 10 closest points
                if not self.intersects_land(start_pos, point):
                    dist = self.calculate_shortest_distance(start_pos, point)
                    point_id = f"{point[0]:.4f}_{point[1]:.4f}"
                    G.add_edge("start", point_id, weight=dist)
                    print(f"Forced connection from start to point at distance {dist:.2f} km")
                    start_connections += 1
                    if start_connections >= 3:  # Get at least 3 connections
                        break
        
        if end_connections == 0:
            closest_to_end = sorted(valid_points, key=lambda p: self.calculate_shortest_distance(end_pos, p))
            for point in closest_to_end[:10]:  # Try the 10 closest points
                if not self.intersects_land(end_pos, point):
                    dist = self.calculate_shortest_distance(end_pos, point)
                    point_id = f"{point[0]:.4f}_{point[1]:.4f}"
                    G.add_edge(point_id, "end", weight=dist)
                    print(f"Forced connection from end to point at distance {dist:.2f} km")
                    end_connections += 1
                    if end_connections >= 3:  # Get at least 3 connections
                        break
        
        # 2. Connect grid points efficiently
        # First identify points connected to start and end for priority
        connected_to_start = set()
        connected_to_end = set()
        
        for node in G.neighbors("start"):
            connected_to_start.add(node)
        
        for node in G.neighbors("end"):
            connected_to_end.add(node)
        
        # Connect these high-priority points first (from start to end)
        edge_count = 0
        
        # Connect from start connections to end connections (high priority)
        for start_node in connected_to_start:
            start_pos = G.nodes[start_node]["pos"]
            for end_node in connected_to_end:
                end_pos = G.nodes[end_node]["pos"]
                
                dist = self.calculate_shortest_distance(start_pos, end_pos)
                if dist < connection_radius * 2 and not self.intersects_land(start_pos, end_pos):
                    G.add_edge(start_node, end_node, weight=dist)
                    edge_count += 1
        
        print(f"Added {edge_count} priority connections from start to end regions")
        
        # If path exists after priority connections, we can stop
        if nx.has_path(G, "start", "end"):
            print("Path found using priority connections")
            return
        
        # 3. Connect normal grid points
        print("Adding regular grid connections...")
        normal_edges = 0
        
        # Use a nearby point approach for efficiency
        for i, p1 in enumerate(tqdm(valid_points, desc="Creating edges")):
            p1_id = f"{p1[0]:.4f}_{p1[1]:.4f}"
            
            # Only check nearby points to improve performance
            nearby_points = []
            for p2 in valid_points:
                # Quick approximate distance check
                if abs(p1[0] - p2[0]) > connection_radius/60 or abs(p1[1] - p2[1]) > connection_radius/60:
                    continue
                nearby_points.append(p2)
            
            for p2 in nearby_points:
                p2_id = f"{p2[0]:.4f}_{p2[1]:.4f}"
                
                # Skip self-connections
                if p1_id == p2_id:
                    continue
                
                # Only process each pair once
                if G.has_edge(p1_id, p2_id):
                    continue
                    
                # Calculate actual distance
                dist = self.calculate_shortest_distance(p1, p2)
                
                # Connect if close enough and no land intersection
                if dist < connection_radius and not self.intersects_land(p1, p2):
                    G.add_edge(p1_id, p2_id, weight=dist)
                    normal_edges += 1
        
        print(f"Added {normal_edges} regular grid connections")




    def simplify_path(self, waypoints, tolerance=0.1):
        """
        Simplify a path by removing unnecessary points.

        INPUT:
            waypoints (list): List of waypoints to simplify
            tolerance (float): Tolerance for simplification

        OUTPUT:
            list: Simplified list of waypoints
        """
        if len(waypoints) <= 2:
            return waypoints

        # Always keep start and end points
        result = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            # Check if we can skip this point without crossing land
            if self.intersects_land(result[-1], waypoints[i+1]):
                result.append(waypoints[i])

        result.append(waypoints[-1])
        return result

    def generate_trajectory(self, interval_minutes=5):
        """
        Generate points along the trajectory at regular time intervals.
        Uses waypoints for land avoidance if available.

        INPUT:
            interval_minutes (int): Time interval between points in minutes

        OUTPUT:
            list: List of tuples (timestamp, (latitude, longitude))
        """
        # Clear existing trajectory data
        self.trajectory_points = []
        self.timestamps        = []

        # Plan route if we have coastline data and no waypoints yet
        if self.ocean_polygons is not None and not self.waypoints:
            self.plan_route()
        elif not self.waypoints:
            # Direct route if no land data
            self.waypoints = [self.start_position, self.end_position]
            self.route_distance = self.calculate_shortest_distance(self.start_position, self.end_position)

        # Calculate time steps
        time_step = timedelta(minutes=interval_minutes)

        # For each segment between waypoints
        total_time = (self.end_time - self.start_time).total_seconds()

        if len(self.waypoints) > 1:
            # Calculate segment lengths
            segment_distances = []
            for i in range(len(self.waypoints) - 1):
                segment_distances.append(self.calculate_shortest_distance(self.waypoints[i], self.waypoints[i+1]))

            total_distance = sum(segment_distances)

            # Segment start times based on proportional distances
            segment_times = [self.start_time]
            for i in range(len(segment_distances)):
                segment_fraction = segment_distances[i] / total_distance
                segment_time     = segment_fraction * total_time
                next_time        = segment_times[-1] + timedelta(seconds=segment_time)
                segment_times.append(next_time)

            # Generate trajectory for each segment
            for i in range(len(self.waypoints) - 1):
                start_pos = self.waypoints[i]
                end_pos   = self.waypoints[i+1]
                start_t   = segment_times[i]
                end_t     = segment_times[i+1]

                current_time = start_t
                while current_time <= end_t:
                    # Calculate progress within this segment
                    progress = min(1.0, (current_time - start_t).total_seconds() / 
                               max(1, (end_t - start_t).total_seconds()))

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

        return list(zip(self.timestamps, self.trajectory_points))

    def create_shapefile(self, output_path, trajectory_type='points'):
        """
        Create a shapefile from the trajectory.

        INPUT:
            output_path (str): Path to save the shapefile
            trajectory_type (str): Type of shapefile to create ('points', 'line', or 'waypoints')
            
        OUTPUT:
            str: Path to the created shapefile
        """
        # Generate trajectory if it hasn't been generated yet
        if not self.trajectory_points:
            self.generate_trajectory()

        # Create GeoDataFrame
        if trajectory_type == 'points':
            # Create points for each position
            geometry = [Point(lon, lat) for lat, lon in self.trajectory_points]
            data = {
                'ship_id': [self.ship_id] * len(self.trajectory_points),
                'timestamp': self.timestamps,
                'latitude': [pos[0] for pos in self.trajectory_points],
                'longitude': [pos[1] for pos in self.trajectory_points],
                'geometry': geometry
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        elif trajectory_type == 'line':
            # Create a single LineString for the whole trajectory
            line_coords = [(lon, lat) for lat, lon in self.trajectory_points]
            line = LineString(line_coords)
            data = {
                'ship_id': [self.ship_id],
                'start_time': [self.start_time],
                'end_time': [self.end_time],
                'speed_knots': [self.ship_speed],
                'distance_nm': [self.route_distance if self.route_distance else None],
                'geometry': [line]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        elif trajectory_type == 'waypoints':
            # Create points for each waypoint
            geometry = [Point(lon, lat) for lat, lon in self.waypoints]
            data = {
                'ship_id': [self.ship_id] * len(self.waypoints),
                'point_type': ['start'] + ['waypoint'] * (len(self.waypoints) - 2) + ['end'] if len(self.waypoints) > 1 else ['single'],
                'latitude': [pos[0] for pos in self.waypoints],
                'longitude': [pos[1] for pos in self.waypoints],
                'geometry': geometry
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

        else:
            raise ValueError(f"Invalid trajectory_type: {trajectory_type}. Must be 'points', 'line', or 'waypoints'.")

        # Save to shapefile
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gdf.to_file(output_path)
        print(f"Created shapefile at {output_path}")
        return output_path

    def create_time_series_shapefile(self, output_path, interval_minutes):
        """
        Create a time series of ship positions that can be used to simulate GNSS tracking.

        INPUT:
            output_path (str): Path to save the shapefile
            interval_minutes (int): Time interval between points in minutes

        OUTPUT:
            str: Path to the created shapefile
        """
        # Generate trajectory with specified interval
        trajectory = self.generate_trajectory(interval_minutes=interval_minutes)

        # Create GeoDataFrame
        geometry = [Point(lon, lat) for _, (lat, lon) in trajectory]
        data = {
            'ship_id':      [self.ship_id] * len(trajectory),
            'timestamp':    [t for t, _ in trajectory],
            'datetime_str': [t.strftime('%Y-%m-%d %H:%M:%S') for t, _ in trajectory],
            'latitude':     [pos[0] for _, pos in trajectory],
            'longitude':    [pos[1] for _, pos in trajectory],
            'geometry':      geometry
        }

        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # Save to shapefile
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gdf.to_file(output_path)
        print(f"Created time series shapefile at {output_path}")
        return output_path

    def plot_route(self, show_trajectory=True, show_waypoints=True, save_path=None):
        """
        Plot the ship's route on a map.

        INPUT:
            show_trajectory (bool): Whether to show the full trajectory
            show_waypoints (bool): Whether to show the waypoints
            save_path (str): Path to save the figure
        """
        # Generate trajectory if needed
        if not self.trajectory_points and show_trajectory:
            self.generate_trajectory()

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
            gpd.GeoSeries(self.ocean_polygons).plot(ax=ax, color='lightgreen', edgecolor='darkgreen', alpha=0.6)

        # Plot trajectory
        if show_trajectory and self.trajectory_points:
            traj_x = [point[1] for point in self.trajectory_points]
            traj_y = [point[0] for point in self.trajectory_points]
            ax.plot(traj_x, traj_y, 'b-', linewidth=1.5, alpha=0.7, label='Ship Trajectory')

        # Plot waypoints
        if show_waypoints and self.waypoints:
            way_x = [point[1] for point in self.waypoints]
            way_y = [point[0] for point in self.waypoints]
            ax.plot(way_x, way_y, 'r--', linewidth=1.5, label='Planned Route')
            ax.scatter(way_x, way_y, c='red', s=50, zorder=5, label='Waypoints')

        # Plot start and end points
        start_x, start_y = self.start_position[1], self.start_position[0]
        end_x, end_y = self.end_position[1], self.end_position[0]

        ax.scatter(start_x, start_y, c='green', s=100, marker='^', zorder=10, label='Start')
        ax.scatter(end_x, end_y, c='purple', s=100, marker='x', zorder=10, label='End')

        # Add labels and legend
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Ship Route: {self.ship_id}')
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Route plot saved to {save_path}")

        # Show plot
        plt.show()

# Example usage:
if __name__ == "__main__":
    helgoland = (54.199416, 7.892937) 
    Murmansk  = (69.345572, 33.646450)
    oslo      = (59.9139,   10.7522)
    stavanger = (58.9700,   5.7331)

    # Path to Natural Earth coastline data
    # Download from https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/
    coastline_path = os.path.join(Constants.shapefile_path, "10m_physical", "ne_10m_ocean.shp")

    # Create ship trajectory
    ship = ShipTrajectory(
        ship_id        = "SHIP001",
        ship_speed     = 15.0,  # 15 knots
        start_position = helgoland,
        start_time     = datetime.now(),
        end_position   = Murmansk,
        coastline_path = coastline_path
    )

    # Plan and plot route
    ship.plot_route(save_path="ship_route_map.png")

    # Generate trajectory and create shapefile
    ship.generate_trajectory(interval_minutes=30)
    ship.create_shapefile("ship_trajectory.shp", trajectory_type="points")
    ship.create_shapefile("ship_route.shp",      trajectory_type="line")
    ship.create_shapefile("ship_waypoints.shp",  trajectory_type="waypoints")

    # Create time series for GNSS tracking simulation
    ship.create_time_series_shapefile("ship_tracking.shp", interval_minutes=5)
