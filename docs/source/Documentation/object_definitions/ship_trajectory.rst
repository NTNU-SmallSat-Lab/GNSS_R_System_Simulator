Ship trajectory (``object_definitions.ShipTrajectory_def``)
===========================================================================

.. module:: object_definitions.ShipTrajectory_def

Executive Summary
-----------------

``ShipTrajectory`` turns a start point, an end point, a start time and a speed into a **time-stamped,
land-avoiding maritime route**. It is the moving-target source for the detection mode: it produces the
ordered ``(timestamp, latitude, longitude)`` samples that
:class:`~object_definitions.MovingTargetDetection_def.MovingTargetDetection` queries to place the target
at each epoch, plus a trajectory shapefile (referenced by ``[COVERAGE].shapefile_path``) and a metadata
JSON for caching.

The route is planned over a **navigation mesh** built from the Natural Earth ocean polygons
(:data:`Constants.earth_coastline_path`): grid points that fall in open water become graph nodes, nearby
nodes are connected if the straight leg between them does not cross land, and a shortest path is found
with ``networkx``. Endpoints on land are snapped to the nearest safe water point. The result is
simplified, time-sampled at a fixed interval, and cached so an unchanged run reloads instantly.

Interface
---------

.. list-table:: Consumed / produced
   :widths: 22 78
   :header-rows: 1

   * - Role
     - Quantity
   * - Consumes (constructor)
     - ``ship_id``; ``ship_speed_knot``; ``start_position`` / ``end_position`` as ``(latitude, longitude)``;
       ``start_time`` (UTC datetime); and meshing controls ``interval_minutes``, ``grid_resolution_km``,
       ``delta_boudingBox_km``, ``fast_path_gen_method``, ``mesh_connection_radius_km``, ``water_buffer_km``.
       The Natural Earth ocean shapefile is read from :data:`Constants.earth_coastline_path`.
   * - Produces
     - ``trajectory_points`` (list of ``(lat, lon)``) and ``timestamps`` (aligned UTC datetimes);
       ``waypoints`` and ``route_distance``; ``end_time`` (derived from distance / speed); and
       ``shapefile_path`` (the time-series shapefile + metadata JSON, written under
       :data:`Constants.autogen_ship_shapefiles`). ``get_position_at_time`` is the per-epoch query.

.. note::

   Positions are ``(latitude, longitude)`` throughout this class, but the shapefile geometry stores
   ``(lon, lat)`` because shapely / GIS use x = longitude.

Detailed Module Description
---------------------------

Endpoint conditioning
^^^^^^^^^^^^^^^^^^^^^^

Each endpoint is first passed through ``get_safe_water_point``: if it falls on land it is pushed out to
the nearest point on the ocean boundary plus a ``water_buffer`` margin (default 22.2 km = 12 nautical
miles, the territorial-sea limit). This guarantees both ends are routable water nodes.

Navigation mesh and graph
^^^^^^^^^^^^^^^^^^^^^^^^^^

A bounding box around the (buffered) endpoints is gridded at ``grid_resolution_km``; the longitude step
is widened with latitude (``fn.delta_lon_from_distance``) so spacing stays roughly metric. Only grid
points inside the ocean polygon become graph nodes. Nodes within ``mesh_connection_radius`` of each other
are joined by an edge **iff** the straight leg does not cross land, with the edge weight set to the
great-circle (haversine) leg length. A ``cKDTree`` over the water points makes the radius queries fast.

Shortest path and component bridging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The route is the minimum-weight ``start -> end`` path (``networkx.shortest_path``, Dijkstra on the
haversine weights). If the start and end fall in **disconnected** components (e.g. separated by a narrow
strait the grid missed), ``_bridge_components`` runs a Delaunay triangulation across the two component
point sets and adds the shortest land-free triangulation edges (and, if needed, midpoints in narrow
passages) until the components connect.

Path simplification and sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``simplify_path`` applies a Douglas-Peucker-style filter: a waypoint is dropped when the leg that skips
it still does not cross land, leaving only the corners that actually matter. The simplified waypoints are
then time-sampled. ``generate_trajectory_equal_time`` distributes the total travel time across the
segments in proportion to their length and emits a sample every ``interval_minutes``, interpolating
linearly in ``(lat, lon)`` within each segment; ``generate_trajectory_equal_dist`` instead emits a sample
every fixed number of metres. The total travel time follows from the route distance and the ship speed,
``end_time = start_time + route_distance / ship_speed``.

Distance and bearing
^^^^^^^^^^^^^^^^^^^^^

Leg lengths use the haversine great-circle distance on a sphere of radius
:data:`Constants.RADIUS_EARTH_EQUAT`,

.. math::
   :label: ship-haversine

   a = \sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1\,\cos\phi_2\,\sin^2\!\frac{\Delta\lambda}{2}, \qquad
   d = 2R\,\arcsin\!\sqrt{a},

with latitudes :math:`\phi` and longitudes :math:`\lambda`. The initial bearing between two points is

.. math::
   :label: ship-bearing

   \theta = \operatorname{atan2}\!\big(\sin\Delta\lambda\,\cos\phi_2,\;
   \cos\phi_1\,\sin\phi_2 - \sin\phi_1\,\cos\phi_2\,\cos\Delta\lambda\big),

normalised to ``[0, 360)`` degrees.

Caching
^^^^^^^

``create_trajectory_files`` writes the time-series shapefile and a metadata JSON (positions, speed,
start time, interval, mesh parameters). On the next construction, ``_cache_matches_inputs`` reloads that
cache only when the start / end positions, start time, speed and interval all match; any mismatch (the
metadata is printed) triggers a full re-plan, because the trajectory is time-sampled from the speed.

Public interface
^^^^^^^^^^^^^^^^

.. autoclass:: object_definitions.ShipTrajectory_def.ShipTrajectory
   :members:

Module Assumptions and Limitations
----------------------------------

- **Spherical haversine.** Distances use a sphere at the equatorial radius, not the WGS-84 ellipsoid; the
  sub-percent error is negligible against the routing-grid resolution.
- **Linear interpolation between waypoints.** Within a segment the position is interpolated linearly in
  ``(lat, lon)`` rather than along a great circle or rhumb line. Fine for short legs; coarser for very
  long ones.
- **Constant speed.** Time sampling assumes a single cruise speed; there is no acceleration, current, or
  weather model.
- **Routing fidelity is grid-limited.** Land avoidance quality depends on ``grid_resolution_km`` and
  ``mesh_connection_radius_km``; ``fast_path_gen_method`` trades route optimality for speed. Both
  endpoints must be reachable open water (they are snapped to a 12 nm buffer if on land).
- **Nearest-sample query.** ``get_position_at_time`` returns the closest stored sample, not an
  interpolated position.
- **Shapefiles are for GIS.** They are written in EPSG:4326 for visualization; the simulation consumes
  ``trajectory_points`` / ``timestamps`` directly.

Validation and Tests
--------------------

The full constructor needs the >100 MB Natural Earth mesh, so ``tests/test_ship_trajectory.py`` bypasses
``__init__`` (via ``object.__new__``) and pins the argument-driven helpers:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Test group
     - What it pins
   * - ``TestHaversineDistance``
     - Zero distance for identical points; one degree of latitude ~ 111 km; Oslo->Trondheim ~ 392 km;
       antipodal points ~ :math:`\pi R`; symmetry.
   * - ``TestHaversineVectorized``
     - The vectorized form returns one distance per pair and matches the scalar haversine; the
       ``clip(a, 0, 1)`` guard returns 0 (not NaN) for coincident points.
   * - ``TestCalculateBearing``
     - Due north / east / south / west give 0 / 90 / 180 / 270 deg; results always lie in ``[0, 360)``.
   * - ``TestTrajectoryCacheLoading``
     - Matching metadata reloads the cache (no re-plan); a changed ``ship_speed_m_s`` marks the cache
       stale and forces a re-plan.

User Guide
----------

The route inputs come from the moving-target configuration; the mesh-resolution knobs are currently set
where the trajectory is constructed in ``main_simulation_movingTarget.py`` (not exposed in the INI).

.. list-table:: Route inputs (INI)
   :widths: 34 66
   :header-rows: 1

   * - INI key
     - Constructor argument
   * - ``[MOVING TARGET].speed``
     - ``ship_speed_knot`` (knots).
   * - ``[MOVING TARGET].start_pos``
     - ``start_position`` as ``[latitude, longitude]``.
   * - ``[MOVING TARGET].end_pos``
     - ``end_position`` as ``[latitude, longitude]``.
   * - ``[SIMULATION SETTINGS].starttime``
     - ``start_time`` (UTC epoch).

.. code-block:: python

   from datetime import datetime, UTC
   from object_definitions.ShipTrajectory_def import ShipTrajectory

   ship = ShipTrajectory(
       ship_id="DarkShip_001",
       ship_speed_knot=15.0,
       start_position=(54.199416, 7.892937),   # (lat, lon) -- Helgoland
       start_time=datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC),
       end_position=(69.345572, 33.646450),    # (lat, lon) -- Murmansk
       interval_minutes=10,
       grid_resolution_km=150,
       fast_path_gen_method=False,
   )

   pos = ship.get_position_at_time(datetime(2026, 5, 31, 18, 0, 0, tzinfo=UTC))

References
----------

- Sinnott, R. W., "Virtues of the Haversine", *Sky and Telescope* 68(2), 1984.
- Douglas, D. & Peucker, T., "Algorithms for the reduction of the number of points required to represent
  a digitized line or its caricature", *Cartographica* 10(2), 1973.
- Hagberg, A., Schult, D. & Swart, P., "Exploring network structure, dynamics, and function using
  NetworkX", *Proc. 7th Python in Science Conf. (SciPy)*, 2008.
