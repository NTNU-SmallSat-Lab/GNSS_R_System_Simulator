Outputs
=======

Every run writes its figures to ``OUTPUT/`` in the repository root and prints a summary to the console.
Files with the same name are overwritten by the next run, so copy ``OUTPUT/`` away to keep a result.
Intermediate data that is expensive to compute is cached under ``DATA/`` (see `Cached data`_).

In the file names below, ``<CONST>`` is a GNSS constellation (``GPS``, ``GALILEO``, ``GLONASS``,
``BEIDOU``), ``<band>`` a signal from ``[ANTENNA].bands`` (e.g. ``L1``), ``<mode>`` the
``[REFLECTION].reflection_type`` (``specular`` or ``backscatter``), and ``<contour>`` the
``[ANTENNA].footprint_db`` level (e.g. ``-0.5dB``).

Area coverage
-------------

Coverage is computed per constellation, not per band (see :doc:`Documentation/object_definitions/coverage`).
In backscatter mode every coverage map carries the suffix ``_backscatter``; specular maps have no suffix.

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - File
     - Content
   * - ``Coverage_<CONST>[_backscatter].png``
     - The region with the reflection tracks (specular points, or near-backscatter swaths) of one
       constellation.
   * - ``Coverage_grid_<CONST>[_backscatter].png``
     - The gridded region, each cell coloured by its number of contiguous track visits.
   * - ``Coverage_all_configurations[_backscatter].png`` / ``Coverage_grid_all_configurations[_backscatter].png``
     - The same two maps with all constellations combined.
   * - ``Reflection_points_<CONST>.png``
     - World map of the reflection points of one constellation over the whole window.
   * - ``Reflection_traces_<CONST1>_<CONST2>....png``
     - World map of the reflection traces of all constellations together.
   * - ``Orbits.png`` / ``Orbits.html``
     - Receiver and GNSS orbits; the HTML file is an interactive, rotatable 3D view that opens offline in
       a browser.
   * - ``antennaRadiationPattern/<freq>MHz.png``
     - Boresight array pattern per carrier frequency, written when ``[ANTENNA PATTERN]`` regenerates the
       patterns.

The console prints, per constellation and for the combination, the coverage percentage, the number of
covered cells out of all cells, the total number of reflections and the mean number of visits per covered
cell.

Moving-target detection
-----------------------

Detection results are per (constellation, band): each band has its own carrier frequency and transmit
EIRP (see :doc:`Documentation/object_definitions/moving_target_detection`). Every per-band figure also
exists as a ``combined`` version with all constellations and bands together, colour-coded by
constellation.

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - File
     - Content
   * - ``ship_route_<ship_id>.png``
     - The planned ship route (the entry script uses the ship ID ``DarkShip_001``).
   * - ``Footprint_overlap_<mode>_<CONST>_<band>_<contour>.png``
     - The ship track and every reflection whose footprint overlapped the ship within one time step.
       This is a geometric overlap, not a detection decision.
   * - ``Link_budget_breakdown_<mode>_<CONST>_<band>_power.png``
     - Backscatter only. Ship power against ocean-clutter power per overlap event, with the
       :math:`\mathrm{SCR} = 0` dB line and the thermal floor.
   * - ``..._scr_hist.png`` / ``..._sinr_hist.png``
     - Backscatter only. Histograms of the signal-to-clutter ratio and of the ship SINR, with the
       detection threshold.
   * - ``..._margin.png``
     - Backscatter only. SINR margin above the threshold per overlap event.
   * - ``Forward_shadow_specular_<CONST>_<band>_shadow_fraction.png``
     - Specular only. Histogram of the forward-shadow change relative to the footprint glint,
       :math:`10\log_{10}(|\Delta P_r|/P_\mathrm{ocean})`.
   * - ``Forward_shadow_specular_<CONST>_<band>_shadow_over_noise.png``
     - Specular only. Histogram of the forward-shadow change relative to the detection-equivalent noise,
       :math:`10\log_{10}(|\Delta P_r|/N_\mathrm{eff})`.
   * - ``<animation_filename>_<mode>.gif``
     - Optional animation of the reflections travelling along the track
       (``[PLOT].create_animation = True``).
   * - ``antennaRadiationPattern/<freq>MHz.png``
     - As in area coverage.

The link-budget and forward-shadow figures are skipped, with a console warning, when no overlap event
has a finite ship power or shadow change. On a short route this is common: an overlap needs a receiver
pass whose footprint covers the ship, so routes of a few hours often have none. The console also prints a
per-constellation summary of the event counts, the SCR and SINR ranges and the number of events above the
threshold.

Monte Carlo runs
----------------

``SRC/run_monte_carlo.py`` writes its results and figures to ``--output-dir``; see :doc:`monte_carlo`.

Cached data
-----------

These files are generated on demand and reused by later runs. Deleting them is safe; they are rebuilt
when needed (the antenna patterns only while ``[ANTENNA PATTERN]`` is present in the INI). All of them
are excluded from version control.

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - Location
     - Content and reuse rule
   * - ``<region>_gridded_m.*`` next to ``[COVERAGE].shapefile_path``
     - The coverage grid. Reused whenever the file exists, **also when** ``grid_size_meters`` **changed**:
       delete it after changing the grid size.
   * - ``DATA/ANTENNA/antenna_patterns/``
     - Array-pattern JSONs per frequency and steering angle, plus ``antenna_pattern_metadata.json``.
       Regenerated automatically when the array, steering or resolution settings change.
   * - ``DATA/TLE_EPHEMIRIS/autogenerated/``
     - TLEs generated from the ``[ORBITS]`` elements, one per receiver satellite.
   * - ``DATA/autogenerated_ship_trajectory/``
     - The planned route (shapefile) and its metadata JSON. Reused when the route inputs (positions,
       speed, start time, sampling and mesh settings) are unchanged.
