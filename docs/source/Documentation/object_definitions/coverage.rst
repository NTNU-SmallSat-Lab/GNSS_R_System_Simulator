Coverage
========

.. module:: object_definitions.Coverage_def

Summary
-------

``Coverage`` answers the area-coverage question: *over this region and this time window, how many
continuous reflection-track visits cross each cell?* It lays a grid over a region (from a shapefile),
then builds a track for every contiguous valid segment of every GNSS/LEO pair. Each cell crossed by a
segment receives one visit. The result is a pass-count map, not a count of propagated time samples.

Both reflection geometries are supported: with ``reflection_type = "specular"`` (the default) the track
is the specular-point line, and with ``"backscatter"`` it is the near-backscatter **swath** -- the BRP
track buffered by the ground half-width of the ``max_bistatic_bksc_angle_deg`` acceptance cone (see
below).

This is a **geometric** measure: it depends only on the orbit and constellation geometry. The band's
frequency and EIRP feed only a finiteness check (is the reflection inside the antenna's field of view), not
a detect/no-detect threshold, so two bands of the same constellation give the same map. The band list
therefore just selects *which constellations* take part. (Detectability-gated coverage, where the band
would matter, is future work.)

Interface
---------

.. list-table:: Inputs / outputs
   :widths: 16 28 42 14
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs**
     - ``sat_const`` / ``gnss_const``
     - LEO receiver constellation and GNSS transmitter constellations.
     - --
   * -
     - ``shapefile``
     - Region outline; the grid is generated next to it.
     - --
   * -
     - ``grid_size_deg`` / ``grid_size_m``
     - Cell size, equiangular (deg) or equidistant (m).
     - deg or m
   * -
     - ``earth_model``
     - "spherical" or "elliptical".
     - --
   * -
     - ``max_central_angle_deg``
     - Tx-Rx geometric gate.
     - deg
   * -
     - ``selected_signals``
     - INI band plan (deduplicated to one per constellation).
     - --
   * -
     - ``reflection_type``
     - "specular" (default) or "backscatter".
     - --
   * -
     - ``max_incidence_angle_deg`` / ``max_bistatic_bksc_angle_deg``
     - BRP incidence gate and near-backscatter acceptance half-angle (both required for backscatter).
     - deg
   * - **Outputs**
     - ``coverage_dict``
     - Per-constellation per-cell hit counts.
     - --
   * -
     - ``*_gridded_m.shp`` / ``*_gridded_deg.shp``
     - The generated grid shapefile.
     - --
   * -
     - ``OUTPUT/Coverage_grid_<CONST>.png``
     - Per-constellation track-visit maps (plus a combined map); backscatter maps carry a
       ``_backscatter`` suffix.
     - --

Detailed Module Description
---------------------------

The grid
^^^^^^^^

The region shapefile is turned into a grid of cells (a "fishnet"). An **equidistant** grid (cells a fixed
number of metres across, ``*_gridded_m.shp``) keeps cell area roughly constant; an **equiangular** grid
(cells a fixed number of degrees, ``*_gridded_deg.shp``) is simpler but its cells shrink toward the poles.
The grid is generated once and cached next to the input shapefile, so later runs just load it.

Counting reflections
^^^^^^^^^^^^^^^^^^^^^

For each (LEO, GNSS) pair the specular reflections are found with
:class:`~object_definitions.Reflection_def.Reflection`. Consecutive valid samples form a ``LineString``;
an invalid or out-of-bounds sample splits the track into separate segments. A one-sample segment is kept
as a ``Point``. Every cell intersected by a segment receives at most one visit for that pair and segment,
including cells crossed between propagated samples.

The track-visit count :math:`C` for a cell is

.. math::
   :label: cov-count

   C_\mathrm{cell} = \sum_{\text{Tx, Rx, contiguous segment}}
   \mathbb{1}\big[\text{segment intersects cell and nearest-sample power is finite}\big],

The time, incidence angle and power stored for that visit come from the segment sample nearest the cell
centroid; the crossing time is not interpolated. The finite-power condition acts as the antenna
field-of-view gate. The band changes neither geometric term's truth value, which is why coverage is per
constellation.

Backscatter coverage (the near-backscatter swath)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``reflection_type = "backscatter"`` the track is the **BRP** line
(:class:`~object_definitions.BackScattPoint_def.BRP`): the point where the straight line through the
transmitter and the receiver meets the surface -- exact retroreflection, bistatic angle
:math:`\beta = 0`. Exact alignment is a zero-width condition, so a cell counts as covered while its
bistatic angle -- the angle at the cell between the directions to the transmitter and to the receiver --
stays below the acceptance half-angle :math:`\beta_\mathrm{max}`
(``[COVERAGE].max_bistatic_bksc_angle_deg``). A surface point displaced by :math:`d` across the shared
line of sight opens the bistatic angle by :math:`\beta \simeq d\,(1/R_r - 1/R_t)` (both satellites sit on
the *same* side of the surface, hence the difference of curvatures), so the acceptance region around the
BRP has cross-LOS half-width

.. math::
   :label: cov-swath

   w = \frac{\beta_\mathrm{max}}{1/R_r - 1/R_t}
   \qquad \left(\approx 92\ \mathrm{km\ for\ } \beta_\mathrm{max}=5^\circ
   \mathrm{\ at\ a\ 700\ km\ orbit}\right),

with slant ranges :math:`R_r` (surface to receiver) and :math:`R_t` (surface to transmitter). The BRP
track is buffered by the segment-minimum :math:`w` -- the conservative, smaller semi-axis of the
acceptance ellipse (the in-plane axis is wider by :math:`1/\cos\theta_i`) -- before the cell
intersection; segmentation, the one-visit-per-segment rule and the finite-power antenna gate are
identical to the specular path. :math:`\beta_\mathrm{max} = 5^\circ` matches the ocean RCS model's
near-backscatter Bragg gate, so "covered" then means "sees the geometry the backscatter clutter model
itself calls backscatter".

API reference
^^^^^^^^^^^^^

.. list-table:: Key public methods
   :widths: 42 58
   :header-rows: 1

   * - Method
     - What it does
   * - ``compute_coverage_for_constellation(constellation, freq_hz, eirp_dbw)``
     - Count reflections into the grid for one constellation.
   * - ``calculate_statistics(...)``
     - Summarise the per-cell counts (coverage fraction, track-visit statistics).
   * - ``plot_coverage_gridded_shapefile(...)``
     - Draw the track-visit map for a constellation.
   * - ``plot_reflection_points()``
     - Scatter the raw reflection points.

.. autoclass:: object_definitions.Coverage_def.Coverage
   :no-members:

Module Assumptions and Limitations
----------------------------------

- **Geometric metric.** Coverage counts reflections, not detections; the band's frequency/EIRP only gate
  finiteness. Per-band coverage is meaningless until a detection threshold is added.
- **Conservative backscatter swath.** The swath uses the cross-LOS half-width (the smaller semi-axis of
  the acceptance ellipse), the segment-minimum radius, and the degree-of-latitude conversion for the
  isotropic buffer, so the counted swath never overstates the :math:`\beta` acceptance region -- it
  slightly understates its in-plane and east-west extent.
- **Grid choice matters.** Equiangular cells distort with latitude; use the equidistant (metre) grid for
  area-fair counts.
- **Track-intersection counting.** Counts are one per contiguous Tx/Rx track segment crossing a cell, not
  one per time sample. Results can change with propagation cadence because straight lines interpolate
  between samples and invalid samples split passes.
- **No footprint spreading.** Only the specular track (or the :math:`\beta`-acceptance swath in
  backscatter) is intersected with cells; the receiver antenna footprint is not distributed across
  neighbouring cells.
- **Approximate visit metadata.** Stored times, angles and powers are those of the sample nearest the cell
  centroid, not an interpolated boundary-crossing event.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_coverage.py``
     - Grid generation and loading, bounding-box handling for both grid types, preservation of singleton
       and separated contiguous runs, per-cell counting, and the statistics summary; the backscatter
       swath half-width (analytic anchors, distant-transmitter limit) and the swath-vs-thin-track cell
       counts (a synthetic BRP track covers the neighbouring cell column that the thin line misses).

User Guide
----------

This is the class behind ``python SRC/main_simulation_areaCoverage.py`` (configured by
``Configuration_area.ini``). The relevant INI pieces:

- ``[COVERAGE].shapefile_path`` -- the region outline.
- ``[COVERAGE].grid_size_meters`` (or a degree size) -- the cell size.
- ``[REFLECTION].reflection_type`` -- ``"specular"`` (default) or ``"backscatter"``; backscatter also
  requires ``[REFLECTION].max_incidence_angle_deg``.
- ``[COVERAGE].max_bistatic_bksc_angle_deg`` -- the near-backscatter acceptance half-angle
  (required for backscatter, unused for specular).
- ``[ANTENNA].bands`` -- selects which constellations take part.
- ``[ORBITS]`` / ``[GNSS CONSTELLATIONS]`` -- the receiver and transmitter fleets.

References
----------

- Jales, P. & Unwin, M., "Mission description -- GNSS Reflectometry on TDS-1 with the SGR-ReSI", SSTL, 2015.
