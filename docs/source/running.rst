Running the simulator
=====================

The simulator has two entry points, one per analysis mode. Both read ``Configuration.ini`` and **must be
launched from the repository root** (the constants resolve all data and output paths from the working directory).

.. code-block:: bash

   # from the repository root
   python SRC/main_simulation_areaCoverage.py
   python SRC/main_simulation_movingTarget.py

Area coverage
-------------

``SRC/main_simulation_areaCoverage.py`` evaluates *how often* a region is covered by reflections. It
builds the receiver and GNSS constellations, and runs :doc:`Documentation/object_definitions/coverage` over a
region predefined by a shapefile (``[COVERAGE].shapefile_path``). Outputs are per-constellation
track-visit maps in ``OUTPUT/``. The default region, the Norwegian exclusive economic zone (EEZ) from
Marine Regions, is not bundled; the repository ``README.md`` gives the command that downloads it to
``DATA/eez/eez_norge.shp``.

Moving-target detection
-----------------------

``SRC/main_simulation_movingTarget.py`` evaluates *when* a moving target (e.g. a ship) is covered by a
reflection, and what the link budget predicts for its detection.
It builds a :doc:`Documentation/object_definitions/ship_trajectory` from ``[MOVING TARGET]``,
propagates the constellations, and runs :doc:`Documentation/object_definitions/moving_target_detection` in
either ``specular`` or ``backscatter`` mode. Outputs are per-band detection plots (and an optional
animation) in ``OUTPUT/``.

Monte Carlo feasibility runs
----------------------------

``SRC/run_monte_carlo.py`` repeats the moving-target analysis over many randomised routes, epochs, sea
states and receiver settings, and reports how often a ship would be detected. It needs an extra
``[MONTE CARLO]`` section in the INI; see :doc:`monte_carlo`.

What a run writes
-----------------

Each script prints its progress and a summary to the console and writes its figures to ``OUTPUT/``;
:doc:`outputs` lists every file. Expensive intermediate results (the coverage grid, the antenna
patterns, generated TLEs and planned ship routes) are cached under ``DATA/`` and reused by later runs.

Run time
--------

The run time grows with the number of receivers, the number of GNSS satellites and bands, and the
simulated duration divided by the step size (``simulation.step_size_s`` in ``simulator_settings.yaml``).
As a reference on a current laptop: the shipped area-coverage scenario (two receivers, GPS and Galileo,
6 h) takes about two minutes, and a 20 h moving-target route with five bands about three minutes. The
moving-target run only propagates satellites while they can see the ship, so it scales with the number
of passes rather than with the route length alone.

Processing pipeline overview
----------------------------

.. figure:: images/pipeline_flow.svg
   :alt: Processing pipeline flow, from the INI file through to the OUTPUT plots
   :align: right
   :width: 800px
   :figclass: pipeline-shift

   The processing flow. The core is shared; the two reflection modes share the link budget but
   produce different detection metrics, and area coverage counts contiguous track visits from the
   selected reflection geometry (the specular point, or the near-backscatter swath around the
   backscatter reflection point), using the link budget only as a validity gate.

Both modes share the same core. The processing flow is:

#. parse the INI (:doc:`Documentation/configuration`) into settings and a validated band plan;
#. build the LEO receiver constellation and the GNSS transmitter constellations
   (:doc:`Documentation/object_definitions/satellites`) and propagate them;
#. solve the reflection geometry per (transmitter, receiver) pair
   (:doc:`Documentation/object_definitions/reflection` or
   :doc:`Documentation/object_definitions/backscatter_point`);
#. evaluate the link budget (:doc:`Documentation/functions/link_budget`) using the ocean and ship RCS
   models (:doc:`Documentation/object_definitions/ocean_target`,
   :doc:`Documentation/object_definitions/ship_target`);
#. accumulate coverage counts or detection events and write the plots.

.. note::

   The first launch in a fresh session can pause for several seconds while ``astropy`` (pulled in by
   ``pymap3d``) imports cold and the antivirus scans the scientific libraries; the second run is fast. Use
   *Run Without Debugging* in the IDE, or pre-warm the import once, if the debugger surfaces an interrupt
   during that import.

.. caution::

   ECI/ECEF conversion depends on Earth-orientation data from ``astropy-iers-data``. Offline execution is
   allowed to use the installed table even when its predictive portion is stale, which favors
   reproducibility over a hard failure. Update ``astropy-iers-data`` (or provide current IERS data) before
   precision geolocation studies, especially for future dates.
