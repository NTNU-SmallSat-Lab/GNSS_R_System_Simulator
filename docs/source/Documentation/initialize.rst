Configuration loader (initialize)
=================================

Summary
-------

``SRC/__init__.py`` turns the two input files into the settings that every run consumes. ``initialize``
reads the scenario INI (``Configuration.ini``) and the engine settings (``simulator_settings.yaml``),
validates them, regenerates the antenna patterns when ``[ANTENNA PATTERN]`` is present, and returns the
raw ``ConfigParser`` together with a settings dictionary. Both entry scripts and the Monte Carlo harness
call it first:

.. code-block:: python

   from __init__ import initialize

   config, settings = initialize("Configuration.ini")

The loader is **strict**: a required key that is missing raises with the name of the key, and invalid
values (an unknown ``earth_model``, a non-finite gate angle, a steering extent outside the array's scan
limit) raise before any propagation starts. The keys themselves are listed in :doc:`configuration`.

What it builds
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Settings entry
     - Content
   * - ``start_time`` / ``step_size`` / ``sim_duration``
     - UTC epoch, propagation step [s] from the YAML, area-coverage window [h].
   * - ``orbits`` / ``No_satellites``
     - One element list per ``orbitN`` key, with the epoch appended; the receiver count.
   * - ``GNSS_constellations``
     - The TLE files, each with the constellation matched from its file name.
   * - ``selected_signals``
     - The validated band plan, one entry per usable band:
       ``{band, constellation, freq_hz, eirp_dbw, chip_rate_hz}`` (see ``build_selected_signals``).
   * - ``antenna_pattern_settings`` / ``antenna_pattern_files`` / ``footprint_dB``
     - Array geometry, the pattern files keyed by (across-track, along-track, frequency), and the
       footprint contour level.
   * - ``reflection_type`` / ``earth_model`` / gate angles
     - The reflection mode and the geometric gates.
   * - ``link_budget``
     - Receiver losses, noise and processing settings, merged with the ``[OCEAN]`` sea state.
   * - ``shapefile_path`` / ``grid_size_m`` / ``max_bistatic_bksc_angle_deg``
     - Area-coverage inputs (``None`` when the section is absent).
   * - ``start_coord_ll`` / ``end_coord_ll`` / ``moving_target_speed`` / ``ship``
     - Moving-target inputs (``None`` when ``[MOVING TARGET]`` is absent).
   * - ``detection_bbox_buffer_deg`` / ``create_animation`` / ``animation_settings``
     - Plot settings.

The band plan
-------------

``[ANTENNA].bands`` names constellation-specific signals. ``build_selected_signals`` keeps a band only if
it is a known GNSS band, has a transmit EIRP in ``Constants.GNSS_EIRP_DBW``, and belongs to a loaded
constellation; every other band is dropped with a warning, so a run continues with the usable subset.
The moving-target detector evaluates each kept band separately; area coverage uses the plan only to
choose the participating constellations.

API reference
-------------

.. autofunction:: __init__.initialize

.. autofunction:: __init__.build_selected_signals

.. autofunction:: __init__.load_simulator_settings

.. autofunction:: __init__.tle_fit_weights_from_yaml

.. autofunction:: __init__.parse_antenna_angles

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_initialize.py``
     - Parsing of every section, the strict missing-key and invalid-value errors, and the mode-specific
       requirements.
   * - ``tests/test_band_selection.py``
     - The band plan: valid bands, warnings and drops for unknown, EIRP-less and unloaded bands.
