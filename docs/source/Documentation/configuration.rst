INI configuration reference
===========================

Both modes are driven by an INI file at the repo root: ``Configuration_area.ini`` for area coverage and
``Configuration_ship.ini`` for moving-target detection. The loader (``SRC/__init__.py::initialize``) is
**strict** -- a required key that is missing raises, rather than falling back to a hidden default. Which
sections are present is what distinguishes the two modes.

This page lists the sections and keys. Per-section physics detail lives on the module pages, linked from
each section.

.. note::

   Engine/numerical settings live in ``simulator_settings.yaml`` at the repo root, kept separate from the
   per-mission INI: the simulation ``step_size_s``, and the antenna-pattern angular resolution
   ``theta_resolution`` / ``phi_resolution`` plus the steering step counts ``along_track_steering_steps`` /
   ``across_track_steering_steps``. It is loaded strictly -- a missing file or key raises.

[SIMULATION SETTINGS]
---------------------

.. list-table::
   :widths: 30 56 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``starttime``
     - Simulation epoch (UTC), format ``dd.MM.YYYY HH:mm:ss``.
     - --
   * - ``simulationDuration``
     - Window length (area-coverage mode).
     - h

[MOVING TARGET] / [SHIP]
------------------------

Moving-target mode only. See :doc:`object_definitions/ship_trajectory` and
:doc:`object_definitions/ship_target`.

.. list-table::
   :widths: 30 56 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``speed``
     - Ship ground speed.
     - knots
   * - ``start_pos`` / ``end_pos``
     - Route endpoints, ``[latitude, longitude]``.
     - deg
   * - ``[SHIP].length_m`` / ``beam_m`` / ``freeboard_m``
     - Ship size (enables the ship return).
     - m

[ANTENNA PATTERN] / [ANTENNA]
-----------------------------

See :doc:`object_definitions/antenna`.

.. list-table::
   :widths: 34 52 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``elements_along_x`` / ``elements_along_y``
     - Array element count per axis.
     - --
   * - ``d_x`` / ``d_y``
     - Element spacing.
     - m
   * - ``along_track_steering`` / ``across_track_steering``
     - Steering extent ``[min, max]`` (two-plane): along-track tilts the beam in the flight direction
       (antenna :math:`x`), across-track perpendicular (antenna :math:`y`). Step counts are in the simulator YAML.
     - deg
   * - ``bands``
     - Constellation-specific signals, e.g. ``["L1", "E1", "B1C"]``.
     - --
   * - ``footprint_db``
     - Footprint contour level (below 0 dB).
     - dB

[ORBITS] / [GNSS CONSTELLATIONS]
--------------------------------

See :doc:`object_definitions/satellites`.

.. list-table::
   :widths: 30 56 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``orbit1``, ``orbit2``, ...
     - ``[inclination, eccentricity, raan, argPeriapsis, trueAnomaly, semiMajorAxis]``, read in order.
     - deg / -- / m
   * - ``gnss_tle_files``
     - TLE files (the constellation is matched from the filename).
     - --

[COVERAGE] / [REFLECTION]
-------------------------

See :doc:`object_definitions/coverage`, :doc:`object_definitions/reflection`, and
:doc:`object_definitions/backscatter_point`.

.. list-table::
   :widths: 34 52 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``[COVERAGE].shapefile_path``
     - Region / trajectory shapefile.
     - --
   * - ``[COVERAGE].grid_size_meters``
     - Coverage cell size.
     - m
   * - ``[REFLECTION].reflection_type``
     - ``"specular"`` or ``"backscatter"``; other values raise.
     - --
   * - ``[REFLECTION].max_central_angle_deg``
     - Tx-Rx geometric gate, finite and in ``(0, 180]``.
     - deg
   * - ``[REFLECTION].max_incidence_angle_deg``
     - BRP incidence gate (backscatter), finite and in ``(0, 90]``.
     - deg

[LINK BUDGET] / [OCEAN]
-----------------------

See :doc:`functions/link_budget`, :doc:`functions/delay_doppler`, and
:doc:`object_definitions/ocean_target`.

.. list-table::
   :widths: 36 50 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``system_losses_db`` / ``polarization_losses_db``
     - Receiver-side losses.
     - dB
   * - ``noise_bandwidth_hz`` / ``system_noise_temperature_k``
     - Thermal-noise terms.
     - Hz, K
   * - ``detection_sinr_threshold_db``
     - Ship detection threshold.
     - dB
   * - ``ocean_patch_integration_enabled`` / ``ocean_patch_grid_size``
     - Patch-integrated ocean clutter.
     - --
   * - ``delay_doppler_cell_enabled`` / ``coherent_integration_time_s``
     - Use the first-order DDM-cell estimate as clutter area; ``T_coh`` sets its Doppler bin and the
       coherent sensitivity factor ``B*T_coh``. Exact specular samples use the footprint cap.
     - --, s
   * - ``num_incoherent_looks``
     - Independent incoherent looks M; ``sqrt(M)`` is used as a detection-sensitivity approximation
       from reduced estimator variance, not as lower mean thermal noise.
     - --
   * - ``[OCEAN].ocean_wave_height_m`` ... ``ocean_inverse_wave_age``
     - Sea state: significant wave height :math:`H_s` (the roughness model uses
       :math:`\sigma_h\simeq H_s/4`), wind speed / direction, temperature, salinity, wave age.
     - m, m/s, deg, degC, PSU, --

[PLOT]
------

.. list-table::
   :widths: 34 52 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``detection_bbox_buffer_deg``
     - Padding around the trajectory for plots.
     - deg
   * - ``create_animation`` / ``animation_filename`` / ``animation_fps``
     - Optional moving-target animation.
     - --
