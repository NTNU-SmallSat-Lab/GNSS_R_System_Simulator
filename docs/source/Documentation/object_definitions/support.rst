Support modules (constants, satellite data I/O, plotting)
=========================================================

Three support modules that the rest of the simulator leans on: the constants/lookup tables, the TLE
read/write/generate code, and the static plotting helpers.

Constants
---------

.. module:: object_definitions.Constants_def

Summary
^^^^^^^

``Constants`` is the single place for the numbers and tables the project shares. It groups:

.. list-table:: What it holds
   :widths: 32 54 14
   :header-rows: 1

   * - Group
     - Contents
     - Unit
   * - Physical constants
     - speed of light, Boltzmann constant, Earth GM, WGS-84 radii, flattening.
     - SI
   * - GNSS signals
     - ``BANDS_CENTER_FREQUENCIES_Hz`` (carriers), ``GNSS_EIRP_DBW`` (transmit EIRP),
       ``GNSS_CHIP_RATE_HZ`` (code rate), ``BAND_TO_CONSTELLATION``.
     - Hz, dBW
   * - Data paths
     - every ``DATA/``, ``OUTPUT/`` and ephemeris folder, built from the working directory.
     - --

.. important::

   Every path is built from ``os.getcwd()``, so the simulator **must** be launched from the repo root or
   the data and output lookups break.

The lookup helpers turn a band name into its transmit EIRP or code chip rate:

.. autoclass:: object_definitions.Constants_def.Constants
   :members: get_tx_eirp_dbw, get_tx_eirp_dbw_for_band, get_constellation_for_band, get_chip_rate_hz_for_band

.. caution::

   ``GNSS_EIRP_DBW`` values are scenario assumptions, not a traceable per-spacecraft antenna/link
   calibration. ``G1`` and ``G2`` use nominal GLONASS carrier centres and do not model FDMA channel
   offsets. Absolute received-power work should replace these tables with signal-, channel- and
   transmitter-specific values.

Satellite data I/O
------------------

.. module:: object_definitions.ReadWriteSatData

Summary
^^^^^^^

``ReadWriteSatData`` converts between the simulator's orbits and TLEs (Two-Line Element Sets, the standard
text format for satellite orbits). It generates a TLE from an ``Orbit_specification`` (solving for the SGP4
mean elements that reproduce the requested osculating orbit), reads TLE files into satellite records, and
can download fresh TLEs.

.. list-table:: Key public methods
   :widths: 40 60
   :header-rows: 1

   * - Method
     - What it does
   * - ``generate_tle_data_string(satellite)``
     - Build a three-line TLE string from a satellite's orbit.
   * - ``read_tle_data(tle_file_path)``
     - Parse a TLE file into per-satellite records.
   * - ``download_and_read_tle(path)``
     - Fetch and read a TLE file.

.. autoclass:: object_definitions.ReadWriteSatData.ReadWriteSatData
   :no-members:

.. automethod:: object_definitions.ReadWriteSatData.ReadWriteSatData.generate_tle_data_string

Plotting
--------

.. module:: object_definitions.Static_classes_def

Summary
^^^^^^^

``Static_classes_def`` holds the ``Plotting`` class -- static helpers that draw reflection traces and
orbits over a map, plus an interactive 3-D globe export. They are visualization-only and read the same
`Natural Earth <https://www.naturalearthdata.com/>`_ land/ocean data used elsewhere.

.. list-table:: Key methods
   :widths: 44 56
   :header-rows: 1

   * - Method
     - What it does
   * - ``plot_reflection_traces_gnss_const(...)``
     - Draw the reflection tracks for each GNSS constellation.
   * - ``plot_reflection_traces_total(...)``
     - Draw the combined reflection tracks.
   * - ``plot_orbits(sat_const)``
     - Plot the satellite orbits.

.. autoclass:: object_definitions.Static_classes_def.Plotting
   :no-members:

Module Assumptions and Limitations
----------------------------------

- **Working directory matters.** ``Constants`` resolves all paths from the current directory; run from the
  repo root.
- **Auto-generated TLEs have no drag.** ``ReadWriteSatData`` writes TLEs without a ``B*`` drag term, so long
  propagations drift (see the satellites page).
- **Signal tables are simplified.** EIRP entries are fixed per constellation/band and GLONASS G1/G2 use
  nominal rather than channel-specific frequencies.
- **Plotting is headless.** The plotting helpers force a non-interactive backend and write files; they are
  not part of the physics path.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_read_write_sat_data.py``
     - TLE parsing snapshots and derived orbital quantities.
   * - ``tests/test_satellite.py``
     - Generated TLE files exist and are loadable; multi-record index/name selection keeps metadata and
       Skyfield propagation aligned.
   * - ``tests/test_constants_eirp.py`` / band lookups in ``tests/test_delay_doppler.py``
     - The EIRP and chip-rate lookups resolve correctly and cover every EIRP band.

References
----------

- Vallado, D. A. & Cefola, P., "Two-line element sets -- practical use and accuracy", IAC, 2012.
- CelesTrak, *NORAD Two-Line Element Set Format*, celestrak.org.
