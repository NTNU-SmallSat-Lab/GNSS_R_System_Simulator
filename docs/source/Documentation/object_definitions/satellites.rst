Satellites, constellations and orbits
=====================================

These three modules build and propagate the satellites. An :class:`~object_definitions.Orbit_def.Orbit_specification`
holds the orbital elements; a :class:`~object_definitions.Satellite_def.Satellite` turns them (or a TLE)
into a propagatable object; a :class:`~object_definitions.SatConstellation_def.SatelliteConstellation`
groups satellites and propagates them together. The same constellation class is used for both the LEO
receiver fleet and the GNSS transmitter fleets.

.. raw:: html

   <div style="text-align: center; margin: 1.2em 0;">
     <iframe src="../../_static/orbits.html" width="100%" height="600"
             style="border: 1px solid #ccc; border-radius: 4px; max-width: 760px;"
             loading="lazy" title="Interactive orbit view"></iframe>
     <p style="font-size: 0.9em; color: #5a6b7a; margin: 0.6em auto 0; max-width: 760px;">
       A LEO receiver constellation (two near-polar planes
       at 99.1&deg; inclination and 7300&nbsp;km semi-major axis, ascending nodes 90&deg; apart).
     </p>
   </div>

Orbit specification
-------------------

.. module:: object_definitions.Orbit_def

Summary
^^^^^^^

``Orbit_specification`` is a plain container of six **osculating** classical orbital elements at an epoch
(:numref:`Table %s <tab-orbit-params>`). From the
semi-major axis :math:`a` it derives the mean motion (revolutions per day),

.. math::
   :label: orb-meanmotion

   n = \frac{T_\mathrm{day}}{2\pi\sqrt{a^3 / \mu_\oplus}},

with the seconds-per-day :math:`T_\mathrm{day}` and Earth's gravitational parameter :math:`\mu_\oplus`. The
epoch is stored in the TLE ``YYDDD.DDDDDDDD`` format, ready for
:meth:`TLE generation <object_definitions.ReadWriteSatData.ReadWriteSatData.generate_tle_data_string>`.

.. list-table:: Parameters
   :name: tab-orbit-params
   :widths: 16 16 56 12
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Unit
   * - ``i``
     - float
     - Inclination.
     - deg
   * - ``e``
     - float
     - Eccentricity.
     - --
   * - ``raan``
     - float
     - Right ascension of the ascending node.
     - deg
   * - ``p``
     - float
     - Argument of periapsis.
     - deg
   * - ``nu``
     - float
     - True anomaly.
     - deg
   * - ``a``
     - float
     - Semi-major axis (from Earth's centre).
     - m
   * - ``epochTime``
     - datetime, optional
     - Orbit epoch (UTC); sets the TLE epoch string.
     - --

Inputs are validated at construction: inclination must be in ``[0, 180]`` deg, eccentricity in ``[0, 1)``,
all elements must be finite, the osculating perigee radius :math:`a(1-e)` must lie above the WGS-84
equatorial radius, and an epoch must be timezone-aware UTC.

.. autoclass:: object_definitions.Orbit_def.Orbit_specification
   :members:

Satellite
---------

.. module:: object_definitions.Satellite_def

Summary
^^^^^^^

The object ``Satellite`` is a wrapper around a Skyfield ``EarthSatellite``. It can **either** be initialised from an
``Orbit_specification`` (SGP4 mean elements are fitted and a TLE is generated; see
:meth:`TLE generation <object_definitions.ReadWriteSatData.ReadWriteSatData.generate_tle_data_string>`)
**or** directly from a TLE file. Once the
constellation sets the shared timescale, the propagate methods fill the ECI position
and velocity arrays which downstream objects read.

.. list-table:: Parameters
   :widths: 22 26 52
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``satID``
     - int
     - Unique identifier for the satellite.
   * - ``orbit_n``
     - Orbit_specification or list, optional
     - Orbit specification (or a 6/7-element parameter list). Mutually exclusive with ``tle_file_path``.
   * - ``satName``
     - str, optional
     - Satellite name.
   * - ``tle_file_path``
     - str, optional
     - Path to a TLE file. Mutually exclusive with ``orbit_n``.
   * - ``satNo_in_tle``
     - int, optional
     - Index of the satellite in the TLE file (default 0; TLE init only).
   * - ``rx_antenna``
     - RxAntenna, optional
     - GNSS-R receiver antenna (not used for coverage).
   * - ``battery``, ``solarArray``
     - object, optional
     - Reserved for future power modelling.
   * - ``verbose``
     - bool, optional
     - Print initialisation progress.

.. list-table:: Key public methods
   :widths: 40 60
   :header-rows: 1

   * - Method
     - What it does
   * - ``propagate_orbit(startTime, duration, deltaT)``
     - Propagate over a window at a fixed step; fills ``positions_eci`` / ``velocities_eci``.
   * - ``propagate_orbit_for_times(times_skyfield)``
     - Propagate only at the given times (fast for sparse, visibility-limited sets).
   * - ``set_timescale_simulation(timeSkyfield)``
     - Attach the shared Skyfield timescale.
   * - ``calculate_sunlight_time()``
     - Mark which samples are sunlit (and the sunlit fraction).
   * - ``write_tle_file()`` / ``get_TLE()``
     - Generate / locate the satellite's TLE file.

.. autoclass:: object_definitions.Satellite_def.Satellite
   :no-members:

Satellite constellation
-----------------------

.. module:: object_definitions.SatConstellation_def

Summary
^^^^^^^

``SatelliteConstellation`` groups satellites and runs them on one clock. ``set_simulation_duration`` builds
the Skyfield timescale from a start time, a duration (hours), and a step (seconds), and pushes it down to
every satellite; ``propagate_constellation`` then propagates them all. The dictionary
``{"GPS": SatelliteConstellation, ...}`` returned by the GNSS loader is what the reflection and detection
code consumes.

.. list-table:: Parameters
   :widths: 22 26 52
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``satArray``
     - list, optional
     - List of :doc:`Satellite <satellites>` objects. Mutually exclusive with ``tle_file``.
   * - ``tle_file``
     - str, optional
     - Path to a TLE file with the satellite set. Mutually exclusive with ``satArray``.
   * - ``satConfigName``
     - str, optional
     - Name of the constellation.
   * - ``rx_antenna``
     - RxAntenna, optional
     - Receiver antenna shared by the satellites.
   * - ``verbose``
     - bool, optional
     - Print initialisation progress.

.. list-table:: Key public methods
   :widths: 40 60
   :header-rows: 1

   * - Method
     - What it does
   * - ``set_simulation_duration(startTime, duration, stepSize)``
     - Build the shared timescale and push it to all satellites.
   * - ``propagate_constellation(start_time, duration, step_size)``
     - Propagate every satellite over the window.
   * - ``add_satellite`` / ``remove_satellite``
     - Add or remove a satellite (keeps the orbit list in sync).
   * - ``get_constellation_size`` / ``get_constellation_orbits``
     - Read the satellite count and the orbit list.

.. autoclass:: object_definitions.SatConstellation_def.SatelliteConstellation
   :no-members:

Classical elements to a TLE
---------------------------

The receiver orbits in ``[ORBITS]`` are **osculating** classical elements -- an instantaneous two-body
state. A TLE instead stores **SGP4 mean** elements: quantities defined *through* SGP4's averaged theory,
not a snapshot of the Keplerian state. The two differ by the short- and long-period terms SGP4 folds in,
so the osculating elements cannot simply be copied into the TLE fields.
:meth:`~object_definitions.ReadWriteSatData.ReadWriteSatData.generate_tle_data_string` therefore *fits* the
mean elements, in three steps.

**1. Osculating state at the epoch.** The classical elements are turned into an inertial Cartesian state
:math:`(\mathbf{r}, \mathbf{v})`. In the perifocal frame, with semi-latus rectum :math:`p = a(1-e^2)` and
true anomaly :math:`\nu`,

.. math::
   :label: sat-perifocal

   r = \frac{p}{1 + e\cos\nu}, \qquad
   \mathbf{r}_\mathrm{pf} = r\begin{pmatrix}\cos\nu\\ \sin\nu\\ 0\end{pmatrix}, \qquad
   \mathbf{v}_\mathrm{pf} = \sqrt{\tfrac{\mu_\oplus}{p}}\begin{pmatrix}-\sin\nu\\ e+\cos\nu\\ 0\end{pmatrix},

rotated into the inertial frame by the 3-1-3 sequence of RAAN, inclination and argument of periapsis,
:math:`\mathbf{r} = R_3(\Omega)\,R_1(i)\,R_3(\omega)\,\mathbf{r}_\mathrm{pf}` (and likewise for
:math:`\mathbf{v}`). This is the **target** state the TLE must reproduce.

**2. Fit the mean elements.** SGP4 mean elements :math:`(a, e, i, \Omega, \omega, M)` are chosen so that
propagating them with SGP4 *back to the same epoch* returns the target state. A least-squares solver
(:func:`scipy.optimize.least_squares`) minimises the scaled position/velocity residual

.. math::
   :label: sat-residual

   \mathbf{R}(\mathbf{x}) =
   \left[\frac{\mathbf{r}_\mathrm{SGP4}(\mathbf{x}) - \mathbf{r}_\mathrm{target}}{100\ \mathrm{m}},\;
   \frac{\mathbf{v}_\mathrm{SGP4}(\mathbf{x}) - \mathbf{v}_\mathrm{target}}{0.1\ \mathrm{m/s}}\right],

where :math:`\mathbf{r}_\mathrm{SGP4}(\mathbf{x})` is the Skyfield/SGP4 epoch state built from the trial
mean elements. The solver variables :math:`\mathbf{x}` are equinoctial-style
(:math:`a,\ i,\ \Omega,\ e\cos\omega,\ e\sin\omega,\ \omega + M`) to stay well-conditioned as
:math:`e \to 0` (where :math:`\omega` becomes undefined); the osculating elements seed the initial guess,
with :math:`M` obtained from :math:`\nu` via the eccentric anomaly. Non-convergence emits a warning rather
than raising.

**3. Write the TLE.** The fitted mean elements are formatted onto the two 69-character lines: inclination,
RAAN, eccentricity (:math:`\times 10^7`), argument of periapsis and mean anomaly in degrees, and the mean
motion :math:`n = \sqrt{\mu_\oplus/a^3}` in revolutions per day, alongside the epoch, drag terms and
checksums. For a near-circular LEO the result reproduces the requested position to a few metres; the floor
comes from TLE field quantisation (angles stored to :math:`10^{-4}` deg), not the solver.

.. note::

   **Reading** a TLE runs the inverse anomaly step. A file stores the *mean* anomaly, so
   :meth:`~object_definitions.ReadWriteSatData.ReadWriteSatData.read_tle_data` and
   :class:`~object_definitions.Satellite_def.Satellite` convert it to the *true* anomaly
   (:meth:`~object_definitions.ReadWriteSatData.ReadWriteSatData.mean_to_true_anomaly_deg`) before storing
   it in an ``Orbit_specification`` -- so that container always holds a true anomaly regardless of source.
   A read TLE's mean elements are then held as if osculating; this is an accepted approximation because
   GNSS transmitters are propagated straight from their own TLE, never re-generated.

The ``ReadWriteSatData`` API (``generate_tle_data_string``, ``read_tle_data``,
``mean_to_true_anomaly_deg``) is documented on the :doc:`support` page.

Module Assumptions and Limitations
----------------------------------

- **SGP4 / TLE propagation.** Motion comes from Skyfield's SGP4 over a TLE. Auto-generated TLEs have no
  drag term (``B*``), so long-window propagation drifts from a real decaying orbit.
- **Osculating-to-mean conversion.** User elements describe an instantaneous two-body osculating state.
  TLE fields are SGP4 mean elements, so generation performs a nonlinear fit whose epoch state matches the
  requested position and velocity before TLE field quantisation. The numbers written into the TLE should
  not be interpreted as the original osculating elements.
- **Frame-conversion data.** Reflection geometry uses Astropy/PyMap3D ECI/ECEF transforms. Offline runs
  can use the installed ``astropy-iers-data`` table even after its predictive validity window; this avoids
  a hard failure but degrades Earth-orientation accuracy. Refresh that package/IERS data for precision
  studies.
- **Either elements or a TLE.** A ``Satellite`` cannot take both; a constellation cannot take both a
  satellite list and a TLE file.
- **Duration in hours, step in seconds.** ``set_simulation_duration`` mixes units by design; the step must
  be an integer number of seconds.
- **Epoch needed for propagation.** Without an epoch on the ``Orbit_specification`` the Skyfield object is
  not built.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_orbit.py``
     - Element storage and the mean-motion / epoch derivations.
   * - ``tests/test_satellite.py``
     - Construction from elements vs TLE (and the "not both" guard); propagation fills the ECI arrays.
   * - ``tests/test_satellite_constellation.py``
     - Timescale set-up, propagation, and add/remove bookkeeping.

User Guide
----------

Orbits come from the ``[ORBITS]`` INI section, one key per satellite
(``orbit1 = [inclination_deg, eccentricity, raan_deg, argPeriapsis_deg, trueAnomaly_deg, semiMajorAxis_m]``),
read in order until the next is missing. The GNSS transmitter constellations are loaded from the TLE files
listed in ``[GNSS CONSTELLATIONS].gnss_tle_files``.

.. code-block:: python

   from object_definitions.SatConstellation_def import SatelliteConstellation
   from object_definitions.Satellite_def import Satellite

   sats = [Satellite(i, orbit_n=orbit_list[i]) for i in range(n)]
   leo = SatelliteConstellation(satArray=sats, satConfigName="LEO")
   leo.set_simulation_duration(start_time, duration_hours, step_seconds)
   leo.propagate_constellation(start_time, duration_seconds, step_seconds)

Two-Line Element Sets (TLEs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **Two-Line Element (TLE)** is the standard `NORAD <https://www.norad.mil/>`_ text format that encodes a satellite's mean
Keplerian elements, epoch, and drag term on two 69-character lines. Every satellite is propagated from a
TLE with SGP4 (via Skyfield), so both transmitter (GNSS) and receiver (GNSS-R) satellites run from TLEs:

- **GNSS transmitters** are loaded directly from the TLE files listed in
  ``[GNSS CONSTELLATIONS].gnss_tle_files`` and stored under ``DATA/TLE_EPHEMIRIS/``. One file becomes one
  ``SatelliteConstellation``. The constellation is identified from the file name, which must contain one of
  ``GPS``, ``GLONASS``, ``GALILEO``, or ``BEIDOU`` (case-insensitive); a name that matches none raises.
- **GNSS-R receivers** are built from the osculating ``[ORBITS]`` elements. The generator fits SGP4 mean
  elements at the configured epoch and writes a per-satellite TLE under
  ``DATA/TLE_EPHEMIRIS/autogenerated/`` (git-ignored). Re-generating the same satellite overwrites that
  satellite's deterministic autogenerated filename.

For example, the GPS satellite PRN 13 -- a name line followed by the two element lines, with the epoch
field (``25135.33377744``) in the ``YYDDD.DDDDDDDD`` form noted above:

.. code-block:: text

   GPS BIIR-2  (PRN 13)
   1 24876U 97035A   25135.33377744  .00000044  00000+0  00000+0 0  9991
   2 24876  55.7964 114.1388 0089709  53.8017 306.9996  2.00562545203970

TLE epochs age, and SGP4 accuracy degrades as the propagation window moves away from the epoch, so the
GNSS files should be refreshed periodically.

The GNSS TLE files come from `CelesTrak <https://celestrak.org/NORAD/elements/>`_.

Each TLE download shall be saved under ``DATA/TLE_EPHEMIRIS/`` with a name that contains the constellation keyword (for
example ``GPS_operational.tle``), then list it in ``[GNSS CONSTELLATIONS].gnss_tle_files``.

References
----------

- Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Microcosm Press, 2013.
- Rhodes, B., *Skyfield: High precision research-grade positions for planets and Earth satellites*, 2019.
