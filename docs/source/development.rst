Development
===========

This page is for working on the simulator itself: the repository layout, the test suite, the code
conventions, the documentation build, and the places to extend.

Repository layout
-----------------

.. code-block:: text

   SRC/
     __init__.py                       INI / YAML parsing (initialize) and the band plan
     main_simulation_areaCoverage.py   area-coverage entry point
     main_simulation_movingTarget.py   moving-target entry point
     run_monte_carlo.py                Monte Carlo feasibility harness
     functions/                        pure-function physics and helpers
     object_definitions/               satellites, reflections, targets, drivers, antenna, constants
   tests/                              pytest suite; golden references in tests/golden/
   DATA/                               TLEs, coastline/ocean shapefiles, ephemeris, antenna helpers
   docs/source/                        this documentation
   ARCHITECTURE/                       UML sources and figure scripts
   Configuration.ini                   scenario (both analysis modes)
   simulator_settings.yaml             engine / numerical settings
   pyproject.toml                      dependencies and tool configuration

The simulator is not an installed package. The entry scripts live in ``SRC/`` so that ``SRC/`` is on
``sys.path`` and modules import as ``functions.link_budget`` or ``object_definitions.Reflection_def``.
``pip install`` only installs the dependencies. All scripts must be started from the repository root,
because ``Constants`` resolves every data and output path from the working directory.

Tests
-----

Install the development tools with ``pip install -e ".[dev]"``, then run from the repository root:

.. code-block:: bash

   pytest                             # full suite
   pytest tests/test_reflection.py    # one module
   pytest -m golden                   # only the golden-file regression tests

``pyproject.toml`` puts ``SRC/`` and ``tests/`` on the test path, so tests import the modules directly.
The suite covers the pure functions (rotations, ocean and ship physics, delay-Doppler cell, link
budget), construction and validation of every class, the INI parser and band plan, and end-to-end
behaviour of the coverage and moving-target drivers on small fixtures.

Two golden-file tests pin complete numerical pipelines against stored references:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test
     - Reference and regeneration
   * - ``tests/test_antenna_gain_golden.py``
     - ``tests/golden/reference_gains.npz``; ``python tests/golden/regenerate_gains.py``
   * - ``tests/test_link_budget_golden.py``
     - ``tests/golden/reference_link_budget.npz``; ``python tests/golden/regenerate_link_budget.py``

Regenerate a reference only when a model change is intended, and commit the new file together with that
change.

Code quality
------------

.. code-block:: bash

   black .          # formatting, line length 160
   ruff check .     # linting
   mypy SRC         # type checking (relaxed: untyped definitions allowed)

All three are configured in ``pyproject.toml``. Scientific variable names in mixed or upper case are
intentional; the corresponding naming rules are disabled.

Conventions
-----------

- **Strict configuration.** Every tunable lives in ``Configuration.ini`` or ``simulator_settings.yaml``.
  A missing required key raises; there are no hidden defaults in the code.
- **Units in names.** Quantities carry their unit as a suffix (``_m``, ``_s``, ``_hz``, ``_db``,
  ``_dbw``, ``_deg``, ``_m2``). Internal computation is SI; angles in the INI are degrees.
- **Time.** All times are UTC. ``functions.functions.ensure_utc_aware`` is the canonical way to coerce
  a ``datetime``.
- **Coordinate order.** ``[MOVING TARGET]`` positions are ``[latitude, longitude]``; bounding boxes are
  ``[min_lon, max_lon, min_lat, max_lat]``. Check the docstring of the consumer before passing a tuple.
- **ASCII source files.** Keep ``.py`` files ASCII-only and spell out mathematical symbols in comments
  and docstrings; some editors corrupt non-ASCII characters on save.

Documentation
-------------

The documentation is written in reStructuredText under ``docs/source/``; the API sections are pulled
from the NumPy-style docstrings by ``sphinx.ext.autodoc``. Build it locally from the repository root:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. The GitHub Actions workflow
``.github/workflows/docs.yml`` runs the same build on every push to ``main`` and publishes the result to
GitHub Pages.

The UML class diagrams in :doc:`architecture` are generated from the PlantUML sources in
``ARCHITECTURE/`` by ``python ARCHITECTURE/make_uml_plantuml.py``, which renders through the public
PlantUML server and writes the SVGs into ``docs/source/images/``.

Extending the simulator
-----------------------

**A new scattering target.** Subclass :class:`object_definitions.Target_def.Target` and implement
``compute_rcs(geometry, wavelength_m, illuminated_area_m2=None)``, returning a
:class:`~object_definitions.Target_def.TargetRcs` with the per-sample RCS in m\ :sup:`2` and any
diagnostics in ``info``. The bistatic radar equation accepts any target through the ``target`` argument
of :func:`functions.link_budget.compute_reflection_link_budget`. The moving-target detector builds its
targets in ``MovingTargetDetection._compute_ocean_pr_at_event`` (``OceanTarget``) and
``MovingTargetDetection._compute_ship_pr_at_event`` (``ShipTarget``); that is where a new model is
swapped in. See :doc:`Documentation/object_definitions/ship_target` for the interface.

**A new GNSS band.** Add the carrier frequency to ``Constants.BANDS_CENTER_FREQUENCIES_Hz`` and the
band to ``Constants.GNSS_EIRP_DBW`` and ``Constants.GNSS_CHIP_RATE_HZ`` under its constellation. Band
names must be unique across constellations, because the owning constellation is looked up from the band
name. See :doc:`Documentation/object_definitions/support`.

**A new GNSS constellation file.** The TLE file name must contain ``GPS``, ``GLONASS``, ``GALILEO`` or
``BEIDOU`` (any case); ``initialize`` matches the constellation from the name and raises for a file that
matches none.
