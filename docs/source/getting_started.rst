Getting started
===============

This page installs the simulator, downloads the one data set that is not bundled, and runs both
analyses with the shipped scenario. It takes about 15 minutes.

Requirements
------------

- Python 3.12 (the simulator is validated on 3.12.x; newer versions may work but scientific binary
  wheels can lag).
- Windows, Linux or macOS. All dependencies, including ``cartopy`` and ``geopandas``, install from PyPI.
- About 1.5 GB of disk space for the code, the dependencies and the generated files.

Install
-------

Clone the repository (or unpack a release archive) and create a virtual environment in it:

.. code-block:: powershell

   # Windows (PowerShell)
   git clone https://github.com/NTNU-SmallSat-Lab/GNSS_R_System_Simulator.git
   cd GNSS_R_System_Simulator
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[dev,docs]"

.. code-block:: bash

   # Linux / macOS
   git clone https://github.com/NTNU-SmallSat-Lab/GNSS_R_System_Simulator.git
   cd GNSS_R_System_Simulator
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,docs]"

``pip install`` only installs the dependencies; the simulator itself runs from the repository. The
``dev`` extra adds the test and lint tools, the ``docs`` extra the documentation toolchain; both are
optional.

Download the region data
------------------------

Area coverage needs the polygon of the Norwegian exclusive economic zone (EEZ), which is not
redistributed with the code (see :doc:`citing`). Download it from the Marine Regions web service
(Marine Regions ID 5686) to the path that ``Configuration.ini`` expects, from the repository root:

.. code-block:: bash

   python -c "import os, geopandas as gpd; os.makedirs('DATA/eez', exist_ok=True); gpd.read_file('https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=MarineRegions:eez&cql_filter=mrgid=5686&outputFormat=application/json').to_file('DATA/eez/eez_norge.shp')"

This writes ``DATA/eez/eez_norge.shp`` and its sidecar files. Moving-target runs do not need it.

Check the installation
----------------------

.. code-block:: bash

   pytest

All tests should pass within a few seconds to a minute.

.. important::

   Always start the simulator **from the repository root**. Every data and output path is resolved from
   the working directory, so a script started elsewhere cannot find ``DATA/`` or ``OUTPUT/``.

First run: area coverage
------------------------

The shipped ``Configuration.ini`` describes two LEO receivers at about 920 km altitude (inclination
99.1 deg, ascending nodes 90 deg apart), GPS and Galileo transmitters, and a 6 h window starting
2 April 2025 12:00 UTC. The region is the Norwegian EEZ on a 30 km grid, and the reflection mode is
``backscatter``.

.. code-block:: bash

   python SRC/main_simulation_areaCoverage.py

The run takes about two minutes; the first run also builds the coverage grid and the antenna patterns.
It ends with the combined statistics:

.. code-block:: text

   Combined Statistics for All Selected Constellations
   --------------------------------------------------
   Combined coverage percentage: 32.46%
   Total areas covered: 421 out of 1297 grid cells

The maps are in ``OUTPUT/``: ``Coverage_grid_all_configurations_backscatter.png`` shows how often each
cell was visited, and ``Orbits.html`` opens an interactive 3D view of the orbits. :doc:`outputs` lists
every file.

To see specular coverage instead, set ``reflection_type = "specular"`` in ``[REFLECTION]`` and run
again.

First run: moving-target detection
----------------------------------

The shipped ship route (North Sea to the Barents Sea, about four days of sailing) is a long run. For a
first run, sail 20 hours across the open North Sea instead:

.. code-block:: ini

   [MOVING TARGET]
   speed = 15
   start_pos = [56.0, 3.0]
   end_pos = [61.0, 3.0]

.. code-block:: bash

   python SRC/main_simulation_movingTarget.py

The run takes about three minutes. It plans the route around land, propagates the satellites only while
they can see the ship, and evaluates the link budget for every footprint overlap and every band in
``[ANTENNA].bands``. In ``OUTPUT/`` you find the route (``ship_route_DarkShip_001.png``), one
footprint-overlap map per constellation and band, and, in backscatter mode, the link-budget breakdown of
ship power, ocean clutter and noise (``Link_budget_breakdown_backscatter_*``). The console prints the
event counts and the signal-to-clutter and SINR ranges per constellation.

Short routes of a few hours often have no footprint overlap at all; the link-budget figures are then
skipped with a warning.

Next steps
----------

- :doc:`howto` -- change the region, orbits, constellations and bands, ship, sea state, antenna and
  receiver settings.
- :doc:`Documentation/configuration` -- every INI key.
- :doc:`modes/index` -- what specular and backscatter mode compute and how a ship shows up in each.
- :doc:`monte_carlo` -- randomised feasibility studies over many routes and receiver settings.
- :doc:`validation` -- what the models have been checked against, and what they are not yet fit for.
