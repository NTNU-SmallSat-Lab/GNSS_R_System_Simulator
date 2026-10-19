Citing, licence and data
========================

Citing the simulator
--------------------

Each GitHub release of the simulator is archived on Zenodo with its own DOI. Cite the release you used,
so that results can be traced to the exact code version. The citation metadata (title, authors with
ORCID, version, licence) is kept in ``CITATION.cff`` in the repository root; GitHub's *Cite this
repository* button renders it as APA or BibTeX.

Authors: Oliver Hasler and Corrado Chiatante, Norwegian University of Science and Technology (NTNU).

Licence
-------

The code is released under the MIT License, copyright NTNU, Department of Engineering Cybernetics (see
``LICENSE``). The data in ``DATA/`` keeps the terms of its source.

Data sources
------------

.. list-table::
   :widths: 30 26 44
   :header-rows: 1

   * - Data
     - Location
     - Source and terms
   * - Coastline, ocean and marine-area polygons
     - ``DATA/world_costline_data/``
     - `Natural Earth <https://www.naturalearthdata.com/>`_ 1:10m physical vectors; public domain.
   * - Planetary ephemeris DE421
     - ``DATA/SKYFIELD_data/de421.bsp``
     - NASA Jet Propulsion Laboratory; read through Skyfield.
   * - GNSS two-line elements
     - ``DATA/TLE_EPHEMIRIS/``
     - `CelesTrak <https://celestrak.org/>`_. Refresh them as their epochs age out (see
       :doc:`howto`).
   * - Norwegian exclusive economic zone (not bundled)
     - ``DATA/eez/eez_norge.shp``
     - Flanders Marine Institute (2023). Maritime Boundaries Geodatabase: Maritime Boundaries and
       Exclusive Economic Zones (200NM), version 12. `marineregions.org <https://www.marineregions.org/>`_,
       https://doi.org/10.14284/632. CC BY 4.0. Download it as described in :doc:`getting_started`.

Figures produced with the EEZ (the area-coverage maps of the default scenario) show Marine Regions data
and need the Flanders Marine Institute citation above when published.
