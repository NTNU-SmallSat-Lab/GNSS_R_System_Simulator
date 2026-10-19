GNSS-R System Simulator
=======================

A preliminary-design tool for GNSS-Reflectometry (GNSS-R) LEO satellite missions. Two analysis modes
share the same satellite / constellation core:

- **Area coverage** -- how often each cell of a gridded region is visited by reflections in a time
  window: by specular-point tracks, or by near-backscatter swaths around the backscatter reflection point.
- **Moving-target detection** -- whether and how often a moving (or stationary) ocean target is covered
  by the reflections of the LEO x GNSS links, and what the link budget predicts for it, in either
  ``specular`` (forward-scatter) or ``backscatter`` mode.

New users start with :doc:`getting_started`, which installs the simulator and runs both analyses. The
:doc:`howto` collects the common scenario changes, and :doc:`outputs` explains the files a run writes.

The module reference mirrors the ``SRC/`` source layout: each module has its own page with an executive
summary, its public interface, a detailed physics / math description, assumptions and limitations, the
verification evidence that constrains it, and a short user guide.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   running
   howto
   outputs
   monte_carlo
   Documentation/configuration

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   architecture
   modes/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   Documentation/index

.. toctree::
   :maxdepth: 1
   :caption: Assessment

   scientific_assessment
   validation

.. toctree::
   :maxdepth: 1
   :caption: Project

   development
   citing
