GNSS-R System Simulator
=======================

A preliminary-design tool for GNSS-Reflectometry (GNSS-R) LEO satellite missions. Two analysis modes
share the same satellite / constellation core:

- **Area coverage** -- how many contiguous specular-reflection track visits cross each grid cell.
- **Moving-target detection** -- whether and how often a moving (or stationary) ocean target gets
  reflections from the LEO x GNSS link, in either ``specular`` (forward-scatter) or ``backscatter`` mode.

This documentation mirrors the ``SRC/`` source layout: each module has its own page with an executive
summary, its public interface, a detailed physics / math description, assumptions and limitations,
the verification evidence that constrains it, and a short user guide.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   running
   architecture
   modes/index
   Documentation/index
   scientific_assessment
   validation
