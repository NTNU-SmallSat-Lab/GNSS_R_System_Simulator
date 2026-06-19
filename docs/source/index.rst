GNSS-R System Simulator
=======================

A preliminary-design tool for GNSS-Reflectometry (GNSS-R) LEO satellite missions. Two analysis modes
share the same satellite / constellation core:

- **Area coverage** -- how often a gridded region is covered by specular reflections in a given window.
- **Moving-target detection** -- whether and how often a moving (or stationary) ocean target gets
  reflections from the LEO x GNSS link, in either ``specular`` (forward-scatter) or ``backscatter`` mode.

This documentation mirrors the ``SRC/`` source layout: each module has its own page with an executive
summary, its public interface, a detailed physics / math description, assumptions and limitations,
the validation that pins it, and a short user guide.

.. note::

   The documentation is being rolled out module by module. Sections that are not yet written are
   tracked in the project ``TODO.md`` and the documentation rollout plan.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Documentation/index
