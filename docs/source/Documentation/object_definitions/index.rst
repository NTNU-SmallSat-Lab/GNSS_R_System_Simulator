Simulation objects (``SRC/object_definitions``)
===============================================

The classes that make up a simulation run:

- **Geometry / propagation** -- ``Satellite``, ``SatConstellation``, ``Orbit``, ``Rx_antenna``,
  ``AntennaPatternGenerator``, ``ReadWriteSatData``.
- **Reflection geometry** -- ``Reflection`` (specular) and ``BackScattPoint`` (bistatic reflection point).
- **Targets** -- ``Target`` (abstract base), ``OceanTarget``, ``ShipTarget``.
- **Analysis drivers** -- ``Coverage`` (area coverage) and ``MovingTargetDetection`` (target mode).
- **Support** -- ``Constants``, ``ShipTrajectory``, ``Static_classes``.

.. toctree::
   :maxdepth: 1
   :caption: Module pages

   ship_trajectory

.. note::

   Remaining module pages are added per the documentation rollout.
