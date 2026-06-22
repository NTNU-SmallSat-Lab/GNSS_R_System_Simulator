Cross-cutting functions
=======================

Pure-function geometry, propagation, link-budget, and ocean-physics helpers shared by both analysis
modes:

- ``ocean_physics`` -- seawater permittivity, Fresnel / circular reflectivity, Cox-Munk slopes, Bragg.
- ``link_budget`` -- bistatic geometry, the radar equation, and the link-budget settings.
- ``delay_doppler`` -- the first-order delay-Doppler resolution-cell area.
- ``rotations`` -- VNC frame construction and ECI-to-antenna rotations.
- ``functions`` -- the shared geometry / propagation / IO helper module.
- ``ship_physics`` -- ship-target scattering helpers.

.. toctree::
   :maxdepth: 1
   :caption: Module pages

   ocean_physics
   link_budget
   delay_doppler
   helpers

.. note::

   Remaining module pages are added per the documentation rollout.
