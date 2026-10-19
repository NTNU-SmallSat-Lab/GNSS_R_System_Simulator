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
- ``mc_event_plots`` -- event-level power records and figures of the Monte Carlo harness (documented
  in :doc:`../../monte_carlo`).

.. toctree::
   :maxdepth: 1
   :caption: Module pages

   ocean_physics
   link_budget
   delay_doppler
   helpers
