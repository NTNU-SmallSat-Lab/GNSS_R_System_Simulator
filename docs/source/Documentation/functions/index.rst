Cross-cutting functions (``SRC/functions``)
===========================================

Pure-function geometry, propagation, link-budget, and ocean-physics helpers shared by both analysis
modes:

- ``ocean_physics`` -- seawater permittivity, Fresnel / circular reflectivity, Cox-Munk slopes, Bragg.
- ``link_budget`` -- bistatic geometry, the radar equation, and the link-budget settings.
- ``delay_doppler`` -- the delay-Doppler resolution-cell (DDM pixel) area.
- ``rotations`` -- VNC frame construction and ECI-to-antenna rotations.
- ``functions`` -- the shared geometry / propagation / IO helper module.
- ``ship_physics`` -- ship-target scattering helpers.

.. note::

   Module pages are added per the documentation rollout (pilot: ``link_budget``).
