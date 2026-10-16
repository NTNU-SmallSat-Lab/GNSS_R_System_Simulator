Cross-cutting helpers (rotations, functions, ship_physics)
==========================================================

Three modules of shared helper functions: frame rotations, a general geometry/propagation/IO toolbox, and
the ship-RCS shapes.

Rotations
---------

Summary
^^^^^^^

``rotations`` builds the chain of frames that turns an ECI direction into the antenna frame, so the
antenna model can look up gain. The chain is **ECI -> VNC -> body -> antenna**:

- **VNC** (Velocity-Normal-Cross-track) is the orbit frame, built from the satellite's position and
  velocity.
- the satellite **attitude** quaternion rotates VNC into the body frame;
- a fixed **mounting** rotation takes the body frame to the antenna frame.

A direction is transformed as ``v_antenna = R_eci_to_antenna @ v_eci``; the reverse is the transpose.

.. automodule:: functions.rotations
   :members:

Ship physics
------------

Summary
^^^^^^^

``ship_physics`` provides two idealised radar shapes used to model a ship as a hard target: a scalar
physical-optics flat plate (a hull side or deck) and an engineering surrogate for a 90-degree dihedral.

The flat plate peaks when its normal bisects the transmitter and receiver directions, with the
cross-section

.. math::
   :label: ship-plate

   \sigma_\mathrm{plate} = \frac{4\pi A^2}{\lambda^2}\cos^2\left(\alpha\right)\;
   \operatorname{sinc}^2(u_1)\,\operatorname{sinc}^2(u_2),

for plate area :math:`A`, the angle :math:`\alpha` between the plate normal and the bistatic bisector, and
the edge form-factor arguments :math:`u_i`. The dihedral peaks when the double bounce preserves the
edge-parallel direction component, i.e. at retroreflection only for directions perpendicular to the edge,

.. math::
   :label: ship-dihedral

   \sigma_\mathrm{dihedral} = \frac{8\pi (LH)^2}{\lambda^2}\,\operatorname{sinc}^2(u_\mathrm{edge}),

for edge length :math:`L` and leg height :math:`H`. Both are zero when the shape faces away (back-face /
wedge visibility).

.. automodule:: functions.ship_physics
   :members:

General toolbox
---------------

Summary
^^^^^^^

``functions.functions`` is the project's shared toolbox (imported as ``fn``). It collects small
geometry, propagation, and input/output helpers used across the simulator. The main groups:

.. list-table:: What's inside
   :widths: 30 56 14
   :header-rows: 1

   * - Group
     - Functions
     - Returns
   * - Time / coordinates
     - ``ensure_utc_aware``, ``to_utc_datetime_array``, ``ecef_to_lla``, ``geodetic_to_ecef``
     - --
   * - Vector geometry
     - ``angle_between_vectors``, ``skew_matrix``, ``rotation_matrix_from_vectors``
     - deg, --
   * - Distance on the sphere
     - ``delta_lon_from_distance``, ``delta_lat_from_distance``
     - deg
   * - Constellations
     - ``load_gnss_constellation_dict``, ``propagate_all_orbits``, ``color_gnss_const``,
       ``normalize_gnss_constellation_arg``
     - --
   * - Visibility optimisation
     - ``compute_visible_times_and_resample_ship``, ``propagate_only_visible_satellites``
     - --
   * - Footprint
     - ``compute_footprint``
     - --

These are utilities rather than physics, so they are summarised here rather than each given a full
section; the docstrings in the source carry the per-function detail.

Module Assumptions and Limitations
----------------------------------

- **VNC needs a valid state.** ``build_vnc_frame`` returns ``None`` for a degenerate state (zero velocity
  or position, or position parallel to velocity); callers must handle that.
- **Nominal boresight is not exact nadir for eccentric motion.** The VNC ``C`` axis is the component of
  position perpendicular to velocity. With the configured mounting, antenna :math:`+z` points along
  :math:`-C`; this equals geocentric nadir for a circular orbit, but differs slightly when radial velocity
  is nonzero.
- **Ship shapes are idealised.** ``ship_physics`` uses perfect conducting plates and a scalar dihedral
  peak with simple visibility. It omits full polarimetric angular response, component interference and
  rough-sea coupling; see :doc:`../object_definitions/ship_target`.
- **Local ellipsoid approximation.** ``delta_lon/lat_from_distance`` use WGS-84 local radii of curvature.
  They convert a local east/north displacement, not an arbitrary long geodesic.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_rotations.py``
     - Quaternion-to-matrix, the VNC frame (orthonormal, degenerate cases), and the composed
       ECI-to-antenna rotation.
   * - ``tests/test_functions.py``
     - The geometry / time / distance helpers.
   * - ``tests/test_ship_physics.py``
     - Plate peak and sinc nulls, back-face culling, dihedral retroreflection peak, edge nulls, and wedge
       visibility. These are formula checks, not measured-ship validation.

``ship_physics`` is also exercised end-to-end through ``tests/test_ship_target.py``.

References
----------

- Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013 (orbit frames).
- Knott, E., Schaeffer, J. & Tulley, M., *Radar Cross Section*, 2nd ed., SciTech, 2004 (plate / dihedral RCS).
