Backscatter point / BRP
=======================

.. module:: object_definitions.BackScattPoint_def

Summary
-------

``BRP`` finds the surface point used in **backscatter mode**. There, the transmitter, the surface, and the
receiver are almost in a straight line (the signal comes nearly straight back, a retroreflection). So the
reflection point is simply where the straight line through the transmitter and receiver meets the Earth's
surface on the receiver's side. ``BRP`` solves that point for every time sample, checks the geometry, and
-- with a link-budget configuration -- works out the received power.

It is the backscatter-mode twin of :doc:`reflection` (which handles the forward specular reflection point). The
two share the same stored fields, so the rest of the simulator treats them the same way.

Interface
---------

.. list-table:: Inputs / outputs
   :widths: 22 28 38 12
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs** (constructor)
     - ``gps_index`` / ``leo_index``
     - Transmitting (GNSS) and receiving (LEO) satellite identifiers.
     - --
   * -
     - ``max_central_angle_deg``
     - Coarse Tx-Rx geometric gate.
     - deg
   * -
     - ``max_incidence_angle_deg``
     - Rejects too-grazing geometry at the BRP.
     - deg
   * -
     - ``earth_model``
     - "spherical" / "elliptical".
     - --
   * - **Inputs** (``compute_brp_coordinates``)
     - ``GNSS_state`` / ``Sat_state``
     - Propagated :class:`~object_definitions.Satellite_def.Satellite` states.
     - m, m/s
   * -
     - ``times`` / ``tx_eirp_dbw`` / ``antenna`` / ``frequency_hz`` / ``link_budget_settings``
     - Timestamps, EIRP, antenna, frequency, and Rx-system settings.
     - --
   * - **Outputs** (attributes)
     - ``refl_ECI_vec`` / ``refl_ECEF_vec``
     - BRP positions in ECI and ECEF ``(n, 3)``.
     - m
   * -
     - ``refl_LL_vec``
     - BRP positions as ``(latitude, longitude)``.
     - deg
   * -
     - ``refl_ang_vec``
     - Incidence angle at the BRP.
     - deg
   * -
     - ``link_budget`` / ``power``
     - Link-budget dict and received power.
     - --, W

Detailed Module Description
---------------------------

The retroreflection line
^^^^^^^^^^^^^^^^^^^^^^^^^

Write the line through the transmitter :math:`T` and receiver :math:`R` as

.. math::
   :label: brp-line

   \mathbf{P}(t) = T + t\,\hat{u}, \qquad \hat{u} = \frac{R - T}{\lVert R - T \rVert}.

Moving along :math:`\hat{u}` with :math:`t > 0` goes from the transmitter, past the receiver, toward
Earth. The BRP is the first surface point on that side.

Where the line meets the ellipsoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Putting :math:`\mathbf{P}(t)` into the WGS-84 ellipsoid equation
:math:`(x^2 + y^2)/a^2 + z^2/b^2 = 1` gives a quadratic in :math:`t`,

.. math::
   :label: brp-quadratic

   A\,t^2 + 2B\,t + C = 0,

with, for :math:`T = (x_0, y_0, z_0)` and :math:`\hat{u} = (u_x, u_y, u_z)`,

.. math::
   :label: brp-coeffs

   \begin{aligned}
   A &= b^2(u_x^2 + u_y^2) + a^2 u_z^2,\\
   B &= b^2(x_0 u_x + y_0 u_y) + a^2 z_0 u_z,\\
   C &= b^2(x_0^2 + y_0^2 - a^2) + a^2 z_0^2.
   \end{aligned}

The selected root must satisfy :math:`t>\lVert R-T\rVert`, so it is beyond the receiver along the
Tx-to-Rx ray; this surface point is the BRP, denoted :math:`S`. Merely choosing a positive root can
incorrectly select an intersection between the transmitter and receiver. This condition doubles as the
backscatter visibility check -- a geometry whose Tx-Rx line misses the ellipsoid, or whose intersection
does not lie beyond the receiver, is rejected. It is the counterpart of the chord line-of-sight test used
for the specular point (:doc:`reflection`), which the BRP path does not reuse.

Incidence angle and gates
^^^^^^^^^^^^^^^^^^^^^^^^^^

The incidence angle is measured from the local ellipsoid normal :math:`\hat{n}(S) \propto (S_x/a^2,
S_y/a^2, S_z/b^2)`,

.. math::
   :label: brp-incidence

   \cos\left(\theta_i\right) = \frac{(R - S)\cdot\hat{n}(S)}{\lVert R - S\rVert},

and the sample is kept only if :math:`\theta_i` is below ``max_incidence_angle_deg``. Two cheaper gates
run first: the Tx-Rx central angle must be below ``max_central_angle_deg`` (the same central-angle gate as
the specular path), and -- when an antenna is given -- the look direction must fall inside the antenna's
steering cone (the retroreflector look direction is parallel to :math:`R - T`, so this can be checked
before solving the quadratic).

Module Assumptions and Limitations
----------------------------------

- **Backscatter (retroreflection) geometry.** The BRP assumes Tx, surface, and Rx are collinear. This is
  the right point for the ocean/ship backscatter mode, not for the forward specular reflection point (use
  :doc:`reflection` for that).
- **Selected Earth model.** ``earth_model="elliptical"`` uses the WGS-84 ellipsoid and its geodetic
  normal. ``earth_model="spherical"`` uses the configured mean Earth radius and a radial normal.
- **Coarse central-angle gate.** ``max_central_angle_deg`` is a quick reject, not the physical horizon.
- **One Tx-Rx pair per object**, like :class:`~object_definitions.Reflection_def.Reflection`.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test group
     - What it checks
   * - ``TestEllipsoidIntersection``
     - The BRP lies on the Tx-Rx line and on the ellipsoid; the sub-point case is correct; too-far-apart
       and Earth-missing geometries are rejected; the selected root is beyond the receiver; identical Tx
       and Rx is handled; spherical and ellipsoidal modes use their requested surface.
   * - ``TestAntennaFovPreGate``
     - A nadir BRP passes the field-of-view gate, a far-off-nadir one is rejected, and no antenna skips the
       gate.

User Guide
----------

``BRP`` is built internally when ``[REFLECTION].reflection_type = "backscatter"``. The two gate angles come
from ``[REFLECTION]``:

.. list-table:: ``[REFLECTION]`` keys
   :widths: 36 50 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``earth_model``
     - ``"spherical"`` (mean-radius sphere) or ``"elliptical"`` (WGS-84 ellipsoid). Required; missing
       or invalid values raise.
     - --
   * - ``reflection_type``
     - ``"backscatter"`` selects this BRP path (vs ``"specular"``).
     - --
   * - ``max_central_angle_deg``
     - The constructor's ``max_central_angle_deg``.
     - deg
   * - ``max_incidence_angle_deg``
     - The constructor's ``max_incidence_angle_deg`` (required for backscatter).
     - deg

Public interface
^^^^^^^^^^^^^^^^

.. autoclass:: object_definitions.BackScattPoint_def.BRP
   :members:

.. autofunction:: object_definitions.BackScattPoint_def.calculate_ellipsoid_intersection_t

References
----------

- Gleason, S. & Gebre-Egziabher, D. (eds.), *GNSS Applications and Methods*, Artech House, 2009.
