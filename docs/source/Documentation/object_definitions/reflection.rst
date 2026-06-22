Specular reflection
===================

.. module:: object_definitions.Reflection_def

Summary
-------

``Reflection`` finds the **specular reflection point** of the GNSS signal on the Earth's surface
for one transmitting GNSS satellite and one receiving LEO satellite, at every time sample in the
propagated window. It is the geometry engine of the forward-scatter path: for each sample it solves where
the bounce lands, the incidence angle there, and -- with a link-budget configuration -- the received
power. :class:`~object_definitions.Coverage_def.Coverage` builds one ``Reflection`` per
(LEO, GNSS) pair to accumulate revisit counts; the moving-target detector uses the same geometry per
Tx/Rx pair.

The specular point is the surface point :math:`S` for which the incoming ray :math:`T\!\to\!S` and the
outgoing ray :math:`S\!\to\!R` make equal angles with the local surface normal (the law of reflection).
Two solvers implement this:

- **spherical** -- a closed-form solution: rotate into the Tx-Rx orbital plane and solve a quartic on the
  circle;
- **elliptical** -- solve the two tangent components of Fermat's stationary-path condition on a
  two-angle WGS-84 surface parameterisation, then verify the specular condition.

Interface
---------

.. list-table:: Inputs / outputs
   :widths: 22 26 40 12
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs** (constructor)
     - ``tx_satID``
     - Transmitting (GNSS) satellite identifier.
     - --
   * -
     - ``rx_satID``
     - Receiving (LEO) satellite identifier.
     - --
   * -
     - ``earth_model``
     - "spherical" / "elliptical" surface model.
     - --
   * -
     - ``max_central_angle_deg``
     - Coarse Tx-Rx geometric gate (required, keyword-only).
     - deg
   * -
     - ``gnss_const``
     - Constellation label (optional).
     - --
   * - **Inputs** (``compute_reflection_coordinates``)
     - ``GNSS_state`` / ``Sat_state``
     - Propagated :class:`~object_definitions.Satellite_def.Satellite` states (ECI pos / vel).
     - m, m/s
   * -
     - ``times``
     - UTC timestamps aligned with the states.
     - --
   * -
     - ``tx_eirp_dbw``
     - Transmit EIRP for this (constellation, band).
     - dBW
   * -
     - ``antenna`` / ``frequency_hz`` / ``link_budget_settings`` / ``q_sat`` / ``compute_link_budget``
     - Optional link-budget arguments.
     - --
   * - **Outputs** (attributes)
     - ``refl_ECI_vec`` / ``refl_ECEF_vec``
     - Specular points in ECI ``(n, 3)`` (and ECEF, elliptical only).
     - m
   * -
     - ``refl_LL_vec``
     - Specular points as ``(latitude, longitude)`` ``(n, 2)``.
     - deg
   * -
     - ``refl_ang_vec``
     - Incidence angle ``(n,)``.
     - deg
   * -
     - ``angle_diagnostics``
     - Solver diagnostics ``[theta_in, theta_out, central_angle]``.
     - deg
   * -
     - ``link_budget`` / ``power``
     - Link-budget dict and received power.
     - --, W

Detailed Module Description
---------------------------

Central-angle gate and line of sight
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before any solve, two cheap filters reject samples that cannot produce a reflection. The Tx-Rx geocentric
central angle

.. math::
   :label: refl-central

   \theta_c = \arccos\!\left(\frac{\mathbf{r}_{tx}\cdot\mathbf{r}_{rx}}
   {\lVert\mathbf{r}_{tx}\rVert\,\lVert\mathbf{r}_{rx}\rVert}\right)

must be below ``max_central_angle_deg``. Line of sight is tested analytically on the finite chord between
the two satellites,

.. math::
   :label: refl-los

   \mathbf{P}(t) = \mathbf{r}_1 + t\,(\mathbf{r}_2 - \mathbf{r}_1), \quad t\in[0,1],

by projecting the origin onto the chord, clamping the parameter to :math:`[0,1]`, and requiring that
closest point to clear the Earth. For WGS-84, each coordinate is first scaled by :math:`(a,a,b)`, mapping
the ellipsoid to a unit sphere. This catches grazing intersections without sampling error and avoids the
overly conservative equatorial-sphere approximation near the poles.

Law of reflection
^^^^^^^^^^^^^^^^^

With the incoming and outgoing unit propagation directions and the outward surface normal at :math:`S`,

.. math::
   :label: refl-dirs

   \hat{v}_{in}  = \frac{S - T}{\lVert S - T\rVert}, \qquad
   \hat{v}_{out} = \frac{R - S}{\lVert R - S\rVert}, \qquad
   \hat{n} = \hat{n}(S),

the specular condition is that the specularly reflected incoming ray equals the outgoing ray:

.. math::
   :label: refl-mirror

   \hat{v}_{refl} = \hat{v}_{in} - 2\,(\hat{v}_{in}\cdot\hat{n})\,\hat{n}, \qquad
   \mathbf{e}(S) = \hat{v}_{refl} - \hat{v}_{out} = \mathbf{0}.

Equivalently the incidence and reflected angles measured from the normal are equal,

.. math::
   :label: refl-angles

   \cos\left(\theta_{in}\right) = -\,\hat{v}_{in}\cdot\hat{n}, \qquad
   \cos\left(\theta_{out}\right) = \hat{v}_{out}\cdot\hat{n}, \qquad
   \theta_{in} = \theta_{out}.

The stored incidence angle ``refl_ang_vec`` is the mean of :math:`\theta_{in}` and :math:`\theta_{out}`
(equal at a converged solution).

Spherical solver (FreeFlyer method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a sphere the problem is planar. The geometry is rotated so the Tx-Rx plane becomes :math:`z = 0`
(:func:`object_definitions.Reflection_def.Reflection.get_rotation_3D_to_plane_coords`), reducing it to two
dimensions. Writing the reflection point as a combination of the normalised satellite and GNSS vectors
:math:`A = \mathbf{r}_{rx}/R_\oplus`, :math:`B = \mathbf{r}_{tx}/R_\oplus` and
:math:`N = x\,A + y\,B`, the law of reflection collapses to a quartic in :math:`x`,

.. math::
   :label: refl-quartic

   r_4 x^4 + r_3 x^3 + r_2 x^2 + r_1 x + r_0 = 0,

with coefficients in :math:`u = A\cdot A`, :math:`v = A\cdot B`, :math:`w = B\cdot B`,

.. math::
   :label: refl-quartic-coeffs

   \begin{aligned}
   r_0 &= w - 1, & r_1 &= 2(w - v), & r_2 &= u + 2v + w - 4uw,\\
   r_3 &= 4(v^2 - uw), & r_4 &= 4u(uw - v^2). &
   \end{aligned}

The physical root is the real, positive :math:`x` whose partner

.. math::
   :label: refl-quartic-y

   y = \frac{1 + x - 2u x^2}{1 + 2v x}

is also positive; the point is then scaled back to the sphere and rotated into 3-D. This is exact and fast.

Elliptical solver (stationary path on a parameterised surface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the WGS-84 ellipsoid there is no such closed form, so the specular point is found by **Fermat's
principle** -- the reflected ray has stationary optical length. The surface is parameterised by two
angles,

.. math::
   :label: refl-ell-param

   S(\varphi,\lambda) =
   \left(a\cos\varphi\cos\lambda,\ a\cos\varphi\sin\lambda,\ b\sin\varphi\right),

with semi-axes :math:`a = R_{eq}` and :math:`b = a(1-f)`. Consequently every solver iterate lies on the
ellipsoid. For

.. math::
   :label: refl-fermat

   L(S) = \lVert S-T\rVert + \lVert R-S\rVert,

the solver sets the two tangent components of the path gradient to zero,

.. math::
   :label: refl-stationarity

   \nabla_S L\cdot\widehat{S_\varphi}=0,\qquad
   \nabla_S L\cdot\widehat{S_\lambda}=0.

The outward ellipsoid normal at the solution is

.. math::
   :label: refl-ell-normal

   \hat{n}(S) = \frac{(S_x/a^2,\; S_y/a^2,\; S_z/b^2)}{\lVert(S_x/a^2,\; S_y/a^2,\; S_z/b^2)\rVert}.

``scipy.optimize.least_squares`` solves these two dimensionless residuals. The candidate is accepted only
if it faces both satellites (:math:`\cos\theta_{in}>0` and :math:`\cos\theta_{out}>0`) and the independent
vector reflection residual :eq:`refl-mirror` satisfies
:math:`\lVert\mathbf{e}(S)\rVert \le 10^{-8}`. The spherical solution seeds the solve.

Module Assumptions and Limitations
----------------------------------

- **Specular geometry only.** This module locates the specular reflection point and incidence angle; the *scattered
  power* (coherent specular plus incoherent glistening zone) is the
  :class:`~object_definitions.OceanTarget_def.OceanTarget` model, invoked through the link budget. Diffuse
  multipath and the finite glistening zone are not represented here.
- **Ideal reference surface.** Visibility and reflection use either the configured sphere or smooth
  WGS-84 ellipsoid. Terrain, geoid undulation, ice and atmospheric refraction are absent.
- **Coarse central-angle pre-filter.** ``max_central_angle_deg`` only removes far geometries cheaply; it
  is not the physical horizon limit.
- **One Tx-Rx pair per object.** A constellation sweep builds many ``Reflection`` objects.
- **Array layout.** Reflection points are stored ``(n, 3)`` / ``(n, 2)`` while satellite states are
  ``(3, n)`` (a transpose clean-up is noted in the source TODOs).

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test group
     - What it pins
   * - ``test_reflection.py`` :: ``TestReflectionConstruction``
     - Default ``earth_model`` is spherical; explicit models accepted; invalid model, missing satellite
       IDs and missing ``max_central_angle_deg`` all raise.
   * - ``test_reflection_geometry.py`` :: ``TestCheckLosSatellites``
     - Same-side geometry has line of sight; antipodal / far-side and grazing-intersection geometries are
       blocked analytically; the scaled-coordinate ellipsoid test distinguishes a valid polar grazing
       chord from the larger equatorial sphere.
   * - ``...`` :: ``TestGetRotation3DToPlaneCoords``
     - The plane rotation is orthogonal with determinant 1 and flattens the orbital plane onto
       :math:`z = 0`.
   * - ``...`` :: ``TestGetReflectionPoint2D`` / ``TestGetReflectionPointFf``
     - The reflection lands on the sphere and in the orbital plane; returns NaN with no line of sight; the
       law of reflection holds (:math:`\theta_{in} = \theta_{out}`); symmetric and exactly collinear
       same-side geometries are finite and reflect on the symmetry axis.
   * - ``...`` :: ``TestFindSpecularPointEllipseEcef``
     - The point satisfies the ellipsoid constraint to floating-point precision and the incidence/reflection
       angle mismatch is below ``1e-7`` deg for tested geometries.
   * - ``...`` :: ``TestSphericalVsEllipticalAgreement``
     - The two solvers agree within a few km (< 20 km) and both keep an equatorial reflection in
       :math:`z = 0`.

User Guide
----------

``Reflection`` is built internally by the analysis drivers, one per (GNSS, LEO) pair; the relevant INI
knobs live in ``[REFLECTION]``.

.. list-table:: ``[REFLECTION]`` keys
   :widths: 34 66
   :header-rows: 1

   * - Key
     - Meaning
   * - ``reflection_type``
     - ``"specular"`` selects this specular-reflection path (vs ``"backscatter"``, the
       :class:`~object_definitions.BackScattPoint_def.BRP` path).
   * - ``max_central_angle_deg``
     - The constructor's ``max_central_angle_deg`` geometric gate.

.. code-block:: python

   import numpy as np
   from object_definitions.Reflection_def import Reflection

   refl = Reflection(
       tx_satID="GPS-07",
       rx_satID="LEO-1",
       earth_model="elliptical",
       max_central_angle_deg=45.0,
   )

   # GNSS_state / Sat_state are propagated Satellite objects sharing the time vector `times`.
   refl.compute_reflection_coordinates(
       GNSS_state=gnss_sat,
       Sat_state=leo_sat,
       times=times,
       tx_eirp_dbw=27.0,           # from Constants.GNSS_EIRP_DBW
       frequency_hz=1.57542e9,     # GPS L1
       link_budget_settings=settings,
   )
   reflections_eci = refl.refl_ECI_vec   # (n, 3), NaN where no reflection
   incidence_deg = refl.refl_ang_vec     # (n,)

Public interface
^^^^^^^^^^^^^^^^

.. autoclass:: object_definitions.Reflection_def.Reflection
   :members:
   :exclude-members: initialization

References
----------

- Born, M. & Wolf, E., *Principles of Optics*, 7th ed., Cambridge University Press, 1999 (Fermat's
  principle, law of reflection).
- Gleason, S. & Gebre-Egziabher, D. (eds.), *GNSS Applications and Methods*, Artech House, 2009
  (GNSS-R specular-point geometry).
