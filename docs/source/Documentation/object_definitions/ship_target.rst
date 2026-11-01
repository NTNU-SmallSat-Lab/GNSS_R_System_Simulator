Targets: base class and ship
============================

Every scattering target (the ocean, a ship) shares one small interface so the link budget can treat them
the same way. This page covers the abstract base and the ship model; the ocean model has its own page
(:doc:`ocean_target`).

Target base
-----------

.. module:: object_definitions.Target_def

Summary
^^^^^^^

``Target`` is the abstract base for any RCS model. It defines one method, ``compute_rcs(geometry,
wavelength_m, illuminated_area_m2)``, which returns a ``TargetRcs`` -- the RCS per sample plus an ``info``
dictionary of diagnostics. The ``illuminated_area_m2`` argument is for distributed (area-extensive) targets
like the ocean; point targets such as a ship ignore it. This shared interface is what lets the link budget
swap one target for another without changing.

.. autoclass:: object_definitions.Target_def.Target
   :members:

.. autoclass:: object_definitions.Target_def.TargetRcs
   :members:

Ship target
-----------

.. module:: object_definitions.ShipTarget_def

Summary
^^^^^^^

``ShipTarget`` models a ship as a simple box of metal surfaces and returns its radar cross-section. The
hull is broken into **five flat plates** (deck, port side, starboard side, bow, stern) and **two
hull-water dihedrals** (the right-angle corner where each side meets the sea). Each plate uses the
flat-plate formula and each dihedral the corner formula from :doc:`../functions/helpers`, with a seawater
sea leg and a diffuse part from the sea scatter that the hull mirrors. The pieces add up without phase (non-coherently), and the ``info`` dict keeps the per-piece
breakdown. This is an ideal-conductor engineering surrogate for aspect studies, not a validated absolute
ship-RCS model.

Interface
^^^^^^^^^

.. list-table:: Inputs / outputs
   :widths: 16 28 42 14
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs** (constructor)
     - ``length_m`` / ``beam_m`` / ``freeboard_m``
     - Ship length, width, and height above the waterline.
     - m
   * -
     - ``heading_eci``
     - Per-sample heading direction ``(n, 3)`` (sets the bow axis).
     - --
   * -
     - ``ocean``
     - The ``OceanTarget`` sea state (wave height, temperature, salinity) for the sea leg of the corners.
     - --
   * - **Inputs** (``compute_rcs``)
     - ``geometry`` / ``wavelength_m``
     - Bistatic geometry and carrier wavelength.
     - --, m
   * - **Outputs**
     - ``TargetRcs.rcs_m2``
     - Total ship RCS per sample.
     - m^2
   * -
     - ``TargetRcs.info``
     - Per-plate and per-dihedral breakdown.
     - m^2

The ship body frame
^^^^^^^^^^^^^^^^^^^^

Each sample builds a ship frame from the heading and the local up direction:

.. math::
   :label: ship-frame

   \hat{b} = \text{heading (flattened onto the sea)}, \qquad
   \hat{m} = \hat{n}_\mathrm{surface}, \qquad
   \hat{s} = \hat{b} \times \hat{m},

i.e. bow :math:`\hat{b}` (forward), mast :math:`\hat{m}` (up), and starboard :math:`\hat{s}` (right). The
five plates and two dihedrals are placed in this frame, each evaluated with its own bistatic visibility.

The sea leg of the corner
^^^^^^^^^^^^^^^^^^^^^^^^^

The dihedral formula :math:`\sigma_\mathrm{dihedral}` has two metal legs. The hull leg is metal, but the
other leg is the sea. A metal bounce reverses the sense of circular polarisation. The transmitted RHCP signal
therefore meets the sea as RHCP on one path and as LHCP after the hull bounce on the other. After both bounces,
only the part that the sea reflects in the same sense arrives as LHCP at the receiver. Sea roughness removes the
coherent corner, as it removes the coherent glint (:doc:`ocean_target`). The coherent corner is therefore

.. math::
   :label: ship-sea-leg

   \sigma_\mathrm{coh} = |R_\mathrm{RR}(\theta_\mathrm{sea})|^2\, e^{-\kappa}\, \sigma_\mathrm{dihedral}, \qquad
   \kappa = \left(\frac{4\pi\,\sigma_h\cos\theta_\mathrm{sea}}{\lambda}\right)^2,

with :math:`\theta_\mathrm{sea}` the incidence angle of the transmitter ray on the sea, :math:`|R_\mathrm{RR}|^2`
the same-sense (RHCP-to-RHCP) power reflectivity of seawater, and :math:`\sigma_h \simeq H_s/4` the RMS surface
height. The hull bounce keeps the vertical component of the ray, so :math:`\theta_\mathrm{sea}` is the same on
both paths.

The diffuse corner
^^^^^^^^^^^^^^^^^^

A rough sea also scatters in all directions, and the hull mirrors part of this scatter to the receiver. This
diffuse part of the corner survives the roughness. On the sea-first path, the sea scatters toward the image of
the receiver in the hull plane. On the hull-first path, the sea is lit by the image of the transmitter. Each
path sees the sea strip that the hull rectangle mirrors: the length :math:`L` times the width
:math:`H\tan\theta_v\cos\phi_v`, with :math:`\theta_v` the angle of the receiver (:math:`v = \mathrm{Rx}`) or of
the transmitter (:math:`v = \mathrm{Tx}`) from the vertical and :math:`\phi_v` its azimuth from the hull normal.
Only the same-sense sea scatter ends as LHCP, so

.. math::
   :label: ship-diffuse-corner

   \sigma_\mathrm{diff} = L H \sum_{v\,\in\,\{\mathrm{Rx},\,\mathrm{Tx}\}} \tan\theta_v \cos\phi_v\;
   \sigma^0_\mathrm{RR}(v),

with :math:`\sigma^0_\mathrm{RR}(v)` the same-sense ocean NRCS of :doc:`ocean_target` (geometric optics with
:math:`|R_\mathrm{RR}|^2` in place of :math:`|R_\mathrm{RL}|^2`, plus the two-scale Bragg term) for the
transmitter and the receiver image (:math:`v = \mathrm{Rx}`) or the transmitter image and the receiver
(:math:`v = \mathrm{Tx}`). The two paths add in power. Like the coherent corner, the diffuse corner needs the
transmitter and the receiver on the outward side of the hull. The pieces add up as

.. math::
   :label: ship-sum

   \sigma_\mathrm{ship} = \sum_{\text{5 plates}} \sigma_\mathrm{plate}
   + \sum_{\text{2 corners}} \left(\sigma_\mathrm{coh} + \sigma_\mathrm{diff}\right).

.. autoclass:: object_definitions.ShipTarget_def.ShipTarget
   :members:

Module Assumptions and Limitations
----------------------------------

- **Box geometry.** The ship is a rectangular box (plates + waterline dihedrals); superstructure, masts,
  curved hulls, and material effects are not modelled.
- **Ideal PEC and scalar response.** Plates are perfectly conducting and use a scalar physical-optics
  form factor. Polarisation matrices, diffraction, creeping waves and edge interactions are omitted.
- **Simplified corner.** The corner uses a scalar two-path double bounce with a flat metal hull leg and
  binary open-wedge visibility. Its diffuse part uses the ocean NRCS over the mirrored strip, and its two
  paths add in power. Edge diffraction and the part of the strip that stays lit when the transmitter and the
  receiver are on opposite sides of the hull are omitted.
- **Non-coherent sum.** Components add in power, not amplitude, so interference between facets is not
  captured.
- **Above-waterline only.** Only the freeboard (above-water) part contributes; the submerged hull does not
  return at L-band.
- **Deterministic, not stochastic.** RCS is a fixed function of aspect; real ships fluctuate with roll,
  pitch, and yaw (a planned refinement).
- **No absolute calibration.** Tests verify implementation identities and visibility rules only. They do
  not compare against full-wave simulation, an RCS database or measurements, so absolute received power
  and detection range should be treated as preliminary.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_ship_target.py``
     - Construction and validation; the body-frame build; the per-plate / per-dihedral RCS and their sum;
       back-face and wedge visibility behaviour; the seawater sea leg and its roughness loss; the diffuse
       corner (mirrored strip, same-sense ocean NRCS) and its visibility.

User Guide
----------

The ship dimensions come from the ``[SHIP]`` section of the moving-target config:

.. list-table:: ``[SHIP]`` keys
   :widths: 30 56 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``length_m``
     - Ship length.
     - m
   * - ``beam_m``
     - Ship width.
     - m
   * - ``freeboard_m``
     - Height above the waterline.
     - m

When ``[SHIP]`` is set, the detector evaluates the ship return at each detection event; otherwise the run
is ocean-only.

References
----------

- Knott, E., Schaeffer, J. & Tulley, M., *Radar Cross Section*, 2nd ed., SciTech, 2004.
- Skolnik, M., *Radar Handbook*, 3rd ed., McGraw-Hill, 2008 (ship RCS).
