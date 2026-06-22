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
hull-water dihedrals** (the right-angle corner where each side meets the sea, which are strong
retroreflectors). Each plate uses the flat-plate formula and each dihedral the corner formula from
:doc:`../functions/helpers`. The pieces add up without phase (non-coherently), and the ``info`` dict keeps
the per-piece breakdown. This is an ideal-conductor engineering surrogate for aspect studies, not a
validated absolute ship-RCS model.

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
five plates and two dihedrals are placed in this frame, each evaluated with its own bistatic visibility, and
summed:

.. math::
   :label: ship-sum

   \sigma_\mathrm{ship} = \sum_{\text{5 plates}} \sigma_\mathrm{plate} + \sum_{\text{2 dihedrals}} \sigma_\mathrm{dihedral}.

.. autoclass:: object_definitions.ShipTarget_def.ShipTarget
   :members:

Module Assumptions and Limitations
----------------------------------

- **Box geometry.** The ship is a rectangular box (plates + waterline dihedrals); superstructure, masts,
  curved hulls, and material effects are not modelled.
- **Ideal PEC and scalar response.** Plates are perfectly conducting and use a scalar physical-optics
  form factor. Polarisation matrices, diffraction, creeping waves and edge interactions are omitted.
- **Simplified dihedral.** The dihedral uses an ideal peak with an along-edge sinc and binary open-wedge
  visibility. It omits the full angular/polarimetric response, finite leg diffraction and coupling to a
  rough moving sea.
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
       back-face and wedge visibility behaviour.

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
