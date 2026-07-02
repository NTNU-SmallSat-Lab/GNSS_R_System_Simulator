Architecture
============

Both analysis modes are built on the same core: the satellites, the reflection geometry, and the link
budget. This page illustrates how the different parts connect.

Shared core
-----------

Everything up to the reflection geometry is shared: both modes build and propagate the same
constellations and solve the same geometry:

- **Area coverage** counts contiguous specular-track visits over a grid. It has no detection threshold;
  link-budget finiteness is used only as an antenna/FOV validity gate.
- **Moving-target detection** follows a ship and scores each reflection, in specular or backscatter mode.

For the step-by-step flow, see :doc:`running`.

Software architecture
---------------------

.. figure:: images/architecture.svg
   :alt: Layered software architecture from the entry scripts to the shared foundation
   :align: center
   :width: 100%

   The modules in four layers. Thick arrows are instantiation, thin arrows composition, and dashed
   arrows foundation use. ``Constants`` is imported almost everywhere and is left out for clarity.

From top to bottom: the entry scripts and ``initialize`` (INI parsing); the two mode drivers
(:doc:`Coverage <Documentation/object_definitions/coverage>`,
:doc:`MovingTargetDetection <Documentation/object_definitions/moving_target_detection>`); the shared
domain model (satellites, reflection geometry, ship trajectory, antenna); and the foundation
(constants, link budget, helpers, IO), which is imported broadly.

Class diagrams (UML)
--------------------

Two UML class diagrams document the object model. The editable PlantUML sources live in
``ARCHITECTURE/`` (``architecture.puml``, ``scattering.puml``) and are regenerated -- together with
the SVGs embedded here -- by ``python ARCHITECTURE/make_uml_plantuml.py``; the same sources render
to vector PDF for publication use (see the script header).

.. figure:: images/uml_system.svg
   :alt: UML class diagram of the system architecture
   :align: center
   :width: 100%

   System architecture. The two analysis modes (``Coverage``, ``MovingTargetDetection``) orchestrate
   the propagated constellations, the per-pair reflection geometry (``Reflection``, specular, and
   ``BRP``, backscatter), the steerable receiver antenna, and the TLE foundation. The
   ``functions.link_budget`` stub marks the hand-off to the scattering model below.

.. figure:: images/uml_scattering.svg
   :alt: UML class diagram of the scattering and link-budget model
   :align: center
   :width: 100%

   Scattering and link-budget model. ``functions.link_budget`` builds the per-sample
   ``BistaticGeometry`` and evaluates the bistatic radar equation against a pluggable ``Target``
   RCS model -- ``OceanTarget`` (coherent + geometric-optics + Bragg sea surface) or ``ShipTarget``
   (plates and hull-water dihedrals) -- using the pure-function physics in
   ``functions.ocean_physics`` / ``functions.ship_physics`` and the delay-Doppler clutter cell in
   ``functions.delay_doppler``.
