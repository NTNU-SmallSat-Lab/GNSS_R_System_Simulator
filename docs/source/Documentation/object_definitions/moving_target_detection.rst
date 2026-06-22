Moving-target detection
=======================

.. module:: object_definitions.MovingTargetDetection_def

Summary
-------

``MovingTargetDetection`` is the driver for the moving-target mode. It answers: *as a ship moves along its
route, when is it inside a valid reflection footprint, and what does the present signal model predict?*
It walks the ship trajectory, finds footprint-overlap candidates within a time tolerance, and -- for each
radio band -- evaluates ocean, ship and thermal-noise terms. These are screening-level detectability
metrics, not a full delay-Doppler detector or a calibrated probability of detection.

It ties the whole simulator together: it uses the propagated satellites, the reflection geometry
(:doc:`reflection` or :doc:`backscatter_point`), the ocean and ship RCS models, and the link budget. Its
outputs (detection events, plots, an optional animation) are produced **per band**.

How a detection is scored
^^^^^^^^^^^^^^^^^^^^^^^^^^

Geometry is solved once per transmitter/receiver pair, then the link budget runs once per band (each band
has its own carrier frequency and transmit EIRP). The score depends on the mode:

- **Backscatter mode** -- the ship and the ocean are on the same path, so they can be compared directly.
  The signal-to-clutter ratio and the signal-to-clutter-plus-noise ratio are

  .. math::
     :label: mtd-scr

     \mathrm{SCR} = \frac{P_\mathrm{ship}}{P_\mathrm{ocean}}, \qquad
     \mathrm{SINR} = \frac{P_\mathrm{ship}}{P_\mathrm{ocean} + N_\mathrm{eff}}

  where :math:`N_\mathrm{eff} = N / G_p` is a **detection-equivalent** noise scale: the raw thermal noise
  :math:`N = k\,T_\mathrm{sys}\,B` divided by the coherent gain and the approximate incoherent sensitivity
  :math:`G_p = B\,T_\mathrm{coh}\,\sqrt{M}`, so :math:`N_\mathrm{eff} = k\,T_\mathrm{sys}/(T_\mathrm{coh}\sqrt{M})`
  (see :doc:`../functions/link_budget`). The :math:`\sqrt{M}` term represents reduced estimator variance;
  it is not a reduction of mean noise power. The gain is on the noise term only -- the
  ship and the ocean carry the same GNSS code, so :math:`G_p` cancels in the SCR and only lifts both above
  the thermal floor. A detection is declared when the SINR passes the configured threshold.

- **Specular (forward) mode** -- the ship is seen as a **change** in the forward glint: it blocks (shadows)
  the patch of sea it covers, so the detectable is the small drop in received power,

  .. math::
     :label: mtd-shadow

     \Delta P_r = -\,P_\mathrm{ocean}(A_\mathrm{ship}),

  the ocean glint power over the ship's waterline area (length x beam). It is reported on its own, as a
  fraction of the footprint glint, and against the noise floor. This signal is faint by design and is
  reported honestly, with no flooring.

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
     - ``sat_const`` / ``gnss_const``
     - LEO receiver constellation and the GNSS transmitter constellations.
     - --
   * -
     - ``moving_target_object``
     - The :class:`~object_definitions.ShipTrajectory_def.ShipTrajectory`.
     - --
   * -
     - ``antenna_obj``
     - Receiver antenna.
     - --
   * -
     - ``reflection_mode``
     - "specular" or "backscatter".
     - --
   * -
     - ``deltaT``
     - Time tolerance for matching a reflection to the ship.
     - s
   * -
     - ``link_budget_settings``
     - Rx-system + ocean settings (incl. noise and the detection threshold).
     - --
   * -
     - ``max_central_angle_deg`` / ``max_incidence_angle_deg``
     - Geometry gates (incidence required for backscatter).
     - deg
   * -
     - ``ship_dimensions``
     - ``{length_m, beam_m, freeboard_m}``; enables the ship return.
     - m
   * -
     - ``selected_signals``
     - Per-(constellation, band) plan from the INI bands.
     - --
   * - **Outputs**
     - ``target_detect_dict``
     - Per-constellation detection events (per band).
     - --
   * -
     - ``OUTPUT/Footprint_overlap_<mode>_<CONST>_<band>_<contour>dB.png``
     - Footprint-overlap plots, per-band link-budget breakdown plots, and an optional animation.
     - --

Detailed Module Description
---------------------------

The detection loop
^^^^^^^^^^^^^^^^^^

``calculate_moving_target_reflection_intersection`` is the core. For each GNSS constellation it:

1. takes the reflections that fall inside the trajectory bounding box and time window;
2. for each one, checks whether the reflection point or antenna footprint overlaps the ship's
   position within ``deltaT`` seconds of the matching trajectory time;
3. on a match, evaluates the link budget per band -- ocean power, ship power (if ``ship_dimensions`` is
   set), and noise -- and records a detection event with its SCR/SINR (backscatter) or shadow change
   (specular).

An optimisation only propagates satellites inside windows with geometric satellite visibility
(``fn.compute_visible_times_and_resample_ship`` and ``fn.propagate_only_visible_satellites``). Its default
minimum elevation is the physical horizon (0 deg); the final antenna-frame steering and pattern gate is
applied later.

Per-band evaluation
^^^^^^^^^^^^^^^^^^^

The reflection geometry does not depend on the band, so it is solved once per Tx/Rx pair. The link budget
*does* depend on the band, so it is evaluated for each one in ``signals_per_constellation`` (carrier
frequency, transmit EIRP, and code chip rate). Every output -- events, plots, the SCR/SINR or shadow values
-- is therefore per band, and the band appears in the output filename. The entry script writes
``Footprint_overlap_<mode>_<CONST>_<band>_<contour>dB.png`` for the overlap maps and
``Link_budget_breakdown_<mode>_<CONST>_<band>_*.png`` for the power breakdown.

Key public methods
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Method
     - What it does
   * - ``calculate_moving_target_reflection_intersection``
     - Run the detection loop and fill the per-constellation accumulators.
   * - ``show_link_budget_contributions``
     - Plot the per-event power breakdown (ocean / ship / noise) for a band.
   * - ``show_forward_shadow_summary``
     - Print the forward-shadow change-in-power summary (specular mode).
   * - ``show_track_and_detection_points``
     - Plot the ship track with the detection points marked.
   * - ``create_reflection_animation``
     - Render the reflections travelling along the track as an animation.

API reference
^^^^^^^^^^^^^

.. autoclass:: object_definitions.MovingTargetDetection_def.MovingTargetDetection
   :no-members:

Module Assumptions and Limitations
----------------------------------

- **Two modes, two meanings.** Backscatter gives a modelled ship-vs-ocean SCR/SINR. Specular gives only
  the forward-shadow change in power; it does not yet add the ship's own forward glint.
- **Noise- or clutter-limited depends on settings.** The SINR is formed against a detection-equivalent
  noise sensitivity scale: the detector divides thermal noise by :math:`B\,T_\mathrm{coh}\,\sqrt{M}`. The
  :math:`\sqrt{M}` part is heuristic without an explicit detector statistic and false-alarm probability.
- **Footprint-resolution matching.** A reflection counts as "on the ship" when the point or its footprint
  overlaps the ship position; the footprint is large, so this is a coarse spatial match.
- **No delay-Doppler coincidence check.** The ocean return is evaluated at the reflection/BRP reference
  point and the ship at its own position. The code does not verify that both occupy the same delay and
  Doppler bin. Enabling the DDM-cell area therefore changes the assumed clutter area but does not make the
  candidate association a complete DDM detector.
- **No calibrated detection probability.** ``detection_sinr_threshold_db`` is a user threshold on a
  deterministic metric. False-alarm probability, speckle, fading and measurement statistics are absent.
- **Ship return only when sized.** Without ``ship_dimensions`` the run is ocean-only.

Validation and Tests
--------------------

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_moving_target_detection.py``
     - Construction and the detection-matching logic (with lightweight mocks in place of the heavy mesh).
   * - ``tests/test_moving_target_ship_contribution.py``
     - The per-event ocean / ship power evaluation and the forward-shadow metric.

The verification status and remaining validation work are recorded in :doc:`../../validation`.

User Guide
----------

This is the class behind ``python SRC/main_simulation_movingTarget.py`` (configured by
``Configuration_ship.ini``). The relevant INI pieces:

- ``[MOVING TARGET]`` -- the ship route (see :doc:`ship_trajectory`).
- ``[REFLECTION].reflection_type`` -- ``"specular"`` or ``"backscatter"``.
- ``[ANTENNA].bands`` -- which bands (and so which constellations) to run.
- ``[LINK BUDGET]`` -- losses, the thermal-noise terms, and ``detection_sinr_threshold_db`` (see
  :doc:`../functions/link_budget`).
- ``[OCEAN]`` -- the sea state (see :doc:`ocean_target`).
- ``[SHIP]`` -- ``length_m`` / ``beam_m`` / ``freeboard_m`` to enable the ship return.

References
----------

- Ruf, C. *et al.*, "A new paradigm in earth environmental monitoring with the CYGNSS small satellite
  constellation", *Scientific Reports* 8, 2018.
- Clarizia, M. P. *et al.*, "Target detection using GNSS reflectometry", various, 2010s.
