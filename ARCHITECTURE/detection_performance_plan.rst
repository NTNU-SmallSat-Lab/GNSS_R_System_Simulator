Detection Performance Model Plan
================================

Goal
----

Upgrade the current power-budget prototype into a model that can support
defensible detection-performance claims, including Pd, Pfa, SCR, and SNR
envelopes. The current implementation is useful for received-power estimates,
but detection claims require a measurement model, noise model, clutter
statistics, and an explicit detector.

Recommended Dependency Order
----------------------------

1. Receiver noise and sensitivity model

   Add ``N = k T_sys B``, including antenna temperature, receiver noise figure,
   bandwidth, coherent and noncoherent integration, quantization loss, and
   implementation losses.

2. Delay-Doppler measurement cell

   Replace footprint-only detection logic with a code/Doppler resolution model
   using the GNSS ambiguity function, Doppler bin width, coherent integration
   time, and geometry migration during integration.

3. Local clutter in the same measurement cell as the ship

   Compute ocean clutter and ship signal over the same antenna/delay/Doppler
   cell. Do not mix ocean power at the specular point or BRP with ship power at
   an offset footprint location without a cell-integration model.

4. Statistical clutter and target fluctuations

   Model ocean speckle, glistening-zone variability, Bragg variability,
   sea-state uncertainty, and stochastic ship RCS/aspect fluctuations. Detection
   thresholds need distributions, not only mean power.

5. Detection rule

   Define the detector explicitly: fixed threshold, SCR threshold, CFAR, matched
   filter, multi-look accumulation, or track-before-detect. Report Pd and Pfa
   from that rule.

6. Realistic ship RCS

   Validate or replace the current flat-plate/dihedral ship model with
   aspect-dependent, stochastic, and sea-state-aware RCS. Include roll, pitch,
   yaw, wave masking, superstructure/material effects, or external EM/measured
   references where possible.

7. GNSS transmitter realism

   Include transmitter antenna gain toward the surface, PRN/signal-specific
   power, polarization purity, attitude/block variation, and per-constellation
   and per-band assumptions.

8. Receiver antenna and platform realism

   Validate absolute gain, scan loss, polarization response, axial ratio, side
   lobes, attitude error, pointing error, and direct-signal leakage.

9. External validation

   Compare against accepted GNSS-R ocean DDM power levels, L-band ocean NRCS
   curves, CYGNSS/TDS-1-style reference cases, analytic synthetic cases, and
   measured or externally simulated ship RCS.

Practical Bar
-------------

Internal feasibility:
    Receiver noise, delay-Doppler cell integration, local clutter in the same
    cell, a defined threshold detector, and sensitivity sweeps over sea state,
    aspect, and antenna gain.

Publication-grade claims:
    Stochastic clutter/RCS, calibrated antenna/noise/GNSS transmitter
    assumptions, and validation against external measurements or accepted
    reference models.
