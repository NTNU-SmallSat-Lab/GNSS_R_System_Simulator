Verification and validation status
==================================

Scientific software needs a distinction between **verification** (the equations were implemented as
intended) and **validation** (the model predicts observations in its intended domain). Most of this
repository's evidence is currently verification. The simulator has not yet been calibrated end-to-end
against a specific GNSS-R instrument or a collocated truth dataset.

Evidence levels used here
-------------------------

The documentation uses the following terms deliberately:

#. **Analytic verification** -- a closed-form identity, constraint, symmetry or limiting case is checked.
#. **Numerical verification** -- an analytic derivative or solver is compared with finite differences or
   a tighter independent residual.
#. **Regression protection** -- a golden file or fixed expected value detects unintended code changes.
#. **Plausibility check** -- a result lies in a deliberately broad physical envelope or follows an
   expected trend.
#. **External validation** -- predictions are compared with independent, polarisation- and
   geometry-matched observations with stated uncertainty.

Levels 1--4 exist in the test suite. Level 5 is still required before claiming absolute ocean NRCS,
ship RCS, received power or detection probability.

Current evidence by subsystem
-----------------------------

.. list-table::
   :widths: 24 32 44
   :header-rows: 1

   * - Subsystem
     - Evidence in the repository
     - Scientific interpretation
   * - Sphere and WGS-84 specular geometry
     - Surface constraint, equal-angle residual, collinear case, analytic spherical occultation, and
       sphere/ellipsoid comparison.
     - Strong equation-level verification for tested geometries; no independent flight geolocation
       comparison.
   * - Backscatter reflection point
     - Line/surface constraint, root beyond the receiver, sphere/ellipsoid selection and incidence gate.
     - Strong geometric verification of the defined collinear BRP model.
   * - Delay-Doppler Jacobian
     - Finite-difference checks, including rigid Earth rotation; scaling and degeneracy tests.
     - Verified first-order local map away from singular points. Exact specular cells need second-order
       delay curvature and are not implemented.
   * - Seawater permittivity and Fresnel terms
     - Closed-form limits, passivity and representative L-band regression values.
     - Equation/trend verification; not a fresh reproduction of the source datasets.
   * - Cox-Munk, Elfouhaily and Bragg/two-scale ocean terms
     - PDF normalisation, positivity, wind/incidence trends, anisotropy and broad NRCS envelopes.
     - Plausibility plus regression protection. No polarisation-matched collocated measurement
       validation.
   * - Bistatic link equation and thermal noise
     - Algebraic scaling, :math:`kTB`, :math:`C/N_0`, EIRP deltas and golden regression.
     - Link-equation implementation is verified; the input EIRP, antenna and target models control
       absolute accuracy.
   * - Antenna
     - Array-factor peaks, steering convention, interpolation, input guards and golden regression.
     - Verified ideal isotropic-element array factor; not an efficiency-, coupling- or chamber-calibrated
       antenna model.
   * - Ship RCS
     - Plate/dihedral formula peaks, nulls, visibility and component summation.
     - Internal formula verification only. Absolute ship RCS is preliminary.
   * - Moving-target decision metric
     - Event plumbing and deterministic SCR/SINR/shadow calculations.
     - A screening metric, not a validated detector: no false-alarm probability, speckle distribution or
       delay-Doppler target-bin association.
   * - Coverage
     - Grid construction, contiguous-track segmentation and cell intersection.
     - Verified pass-count algorithm. It is geometric coverage, not detection coverage.

Important interpretation rules
------------------------------

- A passing golden-file test means the implementation did not change; it does not prove the frozen value
  is physically correct.
- A broad ``dB`` envelope catches unit/sign failures but is not a curve-to-data validation.
- ``noise_power_effective_w`` is a threshold-equivalent sensitivity scale. Incoherent averaging reduces
  estimator variance, not mean thermal noise power.
- With ``delay_doppler_cell_enabled``, the code changes the assumed ocean clutter area. It does not prove
  that a ship anywhere in the antenna footprint occupies the reference point's delay-Doppler bin.
- At the exact specular point the first-order delay gradient vanishes, so the implemented DDM area uses
  the antenna-footprint cap.

Minimum external-validation programme
-------------------------------------

Before using the model for absolute performance claims:

#. Reproduce published Klein-Swift, Cox-Munk and Elfouhaily curves from digitised/reference data with
   explicit units, polarisation conventions and tolerances.
#. Compare ocean NRCS or DDM slices against a mission product for matched carrier, incidence, wind speed,
   wind direction, wave age and antenna response.
#. Replace fixed constellation/band EIRP assumptions with signal- and transmitter-specific values and
   propagate their uncertainty.
#. Validate the antenna model against an element pattern or measured full-array pattern including
   efficiency, backlobe and scan loss.
#. Compare the ship model with full-wave simulation, a defensible RCS database or measurements over
   aspect, polarisation and sea state.
#. Implement target delay/Doppler coordinates, bin membership, speckle/noise statistics and a detector
   with specified probability of false alarm; then report probability of detection rather than only a
   deterministic threshold crossing.
#. Record orbit/TLE epoch and IERS data versions for every geolocation comparison.

References
----------

- Klein, L. & Swift, C., *IEEE Transactions on Antennas and Propagation* 25(1), 1977.
- Cox, C. & Munk, W., *Journal of the Optical Society of America* 44(11), 1954.
- Elfouhaily, T. *et al.*, *Journal of Geophysical Research* 102(C7), 1997.
- Zavorotny, V. & Voronovich, A., *IEEE Transactions on Geoscience and Remote Sensing* 38(2), 2000.
- Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013.
