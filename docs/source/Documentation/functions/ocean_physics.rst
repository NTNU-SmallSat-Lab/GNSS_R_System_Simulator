Ocean physics
=============

Summary
-------

This module holds the building-block physics for how the sea reflects and scatters a GNSS signal. It is a
set of pure functions -- no state, no classes -- shared by the ocean target model. They answer four
questions:

- How well does seawater reflect at L-band? (its electrical properties, the **permittivity**)
- How much of a flat surface bounces back? (the **Fresnel** and circular-polarisation reflectivities)
- How rough is the sea? (the **Cox-Munk** wave-slope statistics and the **Elfouhaily** wave spectrum)
- How much signal scatters back at an angle? (the **Bragg** and **two-scale** backscatter)

The implementation follows the cited published models. The tests pin algebraic identities, limiting
cases, regression values and broad trends; they do not constitute independent validation against raw
measurements or a mission-calibrated GNSS-R product.

Interface
---------

.. list-table:: Main functions
   :widths: 26 60 14
   :header-rows: 1

   * - Function
     - What it returns
     - Unit
   * - ``seawater_permittivity``
     - Complex permittivity of seawater (Klein-Swift).
     - --
   * - ``fresnel_coefficients``
     - Fresnel reflection for vertical and horizontal polarisation.
     - --
   * - ``circular_pol_reflectivities``
     - Power reflectivities for RHCP->LHCP and RHCP->RHCP.
     - --
   * - ``cox_munk_slope_variances``
     - Up/cross-wind wave-slope variances vs wind.
     - --
   * - ``cox_munk_slope_pdf``
     - Probability of a given surface slope.
     - --
   * - ``elfouhaily_omnidirectional_spectrum``
     - Sea-surface height spectrum at one wavenumber.
     - m^3
   * - ``bragg_scattering_coefficients``
     - Bragg (small-ripple) amplitudes for H and V.
     - --
   * - ``bragg_backscatter_nrcs``
     - First-order Bragg backscatter (circular).
     - --
   * - ``bragg_bistatic_coefficients``
     - Bistatic Bragg amplitudes for HH, HV, VH, VV.
     - --
   * - ``bragg_bistatic_nrcs``
     - First-order bistatic Bragg (circular).
     - --
   * - ``large_scale_slope_variances``
     - Slope variance of the large tilting waves.
     - --
   * - ``two_scale_bistatic_nrcs``
     - Tilt-modulated (two-scale) bistatic Bragg (circular).
     - --
   * - ``two_scale_backscatter_nrcs``
     - Monostatic case of the two-scale model.
     - --

Detailed Module Description
---------------------------

Seawater permittivity (Klein-Swift)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Permittivity says how strongly seawater responds to the radio wave; it sets how much of the signal
reflects. The Klein-Swift model gives it as a Debye relaxation plus an ionic (salt) loss term,

.. math::
   :label: op-permittivity

   \varepsilon_r = \varepsilon_\infty + \frac{\varepsilon_s - \varepsilon_\infty}{1 + j\,\omega\tau}
   - \frac{j\,\sigma_\mathrm{ion}}{\omega\,\varepsilon_0},

where :math:`\varepsilon_s` is the low-frequency value, :math:`\varepsilon_\infty = 4.9` the
high-frequency limit, :math:`\tau` the relaxation time, :math:`\sigma_\mathrm{ion}` the salt
conductivity, and :math:`\omega = 2\pi f`. The static value, relaxation time, and conductivity all depend
on temperature and salinity (the polynomial fits in the code).

Fresnel and circular reflectivities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a flat surface, the Fresnel coefficients give the reflection for the two linear polarisations,
vertical (V) and horizontal (H):

.. math::
   :label: op-fresnel

   R_h = \frac{\cos\left(\theta\right) - \sqrt{\varepsilon_r - \sin^2\left(\theta\right)}}{\cos\left(\theta\right) + \sqrt{\varepsilon_r - \sin^2\left(\theta\right)}},
   \qquad
   R_v = \frac{\varepsilon_r\cos\left(\theta\right) - \sqrt{\varepsilon_r - \sin^2\left(\theta\right)}}{\varepsilon_r\cos\left(\theta\right) + \sqrt{\varepsilon_r - \sin^2\left(\theta\right)}}.

GNSS transmits Right-Hand Circular (RHCP); a bounce flips most of it to Left-Hand (LHCP). Combining the
Fresnel terms gives the circular reflectivities, cross-pol (the wanted RHCP->LHCP) and co-pol
(RHCP->RHCP):

.. math::
   :label: op-circular

   R_\mathrm{RL} = \tfrac{1}{2}(R_h - R_v), \qquad R_\mathrm{RR} = \tfrac{1}{2}(R_h + R_v),

and the power reflectivities are :math:`|R_\mathrm{RL}|^2` and :math:`|R_\mathrm{RR}|^2`.

How rough the sea is (Cox-Munk and Elfouhaily)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A rough sea has tilted facets. Cox-Munk give the spread of those tilts (the slope variance) as a simple
function of wind speed, larger upwind than crosswind:

.. math::
   :label: op-coxmunk

   \sigma_u^2 = 3.16\times 10^{-3}\,U_{10}, \qquad
   \sigma_c^2 = 3.0\times 10^{-3} + 1.92\times 10^{-3}\,U_{10},

and the chance of a given slope :math:`(s_u, s_c)` is a 2-D Gaussian,

.. math::
   :label: op-slope-pdf

   p(s_u, s_c) = \frac{1}{2\pi\,\sigma_u\,\sigma_c}\,
   \exp\!\left(-\frac{s_u^2}{2\sigma_u^2} - \frac{s_c^2}{2\sigma_c^2}\right).

For backscatter we need the wave height at one exact ripple size, not the whole-sea average. The
Elfouhaily spectrum provides that. It is written as a curvature spectrum :math:`B(k)` (a long-wave plus a
short-wave part), and the height spectrum is

.. math::
   :label: op-elfouhaily

   S(k) = \frac{B(k)}{k^3} = \frac{B_\mathrm{low}(k) + B_\mathrm{high}(k)}{k^3}.

Bragg backscatter
^^^^^^^^^^^^^^^^^

Away from the specular direction (beyond ~20 deg), the return is dominated by **Bragg** scattering: the
radar resonates with the one ripple whose spacing matches the wave, at wavenumber

.. math::
   :label: op-kbragg

   k_B = 2k\sin\left(\theta\right), \qquad k = \frac{2\pi}{\lambda}.

The first-order (small-ripple) backscatter cross-section per unit area is

.. math::
   :label: op-bragg-nrcs

   \sigma^0_{pp} = 16\pi\,k^4\cos^4\left(\theta\right)\,|g_{pp}|^2\,W_\mathrm{iso}(k_B),

where :math:`W_\mathrm{iso}(K) = S(K)/(2\pi K)` is the roughness at the Bragg wavenumber (normalised so
that its integral over the 2-D wavenumber plane is the height variance; with this normalisation
:math:`16\pi` is the small-roughness limit of the same Kirchhoff integral whose large-roughness limit is
the geometric-optics NRCS) and
:math:`g_{pp}` are the Bragg amplitudes for H and V. The circular amplitudes come from those:

.. math::
   :label: op-bragg-circular

   g_\mathrm{RL} = \tfrac{1}{2}(g_{HH} + g_{VV}), \qquad g_\mathrm{RR} = \tfrac{1}{2}(g_{HH} - g_{VV}).

Near nadir :math:`k_B \to 0`, the Bragg return vanishes and the quasi-specular (geometric-optics) term takes
over, so this returns 0 there.

Bistatic Bragg
^^^^^^^^^^^^^^

For a receiver away from the backscatter direction the resonant ripple is set by the horizontal part of
the scattering vector,

.. math::
   :label: op-kbragg-bistatic

   K = k\,\left|\hat{k}_{s,t} - \hat{k}_{i,t}\right|
     = k\sqrt{\sin^2\theta_i + \sin^2\theta_s - 2\sin\theta_i\sin\theta_s\cos\phi_s},

with :math:`\phi_s` the scattering azimuth from the forward direction. The first-order cross-section is

.. math::
   :label: op-bragg-bistatic

   \sigma^0 = 16\pi\,k^4\cos^2\theta_i\cos^2\theta_s\,|a|^2\,W_\mathrm{iso}(K),

with the bistatic amplitudes :math:`a_{pq}` (Valenzuela) in forward-aligned linear bases
(:math:`\hat{h} = \hat{z}\times\hat{k}/|\hat{z}\times\hat{k}|`, :math:`\hat{v} = \hat{h}\times\hat{k}`).
The circular amplitudes are the projections onto each wave's own circular vectors
(:math:`(\hat{v} \mp j\hat{h})/\sqrt{2}` for RHCP/LHCP under ``exp(+j omega t)``):

.. math::
   :label: op-bragg-bistatic-circular

   a_\mathrm{RL} = \tfrac{1}{2}\left[(a_{vv} - a_{hh}) - j(a_{vh} + a_{hv})\right], \qquad
   a_\mathrm{RR} = \tfrac{1}{2}\left[(a_{vv} + a_{hh}) + j(a_{hv} - a_{vh})\right].

At backscatter (:math:`\theta_s = \theta_i`, :math:`\phi_s = \pi`) the amplitudes are
:math:`a_{hh} = -g_{HH}` and :math:`a_{vv} = g_{VV}`, so :eq:`op-bragg-bistatic` reduces to
:eq:`op-bragg-nrcs`. In the specular direction :math:`K = 0` and the first-order return vanishes.

Two-scale (tilt) modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real waves are small ripples riding on big swells. The big swells **tilt** the patches that carry the
ripples, which changes the local angles the radar sees. The two-scale model averages the bistatic Bragg
return over the spread of those tilts: for each facet slope it works out the local incidence and
scattering angles and the local scattering azimuth, evaluates the Bragg return in the facet frame, and
sums it per unit mean-surface area, i.e. weighted by the slope probability and the facet area
:math:`\sqrt{1 + |\nabla z|^2}` per unit horizontal area. This lifts the return a couple of dB at low
incidence and adds an upwind-vs-crosswind difference that first-order Bragg alone misses. The complete
anisotropic slope covariance is rotated into the frame of the transmitter azimuth; its off-diagonal
covariance is retained for intermediate wind-relative azimuths. Facets whose local Bragg wavenumber (the
tangential scattering vector) drops below the two-scale division :math:`k/3` add no Bragg return: they are
quasi-specular mirrors, already counted by the geometric-optics term. Facets hidden from the transmitter
or the receiver add nothing. All facets stay in the normalisation. A tilt rotates the linear polarisation
bases, which only changes the phase of a circular amplitude, so the tilt-induced polarisation rotation is
included. As the swell slope goes to zero it reduces back to plain first-order Bragg.

API reference
^^^^^^^^^^^^^

.. automodule:: functions.ocean_physics
   :members:

Module Assumptions and Limitations
----------------------------------

- **Valid sea-state range.** Klein-Swift is fit for roughly -2 to 30 degC and 0 to 40 PSU; Cox-Munk is for
  clean (slick-free) seas; Bragg holds for moderate incidence (about 10 to 70 deg).
- **Near-nadir Bragg divergence.** First-order Bragg grows without bound toward nadir
  (:math:`\sigma^0 \sim B/(4\sin^4\theta)` as the resonant wavenumber slides into the energetic long-wave
  spectrum), and the bistatic form diverges likewise toward the specular direction. Inside the two-scale
  tilt average this is contained by the facet-local restriction :math:`K_\mathrm{local} \ge k/3`; there the
  geometric-optics term dominates. The restriction is a hard cut at the division wavenumber.
- **First-order Bragg.** The small-ripple theory is first order; higher-order and multiple scattering are
  not included.
- **Numerical two-scale quadrature.** The large-scale slope average uses a fixed 15 by 15 grid over
  :math:`\pm3` standard deviations, normalised over all facets. Convergence for extreme wind or grazing
  incidence is not adaptively controlled.
- **Circular vs linear references.** The model gives circular-polarisation results, while many published
  L-band measurements are linear (V/H); comparisons are envelope-and-trend, not exact dB matches.
- **Sign convention.** The imaginary part of the permittivity is negative here (``exp(+j omega t)``);
  losses add on that side.

Validation and Tests
--------------------

``tests/test_ocean_physics.py`` provides implementation verification at several levels:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test group
     - What it checks
   * - ``TestSeawaterPermittivity``
     - An L-band seawater value lies in a broad expected range; the high-frequency limit tends to 4.9.
   * - ``TestFresnelCoefficients`` / ``TestCircularPolReflectivities``
     - Closed-form limits at normal incidence and grazing; reflectivity never exceeds 1 (no energy gain).
   * - ``TestCoxMunkSlopeVariances`` / ``TestCoxMunkSlopePdf``
     - Slope variance grows with wind; the slope PDF integrates to 1.
   * - ``TestElfouhailySpectrum``
     - Spectrum positivity, shape and wind/wave-age trends are regression-checked.
   * - ``TestBraggScatteringCoefficients`` / ``TestBraggBackscatterNrcs``
     - Bragg amplitude identities hold at nadir; a representative L-band value is constrained to a broad
       plausibility interval and angular trends are checked.
   * - ``TestLargeScaleSlopeVariances`` / ``TestTwoScaleBackscatterNrcs``
     - The tilt average lifts the low-incidence return, adds upwind/crosswind anisotropy, and preserves
       scalar/array consistency at intermediate azimuth.
   * - ``TestBraggBistatic``
     - The bistatic amplitudes match a first-principles perfect-conductor boundary perturbation (signs
       included), the dielectric scattering dyadic is reciprocal, and the bistatic NRCS reduces to the
       backscatter form.
   * - ``TestTwoScaleBistaticNrcs``
     - The facet average matches an independent east/north/up reference with a wind-frame quadrature
       (pins the azimuth conventions) and is continuous through near-backscatter geometry.

The distinction matters: these tests can expose coding and unit errors, but an absolute NRCS claim still
requires comparison with polarisation-matched measurements under known wind, wave-age and incidence
conditions.

User Guide
----------

These are internal physics helpers; you do not call them directly from a run. They are driven by the
``[OCEAN]`` sea-state settings (wind, temperature, salinity, wave age) through
:class:`~object_definitions.OceanTarget_def.OceanTarget`. See that page for the configuration keys.

References
----------

- Klein, L. & Swift, C., "An improved model for the dielectric constant of sea water at microwave
  frequencies", *IEEE Trans. Antennas Propag.* 25(1), 1977.
- Cox, C. & Munk, W., "Measurement of the roughness of the sea surface from photographs of the sun's
  glitter", *J. Opt. Soc. Am.* 44(11), 1954.
- Elfouhaily, T. *et al.*, "A unified directional spectrum for long and short wind-driven waves", *JGR*
  102(C7), 1997.
- Valenzuela, G., "Theories for the interaction of electromagnetic and oceanic waves -- a review",
  *Boundary-Layer Meteorol.* 13, 1978.
