Ocean target
============

.. module:: object_definitions.OceanTarget_def

Summary
-------

``OceanTarget`` is the sea-surface RCS (Radar Cross-Section) model. Given the bistatic geometry and the
carrier wavelength, it returns how strongly the ocean scatters the GNSS signal toward the receiver. It is
the ocean side of the link budget -- the background that a ship has to stand out from in backscatter mode,
and the glint a ship blocks in forward mode.

The model adds two parts:

- a **coherent** part -- a clean specular reflection off the smooth mean surface (matters only for very calm
  water);
- an **incoherent** part -- the diffuse scatter from the rough surface, which is what you usually see. At
  the specular direction this is the bright "glistening zone"; well off the specular direction (backscatter) it
  is **Bragg** scatter from small wind ripples.

It is driven by sea state (wave height, wind speed and direction, water temperature, salinity, wave age),
and it uses the building blocks in :doc:`../functions/ocean_physics`.

Interface
---------

.. list-table:: Inputs / outputs
   :widths: 16 28 42 14
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs** (constructor)
     - ``wave_height_m``
     - Significant wave height (sets the coherent roughness loss).
     - m
   * -
     - ``wind_speed_m_s``
     - Wind speed at 10 m (drives the wave slopes and Bragg ripples).
     - m/s
   * -
     - ``wind_direction_deg``
     - Wind direction (FROM bearing); sets the upwind axis.
     - deg
   * -
     - ``sea_temperature_c`` / ``salinity_psu``
     - Water properties (feed the seawater permittivity).
     - degC, PSU
   * -
     - ``inverse_wave_age``
     - Wave age Omega_c (0.84 fully developed, up to 5 young).
     - --
   * - **Inputs** (``compute_rcs``)
     - ``geometry`` / ``wavelength_m`` / ``illuminated_area_m2``
     - Bistatic geometry, carrier wavelength, and the clutter area to scatter over.
     - --, m, m^2
   * - **Outputs**
     - ``TargetRcs.rcs_m2``
     - Total RCS per sample.
     - m^2
   * -
     - ``TargetRcs.info``
     - Breakdown: coherent / incoherent / Bragg RCS, reflectivity, slopes, areas.
     - --

Detailed Module Description
---------------------------

The two contributions
^^^^^^^^^^^^^^^^^^^^^^

The total cross-section per sample is the sum

.. math::
   :label: ot-total

   \sigma = \sigma_\mathrm{coherent} + \sigma_\mathrm{incoherent}.

Both use the **circular reflectivity** :math:`|R_\mathrm{RL}|^2` of seawater (the RHCP-to-LHCP power
reflectivity from the Klein-Swift permittivity at the carrier frequency, water temperature and salinity).

Coherent (calm-water specular)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the mean sea surface were perfectly smooth, the reflection would be a clean specular one. The cross-section that
reproduces that two-ray reflected power through the radar equation is

.. math::
   :label: ot-coherent

   \sigma_\mathrm{coherent} = 4\pi\,|R_\mathrm{RL}|^2
   \left(\frac{R_t R_r}{R_t + R_r}\right)^2 D^2\, e^{-\kappa},

reduced by the **Rayleigh roughness loss**

.. math::
   :label: ot-kappa

   \kappa = \left(\frac{4\pi\,\sigma_h\,\cos\left(\beta/2\right)}{\lambda}\right)^2,

where :math:`\sigma_h` is RMS surface height, :math:`\beta` the bistatic angle (so :math:`\beta/2` is the
local specular incidence), and :math:`\lambda` the wavelength. The configured value is significant wave
height :math:`H_s`; the implementation uses the Gaussian-sea approximation
:math:`\sigma_h \simeq H_s/4`. For normal sea states :math:`\kappa` is large and this term collapses to
almost nothing; it matters mainly for a very smooth surface. It is applied only where the mean surface is
the specular reflector.

The **spherical-Earth divergence** :math:`D^2` accounts for the convex sea surface defocusing the
specular bundle relative to a flat mirror:

.. math::
   :label: ot-divergence

   D^2 = \left[\left(1 + \frac{2R_\mathrm{red}}{a_e\cos\left(\beta/2\right)}\right)
   \left(1 + \frac{2R_\mathrm{red}\cos\left(\beta/2\right)}{a_e}\right)\right]^{-1},
   \qquad R_\mathrm{red} = \frac{R_t R_r}{R_t + R_r},

with mean Earth radius :math:`a_e`. The two factors are the Coddington astigmatic mirror radii
:math:`a_e\cos\theta` in the plane of incidence and :math:`a_e/\cos\theta` across it; a flat Earth
(:math:`a_e \to \infty`) recovers :math:`D = 1`, and grazing incidence drives :math:`D \to 0`. For a
700 km receiver orbit this costs about 1.7 dB of coherent power. It is exposed in the diagnostics as
``divergence_factor_sq``.

Incoherent (rough surface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rough surface scatters diffusely. Using geometric optics over the wave-slope statistics, the
cross-section per unit area (the NRCS) is

.. math::
   :label: ot-incoherent

   \sigma^0_\mathrm{GO} = \pi\,|R_\mathrm{RL}|^2
   \left(\frac{|\mathbf{q}|}{q_z}\right)^4 P(s_u, s_c),

where :math:`\mathbf{q}` is the scattering vector (the change in propagation direction), :math:`q_z` its
part along the surface normal, and :math:`P(s_u, s_c)` is the probability of the exact surface
slope that would reflect the transmitter into the receiver. Only waves longer than about three radar
wavelengths act as specular facets at L band, so :math:`P` is a 2-D Gaussian over the **large-scale**
slope variances: the Elfouhaily spectrum integrated up to the two-scale division wavenumber
:math:`k_\mathrm{radar}/3` (Brown 1978; the same division the two-scale Bragg term uses), split by the
Cox-Munk upwind/crosswind anisotropy ratio. Using the full optical Cox-Munk variance would count
sub-wavelength ripples as mirrors; at L1 and 7 m/s that understates the glint-peak NRCS by about 3 dB
and overstates the off-specular GO skirt. Multiplying by the illuminated area gives the glistening-zone
return.

When patch integration is disabled and the footprint contains the complete glint, the implementation uses
a peak-equivalent area derived from the local slope-to-surface Jacobian. With reduced range
:math:`R_\mathrm{red}=R_tR_r/(R_t+R_r)`,

.. math::
   :label: ot-glistening-area

   dA=4R_\mathrm{red}^2\,ds_x\,ds_y,\qquad
   A_\mathrm{eff}=\frac{4R_\mathrm{red}^2}{P(0,0)}
   =8\pi R_\mathrm{red}^2\sqrt{\sigma_u^2\sigma_c^2}.

Thus :math:`\sigma^0_\mathrm{peak}A_\mathrm{eff}` preserves the complete first-order geometric-optics
integral. If the supplied footprint is smaller, the scalar fallback caps this effective area; patch
integration is preferable because it evaluates range, gain and NRCS over the actual footprint.

Bragg backscatter
^^^^^^^^^^^^^^^^^

In the backscatter direction at oblique incidence the geometric-optics term collapses (almost no facet is
tilted steeply enough to bounce the signal straight back). There the return comes from **Bragg** resonance
with small wind ripples, computed by the two-scale model in :doc:`../functions/ocean_physics`. It is
switched on only near backscatter (bistatic angle below ~5 deg) and at oblique incidence (at or above
20 deg), so the forward/specular path is left exactly unchanged:

.. math::
   :label: ot-bragg-gate

   \text{Bragg active} \iff \beta < 5^\circ \;\text{and}\; \theta_i \ge 20^\circ.

The incidence floor is **divergence protection**, not calibration: first-order Bragg grows without bound
toward nadir (:math:`\sigma^0 \sim B/(4\sin^4\theta)` as the resonant wavenumber slides into the
energetic long-wave spectrum), and the tilt average leaks that growth through mirror-ward-tilted facets
well above nadir. Below the gate the quasi-specular geometric-optics term owns the return. Lowering the
gate requires first restricting the facet-local Bragg wavenumber inside the tilt average
(:math:`k_B(\theta_\mathrm{local}) \ge k_\mathrm{radar}/3`, the same two-scale division), a refinement
deferred to future work.

The clutter area
^^^^^^^^^^^^^^^^

This model is **area-agnostic**: it multiplies an NRCS by whatever ``illuminated_area_m2`` the link budget
hands it. That area is the antenna footprint by default, one surface patch under patch integration, or a
first-order :doc:`delay-Doppler cell <../functions/delay_doppler>` when that model is on. At exact
specular geometry that first-order cell is singular and the current implementation uses the footprint
cap, not a small second-order cell.

API reference
^^^^^^^^^^^^^

.. autoclass:: object_definitions.OceanTarget_def.OceanTarget
   :members:

Module Assumptions and Limitations
----------------------------------

- **Coherent term gated to specular.** The specular reflection is only added where the surface itself is the
  specular reflector; it is zero off-specular, and collapses for any realistic wave height. Its
  spherical-Earth divergence uses the mean Earth radius, not the local curvature of the ellipsoid.
- **Hard Bragg gating.** Bragg is added only for bistatic angle below 5 deg and incidence at least 20 deg.
  The incidence floor is divergence protection for the near-nadir blow-up of first-order Bragg (see
  above), so parameter sweeps show a step at the gate; the smooth facet-local gate that would replace it
  is a recorded refinement. Out-of-plane polarisation rotation is not included.
- **Sea-state ranges.** Wind speed > 0; wave age between 0.84 and 5.0; temperature and salinity in the
  Klein-Swift range. Out-of-range inputs raise.
- **Area comes from the link budget.** The absolute diffuse power depends on the illuminated area supplied
  from outside (footprint, patch, or DDM cell), not chosen here.
- **Empirical-model scope.** The implementation combines Klein-Swift, Cox-Munk, Elfouhaily, geometric
  optics and a gated two-scale Bragg term. It has not been calibrated end-to-end against a particular
  GNSS-R instrument or collocated ocean dataset.

Validation and Tests
--------------------

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Test group
     - What it checks
   * - ``TestOceanTargetConstruction``
     - Inputs validated; ``from_settings`` round-trips the ``[OCEAN]`` fields.
   * - ``TestOceanTargetComputeRcs``
     - Coherent term dominates for calm/smooth water and collapses for a rough sea; the coherent term
       carries the spherical-Earth divergence :math:`D^2` (closed-form anchor at rtol :math:`10^{-12}`);
       the glistening area grows with wind and is capped by a supplied illuminated area; the roughness
       loss grows with wave height, using :math:`\sigma_h=H_s/4`; the uncapped peak-equivalent area
       preserves the analytic geometric-optics integral.
   * - ``TestOceanTargetComputeRcs`` (Bragg)
     - Bragg is zero at forward specular and near nadir, dominates at oblique backscatter, and uses the
       supplied illuminated area.
   * - ``TestComputeReflectionLinkBudgetIntegration``
     - The default target matches an explicit ``OceanTarget``; a custom target replaces the ocean RCS.

These tests establish equations, limiting cases and broad physical trends. They are not an external
cross-validation against CYGNSS, Aquarius, or measured DDMs; see :doc:`../../validation`.

User Guide
----------

The sea state comes from the ``[OCEAN]`` section of the moving-target configuration:

.. list-table:: ``[OCEAN]`` keys
   :widths: 36 50 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``ocean_wave_height_m``
     - Significant wave height.
     - m
   * - ``ocean_wind_speed_m_s``
     - Wind speed at 10 m.
     - m/s
   * - ``ocean_wind_direction_deg``
     - Wind direction (FROM bearing, clockwise from north).
     - deg
   * - ``ocean_sea_temperature_c``
     - Sea temperature.
     - degC
   * - ``ocean_salinity_psu``
     - Salinity.
     - PSU
   * - ``ocean_inverse_wave_age``
     - Wave age Omega_c (0.84 fully developed).
     - --

The link budget builds the target automatically with ``OceanTarget.from_settings(...)``; you rarely
construct it by hand.

References
----------

- Zavorotny, V. & Voronovich, A., "Scattering of GPS signals from the ocean with wind remote sensing
  application", *IEEE TGRS* 38(2), 2000.
- Klein, L. & Swift, C., "An improved model for the dielectric constant of sea water at microwave
  frequencies", *IEEE Trans. Antennas Propag.* 25(1), 1977.
- Cox, C. & Munk, W., "Measurement of the roughness of the sea surface from photographs of the sun's
  glitter", *J. Opt. Soc. Am.* 44(11), 1954.
- Brown, G., "Backscattering from a Gaussian-distributed perfectly conducting rough surface", *IEEE
  Trans. Antennas Propag.* 26(3), 1978 (two-scale division wavenumber).
- Beckmann, P. & Spizzichino, A., *The Scattering of Electromagnetic Waves from Rough Surfaces*,
  Pergamon, 1963 (spherical-Earth divergence factor).
