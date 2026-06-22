Link budget
===========

.. module:: functions.link_budget

Summary
-------

``functions.link_budget`` is the integrating module of the GNSS-R signal chain: it turns a set of
transmitter / receiver / reflection-point states into a received power, a thermal-noise floor, and the
derived figures of merit (:math:`C/N_0`, SNR). It owns three responsibilities:

#. **Geometry** -- :func:`build_bistatic_geometry` reduces the raw Tx, Rx and surface states to the
   per-sample directions, ranges and angles that every downstream physics model needs
   (:class:`BistaticGeometry`).
#. **Scattering hand-off** -- :func:`compute_reflection_link_budget` asks a
   :class:`~object_definitions.Target_def.Target` (by default
   :class:`~object_definitions.OceanTarget_def.OceanTarget`) for the radar cross section at that
   geometry, sizing the illuminated area for distributed (area-extensive) returns.
#. **The link equation** -- :func:`compute_bistatic_link_budget` evaluates the bistatic radar equation
   and the receiver noise model, returning a fully populated link-budget dictionary.

The module carries **receiver-side losses only**. The GNSS transmit EIRP is a property of the
transmitting signal and lives in :data:`Constants.GNSS_EIRP_DBW` per ``(constellation, band)``; it is
passed in as ``tx_eirp_dbw`` and is never a link-budget configuration knob. Both analysis modes consume
this module: :class:`~object_definitions.Reflection_def.Reflection` and
:class:`~object_definitions.BackScattPoint_def.BRP` call it per reflection, and
:class:`~object_definitions.MovingTargetDetection_def.MovingTargetDetection` calls it per band.

Interface
---------

The public surface is two data classes plus three functions. Vectors follow the simulator convention:
**satellite state vectors are** ``(3, n)`` **and reflection points are** ``(n, 3)``.

.. list-table:: Inputs / outputs of :func:`compute_reflection_link_budget`
   :widths: 12 28 46 14
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs**
     - ``tx_positions_eci``
     - Transmitter ECI positions ``(3, n)``.
     - m
   * -
     - ``rx_positions_eci``
     - Receiver ECI positions ``(3, n)``.
     - m
   * -
     - ``rx_velocities_eci``
     - Receiver ECI velocities ``(3, n)``.
     - m/s
   * -
     - ``reflection_positions_eci``
     - Reflection points ``(n, 3)``.
     - m
   * -
     - ``incidence_deg``
     - Incidence angle ``(n,)``.
     - deg
   * -
     - ``antenna``
     - :class:`~object_definitions.Rx_antenna_def.RxAntenna` (``None`` => isotropic 0 dBi).
     - --
   * -
     - ``frequency_hz``
     - Carrier frequency.
     - Hz
   * -
     - ``tx_eirp_dbw``
     - Transmit EIRP for this (constellation, band).
     - dBW
   * -
     - ``settings``
     - :class:`BistaticLinkBudgetSettings` (Rx-system + ocean settings).
     - --
   * -
     - ``q_sat``
     - Attitude quaternion ``[w, x, y, z]`` (optional).
     - --
   * -
     - ``target``
     - Custom :class:`~object_definitions.Target_def.Target` RCS model (optional; default OceanTarget).
     - --
   * -
     - ``surface_normals_eci``
     - ECI surface normals (optional; geocentric if omitted).
     - --
   * -
     - ``tx_velocities_eci`` / ``chip_length_m``
     - Tx velocities and band chip length; required only for the delay-Doppler cell.
     - m/s, m
   * - **Outputs**
     - ``received_power_w`` / ``_dbw`` / ``_dbm``
     - Received power.
     - W, dBW, dBm
   * -
     - ``fspl_tx_db`` / ``fspl_rx_db`` / ``fspl_sum_db``
     - One-way and summed free-space path loss.
     - dB
   * -
     - ``bistatic_spreading_loss_db``
     - Bistatic spreading term.
     - dB
   * -
     - ``rx_gain_dbi``
     - Steered receiver gain along the look direction.
     - dBi
   * -
     - ``rcs_m2``
     - Target RCS used in the link equation.
     - m^2
   * -
     - ``noise_power_w`` / ``cn0_db_hz`` / ``snr_rf_db``
     - Thermal-noise block (moving-target mode; NaN for area coverage).
     - W, dB-Hz, dB
   * -
     - ``incidence_deg`` / ``scattering_deg`` / ``bistatic_deg``
     - Geometry angles.
     - deg
   * -
     - ``target_info``
     - The target's RCS breakdown (coherent / incoherent, etc.).
     - --
   * -
     - ``delay_doppler_cell``
     - Cell-area diagnostics, present when that area model is enabled.
     - --

.. note::

   The illuminated-area model has three modes, selected by the settings: the steered-beam **footprint**
   (default), one **delay-Doppler resolution cell** (``delay_doppler_cell_enabled``; see
   ``functions.delay_doppler``), or a **patch-integrated** footprint (``ocean_patch_integration_enabled``).
   The first two are mutually exclusive.

Detailed Module Description
---------------------------

Bistatic geometry
^^^^^^^^^^^^^^^^^

For each sample, :func:`build_bistatic_geometry` forms the incoming and outgoing propagation unit
vectors and the outward surface normal:

.. math::
   :label: lb-dirs

   \hat{n}_{tx2s} = \frac{S - T}{\lVert S - T \rVert}, \qquad
   \hat{n}_{s2rx} = \frac{R - S}{\lVert R - S \rVert}, \qquad
   \hat{n}_{surf} = \frac{N}{\lVert N \rVert}

with Tx, Rx and surface positions :math:`T, R, S` and the (unnormalised) surface normal :math:`N`. The
slant ranges are :math:`R_t = \lVert S - T \rVert` and :math:`R_r = \lVert R - S \rVert`, and the three
angles follow from dot products:

.. math::
   :label: lb-angles

   \cos\left(\theta_i\right) = -\,\hat{n}_{tx2s}\cdot\hat{n}_{surf}, \qquad
   \cos\left(\theta_s\right) = \hat{n}_{s2rx}\cdot\hat{n}_{surf}, \qquad
   \cos\left(\beta\right)   = -\,\hat{n}_{tx2s}\cdot\hat{n}_{s2rx}

The **bistatic angle** :math:`\beta` is 0 deg for backscatter, :math:`2\theta_i` for the specular
geometry, and 180 deg for pure forward scatter. Any invalid sample (NaN input, zero range, zero-length
normal) propagates as NaN rather than raising, so a single bad reflection never poisons a batch.

The link equation
^^^^^^^^^^^^^^^^^

:func:`compute_bistatic_link_budget` evaluates the simplified bistatic radar equation,

.. math::
   :label: lb-radar

   P_r = \frac{\mathrm{EIRP}\;G_r\,\lambda^2\,\sigma}{(4\pi)^3\,R_t^2\,R_r^2\,L}

where :math:`\mathrm{EIRP} = P_t G_t` is the transmit EIRP [W], :math:`G_r` the receiver gain along the
look direction, :math:`\lambda = c/f` the carrier wavelength, :math:`\sigma` the target RCS [m^2] supplied
by the :class:`~object_definitions.Target_def.Target`, and :math:`L` the lumped receiver-side loss
(``system_losses_db + polarization_losses_db``). The two one-way free-space path losses are reported
separately,

.. math::
   :label: lb-fspl

   \mathrm{FSPL}_{1\text{-way}} = 20\log_{10}\!\left(\frac{4\pi R}{\lambda}\right),

for the Tx->surface and surface->Rx legs.

Illuminated area and the distributed target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a distributed (area-extensive) return such as the ocean, the RCS scales with an illuminated area.
With an antenna present, the receiver auto-steers to the reflection, so ``rx_gain_dbi`` is the steered-beam
peak directivity and the beam solid angle :math:`4\pi/G_r` projects to a ground footprint:

.. math::
   :label: lb-footprint

   A_\mathrm{fp} = \frac{4\pi}{G_r}\,\frac{R_r^2}{\cos\left(\theta_s\right)}.

The projection angle is the **receiver scattering angle** :math:`\theta_s`, not the transmitter
incidence angle :math:`\theta_i`. The cosine is clipped to ``[0.05, 1.0]`` to bound the grazing-angle
blow-up. This area is then passed
to ``Target.compute_rcs`` so the target owns how it is used. When ``delay_doppler_cell_enabled``, the area
is replaced by a single delay-Doppler resolution cell (capped at :math:`A_\mathrm{fp}`); when
``ocean_patch_integration_enabled``, the ocean clutter is integrated patch-by-patch over the footprint and
converted to an equivalent centre RCS. With ``antenna=None`` no area is imposed here and the target falls
back to its own geometric area.

Thermal noise and C/N0
^^^^^^^^^^^^^^^^^^^^^^^

When noise settings are supplied (moving-target mode), the module forms the ideal thermal floor

.. math::
   :label: lb-noise

   N_0 = k\,T_\mathrm{sys}, \qquad N = k\,T_\mathrm{sys}\,B, \qquad
   C/N_0 = \frac{P_r}{N_0}, \qquad \mathrm{SNR} = \frac{P_r}{N}

with Boltzmann constant :math:`k`, system noise temperature :math:`T_\mathrm{sys}` and noise bandwidth
:math:`B`. :math:`C/N_0` is the bandwidth-independent figure; ``snr_rf_db`` is the **pre-correlation** SNR
in :math:`B` and deliberately **excludes** the code-despreading / coherent-integration processing gain. The
detector adds that gain when it forms the SINR (next subsection). Area coverage supplies no noise settings,
so the noise block is emitted as NaN and the geometric path never depends on a floor it does not model.

Coherent processing gain
^^^^^^^^^^^^^^^^^^^^^^^^^

The figures above are *pre-correlation*: they weigh the received power against the full-bandwidth thermal
floor :math:`N = k\,T_\mathrm{sys}\,B`. A GNSS receiver instead correlates the incoming signal against a
local replica of the transmitted spread-spectrum (PRN) code and integrates. This despreading lifts a
code-carrying signal out of the white thermal noise. :func:`coherent_processing_gain_linear` returns the
linear gain

.. math::
   :label: lb-procgain

   G_p = \underbrace{B\,T_\mathrm{coh}}_{\text{coherent}}\;\underbrace{\sqrt{M}}_{\text{incoherent}}

with coherent integration time :math:`T_\mathrm{coh}` (``coherent_integration_time_s``) and number of
incoherent looks :math:`M` (``num_incoherent_looks``). Coherent integration over :math:`T_\mathrm{coh}`
narrows the equivalent noise bandwidth to order :math:`1/T_\mathrm{coh}`, giving
:math:`B\,T_\mathrm{coh}`. The additional :math:`\sqrt{M}` represents the reduction in estimator
standard deviation from averaging independent power looks. The detector expresses both effects as a
single **detection-equivalent** noise scale

.. math::
   :label: lb-noise-eff

   N_\mathrm{eff} = \frac{N}{G_p} = \frac{k\,T_\mathrm{sys}}{T_\mathrm{coh}\,\sqrt{M}}

The bandwidth cancels algebraically. However, incoherent averaging does **not** reduce the mean thermal
noise power by :math:`\sqrt{M}`; it reduces the fluctuation of an estimator. Consequently
``noise_power_effective_w`` and the resulting threshold comparison are a sensitivity approximation, not
a literal post-detection mean-noise power or a probability-of-detection calculation. A rigorous detector
would specify the statistic, false-alarm probability and signal/noise distributions.

The gain acts on the **noise term only**. The ocean clutter carries the same PRN code as the target, so it
despreads with the same :math:`G_p`; the gain cancels in the signal-to-clutter ratio
:math:`P_\mathrm{ship}/P_\mathrm{ocean}` and merely lifts both returns above the thermal floor. The SINR
formed in :class:`~object_definitions.MovingTargetDetection_def.MovingTargetDetection` is therefore the
implemented detection metric

.. math::
   :label: lb-sinr

   \mathrm{SINR} = \frac{P_\mathrm{ship}}{P_\mathrm{ocean} + N_\mathrm{eff}}

so a long enough :math:`T_\mathrm{coh}` moves the system from noise-limited
(:math:`N_\mathrm{eff} > P_\mathrm{ocean}`) to clutter-limited (:math:`N_\mathrm{eff} < P_\mathrm{ocean}`):
noise- vs clutter-limited is a receiver-setting choice, not a fixed property.

:math:`T_\mathrm{coh}` is physically bounded by the navigation-data bit period and, over the sea, by the
surface decorrelation time (a few ms for the diffuse return). The model applies :math:`G_p` to the thermal
noise and leaves the clutter at full strength -- it assumes the clutter stays coherent over
:math:`T_\mathrm{coh}` (the conservative choice). A clutter return that decorrelates faster than
:math:`T_\mathrm{coh}` would gain less than a steady target, so a finer model would apply separate,
coherence-time-limited gains to the ship and the ocean; that differential-coherence discriminator is not
yet modelled.

API reference
^^^^^^^^^^^^^

.. autoclass:: functions.link_budget.BistaticLinkBudgetSettings

.. autoclass:: functions.link_budget.BistaticGeometry
   :no-members:

.. autofunction:: functions.link_budget.build_bistatic_geometry

.. autofunction:: functions.link_budget.compute_reflection_link_budget

.. autofunction:: functions.link_budget.compute_bistatic_link_budget

.. autofunction:: functions.link_budget.coherent_processing_gain_linear

Module Assumptions and Limitations
----------------------------------

- **Receiver-side losses only.** ``system_losses_db`` and ``polarization_losses_db`` lump into :math:`L`;
  the transmit EIRP is supplied per ``(constellation, band)`` from :data:`Constants.GNSS_EIRP_DBW`.
- **Surface normals default to geocentric.** Unless ``surface_normals_eci`` is supplied, the reflection
  position is used as the normal, which is exact only for a sphere. The elliptical-Earth path supplies
  WGS-84 ellipsoid normals (see :func:`ellipsoid_surface_normals_eci`).
- **Simplified point-target radar equation.** Distributed-target physics enters only through :math:`\sigma`
  and the illuminated area; multiple scattering and coherent speckle statistics are not modelled here.
- **Link-budget SNR is pre-correlation.** ``snr_rf_db`` and ``cn0_db_hz`` report the raw RF link, before
  any despreading / integration gain. The moving-target detector applies the coherent and incoherent
  sensitivity factor :math:`B\,T_\mathrm{coh}\,\sqrt{M}` in its threshold metric (see
  :func:`functions.link_budget.coherent_processing_gain_linear`). The :math:`\sqrt{M}` part is an
  estimator-variance approximation, not a physical reduction of mean noise power.
- **Approximate footprint.** :math:`4\pi/G_r` treats peak directivity as an equivalent beam solid angle
  and projects it with :math:`\cos\theta_s`, clipped to ``[0.05, 1.0]``. It is not an integration of the
  actual contour unless patch integration is enabled.
- **Incoherent-look metric.** The :math:`\sqrt{M}` term models sensitivity improvement through estimator
  variance. It must not be interpreted as a reduction in mean thermal noise power.
- **Isotropic fallback.** ``antenna=None`` assumes a 0 dBi receiver and imposes no footprint area.

Validation and Tests
--------------------

The module is regression-locked from pure geometry up to the end-to-end ocean physics:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Test module
     - What it pins
   * - ``tests/test_bistatic_geometry.py``
     - Zenith geometry gives all angles 0; the specular geometry gives :math:`\beta = 2\theta_i`; ranges;
       NaN propagation and shape validation.
   * - ``tests/test_link_budget.py``
     - ``compute_rx_look_direction_antenna_frame`` NaN guards and match to the manual rotation chain;
       the noise model is exactly :math:`kTB`; SNR and :math:`C/N_0` derivations; doubling the bandwidth
       drops SNR by 3 dB but leaves :math:`C/N_0` unchanged; non-positive noise settings raise; the
       area-coverage path yields NaN noise but finite power.
   * - ``tests/test_link_budget_sanity.py``
     - End-to-end through :class:`~object_definitions.OceanTarget_def.OceanTarget`: finite Pr across
       incidence; the per-constellation EIRP plumbing produces the exact 5 dB Galileo-vs-GPS shift;
       forward specular exceeds oblique backscatter; received power and Bragg :math:`\sigma_0` sit in
       deliberately broad plausibility envelopes. These are sanity bounds, not CYGNSS calibration.
   * - ``tests/test_link_budget_golden.py``
     - Golden-file regression against ``tests/golden/reference_link_budget.npz``; any change in geometry,
       ocean physics or the link equation flips it red. Regenerate intentionally via
       ``tests/golden/regenerate_link_budget.py``.

User Guide
----------

The receiver-system parameters come from the ``[LINK BUDGET]`` INI section; the ocean sea-state fields
come from ``[OCEAN]``. Both feed :class:`BistaticLinkBudgetSettings`. The loader is INI-strict (missing
required keys raise).

.. list-table:: ``[LINK BUDGET]`` keys
   :widths: 34 66
   :header-rows: 1

   * - Key
     - Meaning
   * - ``system_losses_db``
     - Lumped implementation loss (feedline, filter, ADC, correlator) [dB, >= 0].
   * - ``polarization_losses_db``
     - Receiver antenna polarization-mismatch loss (not the RHCP->LHCP reflection flip) [dB, >= 0].
   * - ``noise_bandwidth_hz``
     - Thermal-noise bandwidth :math:`B` [Hz] (moving-target only).
   * - ``system_noise_temperature_k``
     - Equivalent system noise temperature :math:`T_\mathrm{sys}` [K] (moving-target only).
   * - ``detection_sinr_threshold_db``
     - Ship detectability threshold for :math:`P_\mathrm{ship}/(P_\mathrm{ocean}+N)` [dB].
   * - ``ocean_patch_integration_enabled``
     - Integrate ocean clutter patch-by-patch over the footprint instead of one scalar area.
   * - ``ocean_patch_grid_size``
     - Cells per footprint axis for that integration [3 .. 101].
   * - ``delay_doppler_cell_enabled``
     - Use one delay-Doppler resolution cell as the distributed-clutter area (mutually exclusive with
       patch integration).
   * - ``coherent_integration_time_s``
     - Coherent integration time :math:`T_\mathrm{coh}` [s]; sets the cell Doppler bin :math:`1/T_\mathrm{coh}`
       and the coherent processing gain :math:`B\,T_\mathrm{coh}`. Required when the cell is enabled.
   * - ``num_incoherent_looks``
     - Independent incoherent looks :math:`M` (>= 1); applies a heuristic :math:`\sqrt{M}` detection
       sensitivity improvement, not a reduction in mean noise power.

A minimal call (isotropic receiver, GPS L1, area-coverage-style settings):

.. code-block:: python

   import numpy as np
   from functions.link_budget import BistaticLinkBudgetSettings, compute_reflection_link_budget

   settings = BistaticLinkBudgetSettings(
       system_losses_db=2.0,
       polarization_losses_db=0.5,
       ocean_wave_height_m=1.5,
       ocean_wind_speed_m_s=7.0,
       ocean_wind_direction_deg=270.0,
       ocean_sea_temperature_c=10.0,
       ocean_salinity_psu=35.0,
       ocean_inverse_wave_age=0.84,
       noise_bandwidth_hz=2.0e6,
       system_noise_temperature_k=290.0,
   )

   lb = compute_reflection_link_budget(
       tx_positions_eci=tx,             # (3, n) GNSS Tx ECI positions [m]
       rx_positions_eci=rx,             # (3, n) LEO Rx ECI positions [m]
       rx_velocities_eci=rx_vel,        # (3, n) LEO Rx ECI velocities [m/s]
       reflection_positions_eci=surf,   # (n, 3) reflection points [m]
       incidence_deg=inc,               # (n,)  incidence angle [deg]
       antenna=None,                    # isotropic 0 dBi
       frequency_hz=1.57542e9,          # GPS L1
       tx_eirp_dbw=27.0,                # from Constants.GNSS_EIRP_DBW
       settings=settings,
   )
   print(lb["received_power_dbm"], lb["cn0_db_hz"])

References
----------

- Zavorotny, V. & Voronovich, A., "Scattering of GPS signals from the ocean with wind remote sensing
  application", *IEEE TGRS* 38(2), 2000.
- Ruf, C. *et al.*, "A new paradigm in earth environmental monitoring with the CYGNSS small satellite
  constellation", *Scientific Reports* 8, 2018.
