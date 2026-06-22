Link budget (``functions.link_budget``)
=======================================

.. module:: functions.link_budget

Executive Summary
-----------------

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

.. list-table:: Consumed / produced by :func:`compute_reflection_link_budget`
   :widths: 22 78
   :header-rows: 1

   * - Role
     - Quantity
   * - Consumes
     - Tx / Rx ECI positions and Rx ECI velocities ``(3, n)``; reflection points ``(n, 3)``;
       incidence angle ``(n,)``; an :class:`~object_definitions.Rx_antenna_def.RxAntenna` (or ``None``
       for isotropic 0 dBi); carrier frequency [Hz]; transmit EIRP [dBW]; a
       :class:`BistaticLinkBudgetSettings`; optionally the attitude quaternion ``q_sat``, a custom
       :class:`~object_definitions.Target_def.Target`, ECI surface normals, and (for the delay-Doppler
       cell) the Tx velocities and the band chip length.
   * - Produces
     - A ``dict`` of arrays: ``received_power_w`` / ``_dbw`` / ``_dbm``; one-way and summed FSPL;
       ``bistatic_spreading_loss_db``; ``rx_gain_dbi``; ``rcs_m2``; the thermal-noise block
       (``noise_power_w``, ``cn0_db_hz``, ``snr_rf_db``, ...); the geometry angles; ``target_info``
       (the target's RCS breakdown); and ``delay_doppler_cell`` when that area model is enabled.

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

   \cos\theta_i = -\,\hat{n}_{tx2s}\cdot\hat{n}_{surf}, \qquad
   \cos\theta_s = \hat{n}_{s2rx}\cdot\hat{n}_{surf}, \qquad
   \cos\beta   = -\,\hat{n}_{tx2s}\cdot\hat{n}_{s2rx}

The **bistatic angle** :math:`\beta` is 0 deg for backscatter, :math:`2\theta_i` for the specular mirror
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

   A_\mathrm{fp} = \frac{4\pi}{G_r}\,\frac{R_r^2}{\cos\theta_i}

The cosine is clipped to ``[0.05, 1.0]`` to bound the grazing-incidence blow-up. This area is then passed
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
in :math:`B` and deliberately **excludes** the code-despreading / coherent-integration processing gain
(that gain is a separate, receiver-owned term). Area coverage supplies no noise settings, so the noise
block is emitted as NaN and the geometric path never depends on a floor it does not model.

API reference
^^^^^^^^^^^^^

.. autoclass:: functions.link_budget.BistaticLinkBudgetSettings

.. autoclass:: functions.link_budget.BistaticGeometry
   :no-members:

.. autofunction:: functions.link_budget.build_bistatic_geometry

.. autofunction:: functions.link_budget.compute_reflection_link_budget

.. autofunction:: functions.link_budget.compute_bistatic_link_budget

Module Assumptions and Limitations
----------------------------------

- **Receiver-side losses only.** ``system_losses_db`` and ``polarization_losses_db`` lump into :math:`L`;
  the transmit EIRP is supplied per ``(constellation, band)`` from :data:`Constants.GNSS_EIRP_DBW`.
- **Surface normals default to geocentric.** Unless ``surface_normals_eci`` is supplied, the reflection
  position is used as the normal, which is exact only for a sphere. The elliptical-Earth path supplies
  WGS-84 ellipsoid normals (see :func:`ellipsoid_surface_normals_eci`).
- **Simplified point-target radar equation.** Distributed-target physics enters only through :math:`\sigma`
  and the illuminated area; multiple scattering and coherent speckle statistics are not modelled here.
- **SNR is pre-correlation.** ``snr_rf_db`` omits the despreading / integration gain, so it understates
  the achievable detection SNR by that processing gain. This is the dominant reason the end-to-end system
  is currently noise-limited.
- **Grazing clip.** :math:`\cos\theta_i` is clipped to ``[0.05, 1.0]`` in the footprint area.
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
     - Zenith geometry gives all angles 0; the mirror geometry gives :math:`\beta = 2\theta_i`; ranges;
       NaN propagation and shape validation.
   * - ``tests/test_link_budget.py``
     - ``compute_rx_look_direction_antenna_frame`` NaN guards and match to the manual rotation chain;
       the noise model is exactly :math:`kTB`; SNR and :math:`C/N_0` derivations; doubling the bandwidth
       drops SNR by 3 dB but leaves :math:`C/N_0` unchanged; non-positive noise settings raise; the
       area-coverage path yields NaN noise but finite power.
   * - ``tests/test_link_budget_sanity.py``
     - End-to-end through :class:`~object_definitions.OceanTarget_def.OceanTarget`: finite Pr across
       incidence; the per-constellation EIRP plumbing produces the exact 5 dB Galileo-vs-GPS shift;
       forward specular exceeds oblique backscatter; Pr sits in a CYGNSS-class envelope
       (-160 .. -110 dBm); the Bragg :math:`\sigma_0` sits in the L-band envelope.
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
     - Coherent integration time :math:`T_\mathrm{coh}` [s]; sets the cell Doppler bin :math:`1/T_\mathrm{coh}`.
       Required when the cell is enabled.

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
