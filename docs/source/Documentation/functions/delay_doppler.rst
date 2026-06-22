Delay-Doppler cell
==================

.. module:: functions.delay_doppler

Summary
-------

A GNSS-R receiver does not see the surface as one big patch. It compares the reflected signal against a
copy of the GNSS code (this sorts the surface by **delay**) and runs a Fourier transform over the
integration time (this sorts it by **Doppler**, i.e. by range-rate). The result is a Delay-Doppler Map
(DDM), and one pixel is the surface patch having both the selected delay and Doppler.

This module estimates that area locally from first derivatives. Away from singular geometries it can be
much smaller than the antenna footprint. At the exact specular point the first-order mapping is singular,
so this implementation deliberately falls back to the footprint area; it does **not** currently produce a
small specular DDM cell from second-order delay curvature.

Interface
---------

.. list-table:: Inputs / outputs of :func:`delay_doppler_cell_area`
   :widths: 12 28 46 14
   :header-rows: 1

   * -
     - Variable
     - Description
     - Unit
   * - **Inputs**
     - ``geometry``
     - :class:`~functions.link_budget.BistaticGeometry` -- per-sample directions and ranges.
     - --
   * -
     - ``tx_velocities_eci``
     - Transmitter ECI velocities ``(3, n)``.
     - m/s
   * -
     - ``rx_velocities_eci``
     - Receiver ECI velocities ``(3, n)``.
     - m/s
   * -
     - ``surface_positions_eci``
     - Surface positions ``(n, 3)``. When supplied, includes rigid Earth rotation; production calls
       supply it. ``None`` keeps the legacy stationary-in-ECI convention.
     - m
   * -
     - ``wavelength_m``
     - Carrier wavelength.
     - m
   * -
     - ``chip_length_m``
     - One code-chip path length, ``c / chip_rate``.
     - m
   * -
     - ``coherent_integration_time_s``
     - Coherent integration time ``T_coh``.
     - s
   * -
     - ``footprint_area_m2``
     - Upper-bound cap (antenna footprint); ``None`` means no cap.
     - m^2
   * - **Outputs**
     - ``cell_area_m2``
     - DDM cell area, capped at the footprint.
     - m^2
   * -
     - ``uncapped_cell_area_m2``
     - Cell area before the cap.
     - m^2
   * -
     - ``delay_width_m`` / ``doppler_width_m``
     - Cell width along the delay and Doppler directions.
     - m
   * -
     - ``delay_gradient`` / ``doppler_gradient_hz_m``
     - Steepness of the delay and Doppler fields on the surface.
     - --, Hz/m
   * -
     - ``abs_jacobian_det_hz_m``
     - ``|grad L x grad f_D|`` (the Jacobian size).
     - Hz/m
   * -
     - ``footprint_capped``
     - True where the cap (or a near-specular case) set the area to the footprint.
     - --

Detailed Module Description
---------------------------

Two fields on the surface
^^^^^^^^^^^^^^^^^^^^^^^^^^

Every point on the sea has a delay and a Doppler, set by two fields. The first is the total path length
the signal travels, transmitter to surface to receiver,

.. math::
   :label: dd-path

   L(\mathbf{r}) = R_t(\mathbf{r}) + R_r(\mathbf{r}),

which fixes the delay (delay is just :math:`L/c`). The second is the Doppler shift, set by the satellite
velocities relative to the rotating surface,

.. math::
   :label: dd-doppler

   f_D(\mathbf{r}) = \frac{1}{\lambda}\,\left[
   \hat{n}_{tx2s}\cdot(\mathbf{v}_{tx}-\mathbf{v}_{s})
   - \hat{n}_{s2rx}\cdot(\mathbf{v}_{rx}-\mathbf{v}_{s})\right],

with carrier wavelength :math:`\lambda` and
:math:`\mathbf{v}_s=\boldsymbol{\omega}_E\times\mathbf{r}`.

How fast the fields change across the surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To turn "one chip of delay" and "one Doppler bin" into an area, we need how quickly each field changes as
you move across the surface -- its gradient in the local tangent plane. For the path length this is the
**scattering vector**,

.. math::
   :label: dd-gradL

   \nabla L = \hat{n}_{tx2s} - \hat{n}_{s2rx},

and for the Doppler it comes from the parts of the satellite velocities that point across the line of
sight (not along it),

.. math::
   :label: dd-gradfd

   \nabla f_D = \frac{1}{\lambda}\left(
   \frac{\mathbf{v}_{tx,rel,\perp}}{R_t}
   + \frac{\mathbf{v}_{rx,rel,\perp}}{R_r}
   \boldsymbol{\omega}_E\times(\hat{n}_{tx2s}-\hat{n}_{s2rx})\right).

The final term is the spatial derivative of the rigid-rotation velocity. Both gradients are projected
onto the tangent plane (the part along the surface normal is dropped).

The cell area
^^^^^^^^^^^^^

One code chip is a step in path length of :math:`\delta L = c / R_\mathrm{chip}` (for GPS C/A,
:math:`R_\mathrm{chip} = 1.023` Mcps, so :math:`\delta L \approx 293` m), and one Doppler bin is
:math:`\delta f_D = 1 / T_\mathrm{coh}`. Because both gradients lie in the same flat tangent plane, the
size of the (delay, Doppler) -> (surface) mapping is just the size of their cross product,

.. math::
   :label: dd-jacobian

   |\det J| = |\nabla L \times \nabla f_D| \quad [\mathrm{Hz/m}],

and the area of the surface patch that falls into one chip-by-one-bin box is

.. math::
   :label: dd-area

   A_\mathrm{cell} = \frac{\delta L \; \delta f_D}{|\det J|} \quad [\mathrm{m}^2].

A longer integration time :math:`T_\mathrm{coh}` makes the Doppler bins narrower and the cell smaller.

The near-specular cap
^^^^^^^^^^^^^^^^^^^^^^

At the specular reflection point the path length is at its minimum, so :math:`\nabla L \to 0`, the Jacobian
goes to zero, and the formula blows up -- the iso-delay rings are genuinely far apart at the glint. The
same can happen when :math:`\nabla f_D \to 0`. In those cases the cell is capped at the antenna footprint
area, the largest patch the receiver could integrate. In backscatter mode the reflection point sits well
away from the delay minimum, so the cell is well-defined; the cap mainly matters in the forward/specular
direction.

API reference
^^^^^^^^^^^^^

.. autofunction:: functions.delay_doppler.delay_doppler_cell_area

Module Assumptions and Limitations
----------------------------------

- **First-order (linear) cell.** The area uses the first derivatives of the delay and Doppler fields. This
  is accurate away from the specular point but breaks down at the glint, where it is capped at the
  footprint. The exact near-specular cell (from the curvature of the path length) is a future refinement.
- **No target-to-cell membership test.** The moving-target pipeline first checks antenna-footprint
  overlap. It does not yet compute the ship's own delay and Doppler and prove that the target and the
  ocean reference point occupy the same DDM bin. Therefore DDM-cell clutter results are a sensitivity
  approximation, not a complete detector simulation.
- **Code chip sets the delay size.** For spread signals (BOC, CBOC, AltBOC) the base code-chip rate is
  used, which is the conservative (coarser delay, larger cell) choice. Chip rates live per
  ``(constellation, band)`` in :data:`Constants.GNSS_CHIP_RATE_HZ`.
- **No geometry change during integration.** The cell is computed at one instant; smearing from the
  satellites moving during :math:`T_\mathrm{coh}` is not modelled.
- **Cap needs a footprint.** Without a footprint area the near-specular cell is left as NaN rather than
  bounded.

Validation and Tests
--------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Test group
     - What it checks
   * - ``TestJacobianAgainstFiniteDifferences``
     - The analytic Jacobian and delay gradient match direct finite differences, including a rotating
       surface case.
   * - ``TestChipLength``
     - The GPS C/A chip is 293.05 m (``c / 1.023 MHz``).
   * - ``TestScaling``
     - The area grows with the chip length and shrinks with the integration time, exactly as the formula says.
   * - ``TestFootprintCap`` / ``TestDegenerateAndInvalid``
     - A small footprint caps the cell; a large one does not; a near-specular case collapses to the
       footprint; invalid geometry returns NaN.
   * - ``TestOrderOfMagnitude``
     - A representative non-specular geometry lands in a broad plausibility range. This is not external
       instrument validation.
   * - ``TestChipRateTable`` / ``TestSettingsValidation``
     - The chip-rate lookups work and every EIRP band has a chip rate; the cell setting requires a
       coherent time and is mutually exclusive with patch integration.

User Guide
----------

The cell is turned on in the moving-target link budget through ``[LINK BUDGET]``:

.. list-table::
   :widths: 34 50 16
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``delay_doppler_cell_enabled``
     - Use one DDM cell as the ocean-clutter area instead of the footprint.
     - boolean
   * - ``coherent_integration_time_s``
     - Coherent integration time ``T_coh``; sets the Doppler bin ``1 / T_coh``. Required when the cell is on.
     - s

When enabled, :func:`~functions.link_budget.compute_reflection_link_budget` calls this module and uses the
cell area as the distributed-clutter area. The chip length comes from the band's chip rate
(``c / chip_rate``). Check ``footprint_capped`` before interpreting a result: exact specular samples
normally use the footprint cap in the present first-order implementation.

References
----------

- Zavorotny, V. & Voronovich, A., "Scattering of GPS signals from the ocean with wind remote sensing
  application", *IEEE TGRS* 38(2), 2000.
- Clarizia, M. P. *et al.*, "Analysis of GNSS-R delay-Doppler maps from the UK-DMC satellite over the
  ocean", *Geophysical Research Letters* 36, 2009.
