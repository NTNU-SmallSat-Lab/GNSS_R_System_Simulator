Monte Carlo feasibility runs
============================

``SRC/run_monte_carlo.py`` repeats the moving-target analysis many times with randomised conditions and
reports how often a ship would be detected. It answers *whether* and *under which receiver settings* a
detection method is feasible, rather than what happens on one particular route.

What is held fixed and what is drawn
------------------------------------

The platform comes from the INI and is the same in every trial: the LEO receiver orbits, the antenna,
the GNSS constellations and bands, the reflection mode (``[REFLECTION].reflection_type``), the ship size
and speed, and the link-budget losses. Each trial draws, from its own seeded random stream:

- the ship **start position** (on water, inside ``roi_bbox``) and **heading**; the route length is
  ``speed x window_hours``, so the ship sails for one window;
- an **epoch offset** added to ``[SIMULATION SETTINGS].startTime``;
- the **sea state** (wind speed and direction, significant wave height, inverse wave age);
- the **receiver settings**: coherent integration time :math:`T_\mathrm{coh}`, incoherent looks
  :math:`M` and system noise temperature :math:`T_\mathrm{sys}`.

Every quantity has a ``vary_*`` switch; a switched-off quantity keeps its INI value (the heading then
defaults to 90 deg, i.e. due east).

Each trial runs the full :doc:`Documentation/object_definitions/moving_target_detection` pipeline and
keeps its **best** detectability metric over all footprint-overlap events:

- **backscatter** -- the best ship SINR, :math:`\max\,P_\mathrm{ship}/(P_\mathrm{ocean}+N_\mathrm{eff})`;
- **specular** -- the best forward-shadow change over noise, :math:`\max\,|\Delta P_r|/N_\mathrm{eff}`.

A trial counts as a detection when that best metric reaches ``[LINK BUDGET].detection_sinr_threshold_db``.
The detection probability :math:`P_\mathrm{d}` is the fraction of detected trials. Trials without any
footprint overlap have zero opportunities and count as not detected; the summary also reports the
detection probability given at least one opportunity.

The [MONTE CARLO] section
-------------------------

The harness reads the scenario INI plus one extra section, ``[MONTE CARLO]``. ``Configuration.ini`` does
not contain it; add it to a copy of the file (for example ``Configuration_mc.ini``). All keys are
required.

.. list-table::
   :widths: 36 50 14
   :header-rows: 1

   * - Key
     - Meaning
     - Unit
   * - ``n_trials`` / ``seed``
     - Number of trials; seed of the random draws (equal seeds reproduce equal draws).
     - --
   * - ``window_hours``
     - Simulated time window per trial.
     - h
   * - ``vary_heading``
     - Draw the heading uniformly in [0, 360) deg; otherwise 90 deg.
     - --
   * - ``vary_position`` / ``roi_bbox``
     - Draw the start position on water inside ``[min_lon, max_lon, min_lat, max_lat]``; otherwise
       ``[MOVING TARGET].start_pos``.
     - deg
   * - ``vary_epoch`` / ``epoch_window_hours``
     - Draw an epoch offset uniformly in :math:`\pm` ``epoch_window_hours``.
     - h
   * - ``vary_sea_state``
     - Draw the sea state uniformly from the four ranges below; otherwise ``[OCEAN]``.
     - --
   * - ``wind_speed_range_m_s`` / ``wave_height_range_m``
     - ``[min, max]`` wind speed and significant wave height.
     - m/s, m
   * - ``wind_direction_range_deg`` / ``inverse_wave_age_range``
     - ``[min, max]`` wind direction and inverse wave age.
     - deg, --
   * - ``vary_coherent_integration`` / ``coherent_integration_time_s_values``
     - Draw :math:`T_\mathrm{coh}` from the list; otherwise ``[LINK BUDGET]``.
     - s
   * - ``vary_incoherent_looks`` / ``num_incoherent_looks_values``
     - Draw :math:`M` from the list; otherwise ``[LINK BUDGET]``.
     - --
   * - ``vary_system_noise_temperature`` / ``system_noise_temperature_k_range``
     - Draw :math:`T_\mathrm{sys}` uniformly from ``[min, max]``; otherwise ``[LINK BUDGET]``.
     - K

Example, for open Norwegian waters over one week of epochs:

.. code-block:: ini

   [MONTE CARLO]
   n_trials = 500
   seed = 42
   window_hours = 6
   vary_heading = true
   vary_position = true
   roi_bbox = [0.0, 18.0, 62.0, 71.0]
   vary_epoch = true
   epoch_window_hours = 168
   vary_sea_state = true
   wind_speed_range_m_s = [4.0, 16.0]
   wave_height_range_m = [0.5, 4.5]
   wind_direction_range_deg = [0.0, 360.0]
   inverse_wave_age_range = [0.84, 2.0]
   vary_coherent_integration = true
   coherent_integration_time_s_values = [0.001, 0.01, 0.1, 1.0]
   vary_incoherent_looks = true
   num_incoherent_looks_values = [1, 10, 100, 1000]
   vary_system_noise_temperature = true
   system_noise_temperature_k_range = [150.0, 400.0]

Running, resuming and re-plotting
---------------------------------

Run from the repository root:

.. code-block:: bash

   python SRC/run_monte_carlo.py Configuration_mc.ini --output-dir OUTPUT/mc_run1
   python SRC/run_monte_carlo.py Configuration_mc.ini 50 --output-dir OUTPUT/mc_quick   # override n_trials
   python SRC/run_monte_carlo.py Configuration_mc.ini aggregate --output-dir OUTPUT/mc_run1

- Trials run in parallel on all but one CPU core. Each trial runs in its own process under a wall-clock
  limit (``--trial-timeout-s``, default 600 s); a trial that exceeds it, or fails, is recorded with an
  error and counts as zero opportunities instead of stopping the study.
- Every finished trial is appended to ``mc_<mode>_trials.jsonl`` immediately. Running the same command
  again with the same ``--output-dir`` **resumes**: trial indices already in the file are skipped. Use a
  fresh directory whenever the scenario or the ``[MONTE CARLO]`` section changes.
- ``aggregate`` rebuilds the summary and all figures from the saved JSONL without running any trial.

Outputs
-------

All files are written to ``--output-dir`` (default ``OUTPUT/``); ``<mode>`` is ``backscatter`` or
``specular``.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Content
   * - ``mc_<mode>_trials.jsonl``
     - One JSON record per trial: the draws, the number of opportunities, the best metric, the detection
       flag, any error, and the paired powers of every footprint-overlap event.
   * - ``mc_<mode>_results.json``
     - The summary (:math:`P_\mathrm{d}`, detection probability given an opportunity, opportunity rate,
       mean opportunities per trial, error count) and all trial records.
   * - ``mc_<mode>_distribution.png``
     - Histogram of the best metric per trial, with the detection threshold.
   * - ``mc_<mode>_pd_vs_tcoh.png`` / ``mc_<mode>_pd_vs_looks.png``
     - :math:`P_\mathrm{d}` per drawn :math:`T_\mathrm{coh}` and per drawn :math:`M`.
   * - ``mc_<mode>_metric_vs_tsys.png``
     - Best metric against :math:`T_\mathrm{sys}`, coloured by wind speed.
   * - ``mc_<mode>_received_power.png``
     - Every footprint-overlap event: ship power (backscatter) or forward-shadow magnitude (specular)
       against ocean power, with an equal-power line.
   * - ``mc_backscatter_scr.png`` / ``mc_specular_shadow_fraction.png``
     - Histogram of :math:`10\log_{10}(P_\mathrm{ship}/P_\mathrm{ocean})` or
       :math:`10\log_{10}(|\Delta P_r|/P_\mathrm{ocean})` over all events.

The event figures use every finite, positive event, including events below the threshold, so a trial
with many overlaps contributes many samples. They show event distributions, not detection
probabilities. Trial records without saved event powers are left out of these figures, and the
aggregation reports how many were excluded.

Interpreting the results
------------------------

:math:`P_\mathrm{d}` here is the fraction of trials whose deterministic best metric crosses a user
threshold. It is not a calibrated probability of detection: the model has no false-alarm statistic,
speckle or fading (see :doc:`Documentation/object_definitions/moving_target_detection` and
:doc:`validation`). Use the runs to compare configurations and to find the receiver settings at which a
method becomes feasible.

API reference
-------------

.. automodule:: run_monte_carlo
   :members: parse_mc_section, draw_trials, run_trials_timeout_safe, summarize, make_plots, aggregate_and_report, main

.. automodule:: functions.mc_event_plots
   :members: serialize_power_events, make_event_plots
