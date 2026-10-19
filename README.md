# GNSS-R System Simulator

Preliminary-design simulator for GNSS-Reflectometry (GNSS-R) LEO mission studies. The code currently supports two analysis modes:

- **Area coverage**: counts contiguous reflection track visits (specular points or near-backscatter swaths) over a gridded region.
- **Moving-target detection**: follows an ocean target and evaluates specular forward-scatter or backscatter events, including link-budget diagnostics.

The geometry and numerical implementation are tested, but the ocean, antenna, ship-RCS, and detection
chain are not calibrated end-to-end against a flight instrument. Treat absolute received power and ship
detection as preliminary sensitivity results; the documentation's scientific assessment and validation
status identify the remaining work.

## Repository Layout

```text
SRC/
  main_simulation_areaCoverage.py      Area-coverage entry point
  main_simulation_movingTarget.py      Moving-target entry point
  functions/                           Shared geometry, link-budget, ocean, and helper functions
  object_definitions/                  Satellites, reflections, targets, coverage, detector, antenna
tests/                                 Pytest suite and golden-file references
DATA/                                  Input data, TLEs, coastline/ocean shapefiles, antenna helpers
OUTPUT/                                Generated plots, animations, and orbit views
docs/                                  Sphinx documentation
Configuration.ini                      Scenario for both analysis modes
simulator_settings.yaml                Engine/numerical settings (step size, pattern resolution)
```

Run commands from the repository root. Runtime paths are resolved from the current working directory.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,docs]"
```

The project is validated for Python 3.12.x. Newer Python minors may work, but scientific binary wheels and debugger integrations can lag.

Area coverage needs the Norwegian EEZ polygon, which is not bundled (see [Data Sources](#data-sources)). Download it from the Marine Regions web service (Marine Regions ID 5686) to the path `Configuration.ini` expects:

```powershell
python -c "import os, geopandas as gpd; os.makedirs('DATA/eez', exist_ok=True); gpd.read_file('https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=MarineRegions:eez&cql_filter=mrgid=5686&outputFormat=application/json').to_file('DATA/eez/eez_norge.shp')"
```

## Run Simulations

```powershell
python SRC/main_simulation_areaCoverage.py
python SRC/main_simulation_movingTarget.py
```

Both entry points read `Configuration.ini`; each ignores the sections that only the other mode uses.

`main_simulation_areaCoverage.py` builds the receiver and GNSS constellations, computes specular or backscatter coverage over the configured shapefile, and writes coverage/reflection/orbit plots under `OUTPUT/`.

`main_simulation_movingTarget.py` builds the ship trajectory and antenna model, runs either `specular` or `backscatter` mode from `[REFLECTION].reflection_type`, and writes footprint-overlap maps, link-budget breakdown plots, and optional animations under `OUTPUT/`.

## Monte Carlo moving-target runs

`run_monte_carlo.py` reads the scenario from an INI file and samples trials around it. The INI must
contain a `[MONTE CARLO]` section in addition to the sections of `Configuration.ini`. All keys are
required:

| Key | Meaning |
| --- | --- |
| `n_trials`, `seed` | Number of trials; random seed (draws are reproducible) |
| `window_hours` | Simulated time window per trial [h] |
| `vary_heading` | Random ship heading (otherwise 90 deg) |
| `vary_position`, `roi_bbox` | Random start position on water inside `[min_lon, max_lon, min_lat, max_lat]` (otherwise `[MOVING TARGET].start_pos`) |
| `vary_epoch`, `epoch_window_hours` | Random epoch offset, uniform in +/- `epoch_window_hours` |
| `vary_sea_state`, `wind_speed_range_m_s`, `wave_height_range_m`, `wind_direction_range_deg`, `inverse_wave_age_range` | Sea state drawn uniformly from the `[min, max]` ranges (otherwise `[OCEAN]`) |
| `vary_coherent_integration`, `coherent_integration_time_s_values` | T_coh drawn from the list (otherwise `[LINK BUDGET]`) |
| `vary_incoherent_looks`, `num_incoherent_looks_values` | M drawn from the list (otherwise `[LINK BUDGET]`) |
| `vary_system_noise_temperature`, `system_noise_temperature_k_range` | T_sys drawn uniformly from the `[min, max]` range (otherwise `[LINK BUDGET]`) |

Use a fresh output directory for a new run, or the same directory to resume it:

```powershell
python SRC/run_monte_carlo.py <your_mc_config>.ini --output-dir OUTPUT/mc_events_run
```

An optional trial count goes after the INI filename. Omitting `--output-dir` uses `OUTPUT/`.
Completed trial indices in `mc_<mode>_trials.jsonl` are skipped on resume, including legacy
records; use a fresh directory when changing the scenario or starting an independent run.

Each new trial stores `power_events`: constellation, band, ISO timestamp, transmitter/receiver
IDs, ocean power in watts, and ship power (backscatter) or forward-shadow magnitude (specular).
The existing best-per-trial SINR/detectability summaries and figures are retained. Four
additional standalone figures use **all finite, positive paired footprint-overlap powers**,
including events below the detection threshold:

| Figure | Event-level quantity |
| --- | --- |
| `mc_backscatter_received_power.png` | Ship versus ocean power [dBW], with an equal-power line |
| `mc_backscatter_scr.png` | `10 log10(P_ship / P_ocean)` [dB] |
| `mc_specular_received_power.png` | Forward-shadow magnitude versus ocean power [dBW] |
| `mc_specular_shadow_fraction.png` | `10 log10(abs(delta P) / P_ocean)` [dB] |

Colors and marker/line styles distinguish constellations; selected bands are combined.
Every valid event contributes one sample, so trials with more overlap events contribute
more samples. These are event distributions, not detection probabilities. Sampling is
unchanged: sampled coherent integration time still affects delay-Doppler clutter area
when that model is enabled.

Regenerate summaries and plots from saved JSONL data without running simulation trials:

```powershell
python SRC/run_monte_carlo.py <your_mc_config>.ini aggregate --output-dir OUTPUT/mc_events_run
```

`aggregate` reads the JSONL without modifying it and regenerates derived outputs in the
chosen directory. Legacy trials without `power_events` remain in the original summaries
but are reported as excluded from the new plots. Mixed runs plot only saved valid event
pairs; best-per-trial SCR is never substituted. If none exist, new plots are skipped
(existing image files are left unchanged). Complete event plots for an older run require
a new run in a fresh directory. Keep previous results and let any active run finish;
aggregation does not launch or rerun trials. Area coverage continues through its separate
`main_simulation_areaCoverage.py` workflow.

## Configuration

`Configuration.ini` is the source of truth for the scenario: LEO receiver orbits, GNSS TLE files, antenna and bands, reflection mode and gates, link-budget and ocean settings, the area-coverage region and grid, and the ship route, dimensions and plot settings. Its comments describe every key. Engine/numerical settings live in `simulator_settings.yaml`.

GNSS transmit power and chip-rate data are stored in `SRC/object_definitions/Constants_def.py`. `[ANTENNA].bands` selects constellation-specific signals such as `L1`, `E1`, or `B1C`.

## Tests and Checks

```powershell
pytest
black .
ruff check .
mypy SRC
```

Golden references live in `tests/golden/`. Regenerate them only when the underlying model change is intentional.

## Documentation

The documentation is published at <https://ntnu-smallsat-lab.github.io/GNSS_R_System_Simulator/>; new users start with its Getting started guide. Build it locally with:

```powershell
cd docs
.\make.bat html
```

The documentation source lives in `docs/source/`. It mirrors `SRC/` with module pages for functions, object definitions, modes, configuration, architecture, and validation.

## Data Sources

| Data | Location | Source and terms |
| --- | --- | --- |
| Coastline, ocean and marine-area polygons | `DATA/world_costline_data/` | [Natural Earth](https://www.naturalearthdata.com/) 1:10m physical vectors, public domain |
| Planetary ephemeris DE421 | `DATA/SKYFIELD_data/de421.bsp` | NASA/JPL, read through Skyfield |
| GNSS two-line elements | `DATA/TLE_EPHEMIRIS/` | [CelesTrak](https://celestrak.org/); refresh them as their epochs age out |
| Norwegian EEZ (not bundled, see [Setup](#setup)) | `DATA/eez/eez_norge.shp` | Flanders Marine Institute (2023). Maritime Boundaries Geodatabase: Maritime Boundaries and Exclusive Economic Zones (200NM), version 12. [marineregions.org](https://www.marineregions.org/), <https://doi.org/10.14284/632>. CC BY 4.0 |

## Generated Files

Do not commit generated artifacts from `OUTPUT/`, generated Sphinx builds, cached route shapefiles, or autogenerated ephemeris files. Keep committed changes focused on source, tests, configuration, and documentation.

## Citation

Each GitHub release is archived on Zenodo with its own DOI. Cite the release you used; the citation metadata is in `CITATION.cff` (GitHub's "Cite this repository" button).

## License

The code is released under the MIT License (see `LICENSE`). The data in `DATA/` keeps the terms of its source (see [Data Sources](#data-sources)).
