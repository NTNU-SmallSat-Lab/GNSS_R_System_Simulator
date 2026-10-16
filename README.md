# GNSS-R System Simulator

Preliminary-design simulator for GNSS-Reflectometry (GNSS-R) LEO mission studies. The code currently supports two analysis modes:

- **Area coverage**: counts contiguous specular-reflection track visits over a gridded region.
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
Configuration_area.ini                 Area-coverage scenario
Configuration_ship.ini                 Moving-target scenario
```

Run commands from the repository root. Runtime paths are resolved from the current working directory.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,docs]"
```

The project is validated for Python 3.12.x. Newer Python minors may work, but scientific binary wheels and debugger integrations can lag.

## Run Simulations

```powershell
python SRC/main_simulation_areaCoverage.py
python SRC/main_simulation_movingTarget.py
```

`main_simulation_areaCoverage.py` reads `Configuration_area.ini`, builds the receiver and GNSS constellations, computes specular coverage over the configured shapefile, and writes coverage/reflection/orbit plots under `OUTPUT/`.

`main_simulation_movingTarget.py` reads `Configuration_ship.ini`, builds the ship trajectory and antenna model, runs either `specular` or `backscatter` mode from `[REFLECTION].reflection_type`, and writes footprint-overlap maps, link-budget breakdown plots, and optional animations under `OUTPUT/`.

## Monte Carlo moving-target runs

Use a fresh output directory for a new run, or the same directory to resume it:

```powershell
python SRC/run_monte_carlo.py Configuration_mc_backscatter.ini --output-dir OUTPUT/mc_events_run
python SRC/run_monte_carlo.py Configuration_mc_specular.ini --output-dir OUTPUT/mc_events_run
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
python SRC/run_monte_carlo.py Configuration_mc_backscatter.ini aggregate --output-dir OUTPUT/mc_events_run
python SRC/run_monte_carlo.py Configuration_mc_specular.ini aggregate --output-dir OUTPUT/mc_events_run
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

The two root INI files are the source of truth for scenario setup:

- `Configuration_area.ini`: area shapefile, grid size, LEO receiver orbits, GNSS TLE files, antenna bands, reflection gate, link-budget/ocean settings.
- `Configuration_ship.ini`: ship route, antenna pattern generation, LEO receiver orbits, GNSS TLE files, reflection mode, plot/animation settings, link-budget/ocean settings, and ship dimensions.

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

Build the Sphinx docs with:

```powershell
cd docs
.\make.bat html
```

The documentation source lives in `docs/source/`. It mirrors `SRC/` with module pages for functions, object definitions, modes, configuration, architecture, and validation.

## Generated Files

Do not commit generated artifacts from `OUTPUT/`, generated Sphinx builds, cached route shapefiles, or autogenerated ephemeris files. Keep committed changes focused on source, tests, configuration, and documentation.
