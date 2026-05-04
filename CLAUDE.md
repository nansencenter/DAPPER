# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is DAPPER

DAPPER is a Python benchmarking framework for data assimilation (DA) methods. It implements twin experiments: a truth is simulated, observations are generated from it, and various DA methods are run to estimate the truth from those observations. It is a research/teaching tool, not a production framework — readability and reproducibility take priority over performance.

## Commands

**Install (editable with dev tools):**
```bash
pip install -e '.[dev]'
```

**Run tests:**
```bash
pytest tests                    # all integration tests
pytest                          # doctests only (conftest default)
pytest tests/test_iEnKS.py -v   # single test file
pytest -k "test_EnKF"           # filter by name
pytest tests --ignore=dapper/mods/QG --cov=dapper --cov-report term-missing
```

**Lint / format:**
```bash
pre-commit run --all-files
ruff check --output-format=concise
ruff format .
```

**Docs:**
```bash
mkdocs serve
```

## Architecture

### Core data flow

1. **Define a model** — instantiate a `HiddenMarkovModel` (in `dapper/mods/__init__.py`) with:
   - `Dyn`: dynamics operator (state transition)
   - `Obs`: observation operator
   - `tseq`: a `Chronology` object (from `dapper/tools/chronos.py`) specifying time steps and observation times
   - `X0`: initial-state distribution, a `RV` object (from `dapper/tools/randvars.py`)

2. **Simulate truth and obs** — `simulate(HMM)` produces `xx` (true states) and `yy` (observations).

3. **Run DA methods** — each method is a class decorated with `@da_method()` (in `dapper/da_methods/__init__.py`). The decorator injects `Stats` initialization, timing, and error handling. Every method must implement `assimilate(HMM, xx, yy)`.

4. **Collect results** — `xp.stats` holds per-timestep diagnostics. `xp.stats.average_in_time()` produces `xp.avrgs`, the time-averaged scores used for comparison.

5. **Batch experiments** — `xpList` (in `dapper/xp_launch.py`) runs many `(method, tuning)` combinations, optionally in parallel, and serializes results via `dill`.

### Key modules

| Module | Role |
|---|---|
| `dapper/mods/` | Dynamical models (Lorenz63, Lorenz96, KS, QG, …). Each subdirectory has `__init__.py` defining the HMM. |
| `dapper/da_methods/` | DA algorithms: ensemble (`ensemble.py`), variational (`variational.py`), particle (`particle.py`), extended KF (`extended.py`), baselines (`baseline.py`). |
| `dapper/tools/chronos.py` | `Chronology` / `Ticker` — manage discrete time with separate dt (model) and dko (obs) steps. |
| `dapper/tools/randvars.py` | `RV`, `GaussRV`, and other distributions for sampling and scoring. |
| `dapper/tools/matrices.py` | `CovMat` — lazy covariance matrix supporting diagonal, full, and square-root forms. |
| `dapper/stats.py` | `Stats` — records RMSE, spread, and other scores per timestep; computes time-averages with auto-correlation-corrected confidence intervals. |
| `dapper/xp_process.py` | Post-hoc analysis and table/plot generation for `xpList` results. |
| `dapper/tools/liveplotting.py` | Optional real-time visualization during assimilation. |
| `dapper/tools/localization.py` | Ensemble localization (tapering, local analyses) for high-dimensional problems. |

### `@da_method()` decorator contract

A DA method is a plain class whose `__init__` stores hyperparameters and whose `assimilate(self, HMM, xx, yy)` runs the filter/smoother. The decorator:
- wraps `assimilate` to create and attach a `Stats` object as `self.stats`
- provides `fail_gently` (catches and logs errors without killing a batch run)
- adds timing and progress-bar support

### Models

Each model directory under `dapper/mods/` provides one or more pre-configured HMMs (e.g., `dapper/mods/Lorenz63/sak12.py`). The `QG` model (quasi-geostrophic PDE) has a compiled Fortran extension and is excluded from coverage runs.

## Configuration

`pyproject.toml` contains pytest, coverage, and ruff settings. `conftest.py` sets the default pytest collection to doctests when no path is given. Pre-commit hooks run `ruff`, `nbstripout`, and standard checks.
