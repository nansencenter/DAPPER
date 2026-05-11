# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About DAPPER

DAPPER (Data Assimilation with Python: a Package for Experimental Research) is a benchmarking framework for comparing data assimilation (DA) algorithms on dynamical models via synthetic "twin experiments". It is a research tool, not an operational system.

## Commands

```bash
# Run all tests (includes doctests)
pytest tests

# Run a single test file
pytest tests/test_example_1.py

# Run only doctests (default addopts includes --doctest-modules dapper)
pytest

# Lint and auto-fix
ruff check --fix .
ruff format .

# Run pre-commit hooks on all files
pre-commit run --all-files

# Serve documentation locally
mkdocs serve -f properdocs.yml
```

## Architecture

The core data flow:

```
HMM.simulate() → (xx, yy)   # truth and observations
xp = da.EnKF(N=10, ...)     # configure DA method ("xp" = experiment)
xp.assimilate(HMM, xx, yy)  # run assimilation → populates xp.stats
xp.stats.average_in_time()  # → xp.avrgs
xp.avrgs.tabulate(["rmse.a", "rmv.a"])
```

**`dapper/mods/`** — Dynamical models. Each model lives in a subdirectory (e.g. `Lorenz63/`, `Lorenz96/`). Model configurations are in author-year named files (e.g. `sakov2012.py`). The key object is `HiddenMarkovModel` (HMM), which bundles the dynamic operator, observation operator, time sequence (`Chronology`), and noise (`GaussRV`).

**`dapper/da_methods/`** — All DA algorithms. Every method is a subclass of `da_method` (the base class in `__init__.py`). `__init_subclass__` applies `@dataclass`, renames the subclass's `assimilate(self, HMM, xx, yy)` to `_assimilate`, and sets `cls.da_method = cls.__name__`. The base class `assimilate()` is the actual wrapper: it initialises `self.stats`, calls `self._assimilate(HMM, xx, yy)`, and records wall-clock time. Available methods: `EnKF`, `EnKF_N`, `LETKF`, `EnKS`, `EnRTS` (ensemble); `iEnKS`, `Var4D`, `Var3D` (variational); `PartFilt`, `OptPF` (particle); `ExtKF`, `ExtRTS` (extended Kalman); `Climatology`, `OptInterp`, `Persistence` (baselines).

**`dapper/stats.py`** — `Stats` records per-timestep diagnostics during assimilation. `Avrgs` holds time-averaged results. `Stats.register(name, value)` registers a custom scalar stat so it is tracked by `average_in_time()`. The module-level `register_stat` function is the underlying implementation and works on any object (including `DACycleSeries` children); `Stats.register` is bound to it.

**`dapper/xp_launch.py`** — `xpList` manages batches of experiments with parameter sweeps. `run_experiment()` is the main runner used in batch mode.

**`dapper/xp_process.py`** — `xpSpace` organises results from many experiments into a sparse coordinate dict for tabulation and plotting.

**`dapper/tools/`** — Supporting utilities: `chronos.py` (time sequences), `randvars.py` (`GaussRV`), `matrices.py` (`CovMat`), `localization.py`, `liveplotting.py`, `series.py`.

## Key Conventions

- **Ensemble matrices** are shaped `(N, Nx)` — N members × Nx state dims. Never `(Nx, N)`.
- **Double-letter variables** (`xx`, `yy`, `EE`) are time series; single/abbreviated are current-timestep values.
- **`xp`** = experiment = a DA method instance with specific hyperparameters.
- **`HMM`** = Hidden Markov Model = the full twin-experiment setup.
- Docstrings follow NumPy style.
- `ruff` enforces style (line length 88, isort, pyupgrade, bugbear). `docs/examples/` and `scripts/` are excluded from ruff.

## Adding a New DA Method

Subclass `da_method` and define `assimilate`. The base class handles stats init, timing, and `fail_gently`:

```python
from dapper.da_methods import da_method

class MyMethod(da_method):
    param: float = 1.0

    def assimilate(self, HMM, xx, yy):
        # self.stats is available here (initialised by the base class wrapper)
        ...
        self.stats.register("my_scalar_stat", value)  # custom stat → xp.avrgs
```

Export the new class from `dapper/da_methods/__init__.py`.
