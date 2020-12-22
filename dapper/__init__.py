"""(Data Assimilation with Python: a Package for Experimental Research)

## In the README

Make sure you've browsed these sections in the README:

- [Installation](https://github.com/nansencenter/DAPPER#Installation)
- [Quickstart](https://github.com/nansencenter/DAPPER#Quickstart)

## Usage
Adapt one of the examples scripts to your needs.

Fork DAPPER and make changes to its source if you need to
(which is quite likely because the generality of DAPPER is limited)

If, in particular, you wish to illustrate and run benchmarks with
your own **model** or **method**, then

- If it is a complex one, you may be better off using DAPPER
  merely as inspiration (but you should still
  [cite it](https://github.com/nansencenter/DAPPER#Contributors))
  rather than trying to squeeze everything into its templates.
- If it is relatively simple, however, you may well want to use DAPPER.
  In that case, read this:
    - `mods`
    - `da_methods`

## Features beyond the [README/highlights](https://github.com/nansencenter/DAPPER#Highlights)

- Parallelisation:
    - (Independent) experiments can run in parallel; see `examples/basic_3.py`
    - Forecast parallelisation is possible since
        the (user-implemented) model has access to the full ensemble;
        see example in `mods.QG`.
    - Analysis parallelisation over local domains;
        see example in `da_methods.ensemble.LETKF`
- Classes that simplify treating:
    - Experiment administration and launch via `xp_launch.xpList`
      and data processing and presentation via `xp_process.xpSpace`.
    - Time sequences use via `tools.chronos.Chronology`
      and`tools.chronos.Ticker`.
    - Random variables via `tools.randvars.RV`: Gaussian, Student-t, Laplace, Uniform, ...,
      as well as support for custom sampling functions.
    - Covariance matrices via `tools.matrices.CovMat`: provides input flexibility/overloading,
      lazy eval that facilitates the use of non-diagnoal covariance matrices (whether sparse or full).
- Diagnostics and statistics with
    - Confidence interval on times series (e.g. rmse) averages with
        - automatic correction for autocorrelation
        - significant digits printing
    - Automatic averaging of several types for sub-domains
      (e.g. "ocean", "land", etc.)


## Conventions

- Python version `>=3.7` for dicts to maintain ordering.
- Ensemble (data) matrices are np.ndarrays with shape `N-by-Nx`.
  This shape (orientation) is contrary to the EnKF literature,
  but has the following advantages:
    - Improves speed in row-by-row accessing,
      since that's `np`'s default orientation.
    - Facilitates broadcasting for, e.g. centering the matrix.
    - Fewer indices: `[n,:]` yields same as `[n]`
    - Beneficial operator precedence without `()`.
      E.g. `dy @ Rinv @ Y.T @ Pw` (where `dy` is a vector)
    - Less transposing for for ens-space formulae.
    - It's the standard for data matrices in
      statistical regression literature.
- Naming:
    - `E`: ensemble matrix
    - `w`: ensemble weights or coefficients
    - `X`: centered ensemble
    - `N`: ensemble size
    - `Nx`: state size
    - `Ny`: obs size
    - *Double letters* means a sequence of something.
      For example:
        - `xx`: Time series of truth; shape (K+1, Nx)
        - `yy`: Time series of obs; shape (KObs+1, Nx)
        - `EE`: Time series of ensemble matrices
        - `ii`, `jj`: Sequences of indices (integers)
    - `xps`: an `xpList` or `xpDict`,
      where `xp` abbreviates "experiment".

## Dev guide
If you are going to contribute to DAPPER, please read `dev_guide`.

## Bibliography/references
See `bib`.

## API reference
The rendered docstrings can be browsed
through the following links, which are also available in the left sidebar.
"""

__version__ = "1.1.0"

import sys

# Imports used by examples
from .dpr_config import rc
from .tools.magic import magic_naming, spell_out
from .tools.rounding import round2, round2sigfig
from .tools.seeding import set_seed
from .xp_launch import get_param_setter, seed_and_simulate, xpList
from .xp_process import (default_fig_adjustments, default_styles,
                         discretize_cmap, load_xps, make_label, rel_index,
                         xpSpace)
