"""(Data Assimilation with Python: a Package for Experimental Research)

## In the README

Make sure you've browsed these sections in the README:

- [Installation](https://github.com/nansencenter/DAPPER#Installation)
- [Quickstart](https://github.com/nansencenter/DAPPER#Quickstart)

## Usage
Do you wish to illustrate and run benchmarks with your own
**model** or **method**?

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
    - (Independent) experiments can run in parallel; see `example_3.py`
    - Forecast parallelisation is possible since
        the (user-implemented) model has access to the full ensemble;
        see example in `mods.QG`.
    - Analysis parallelisation over local domains;
        see example in `da_methods.ensemble.LETKF`
- Classes that simplify treating:
    - Experiment administration and launch via `admin.xpList`
      and data processing and presentation via `data_management.xpSpace`.
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

__version__ = "1.0.0"

import sys

from dapper.tools.series import UncertainQtty

from .admin import (HiddenMarkovModel, Operator, da_method, get_param_setter,
                    seed_and_simulate, xpList)
from .da_methods.baseline import Climatology, OptInterp, Var3D
# DA methods
from .da_methods.ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .da_methods.extended import ExtKF, ExtRTS
from .da_methods.other import LNETF, RHF
from .da_methods.particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .da_methods.variational import Var4D, iEnKS
from .data_management import (default_fig_adjustments, default_styles,
                              discretize_cmap, load_xps, make_label, rel_index,
                              xpSpace)
from .dpr_config import rc
from .stats import register_stat
from .tools.chronos import Chronology
from .tools.magic import magic_naming, spell_out
from .tools.math import (Id_Obs, ens_compatible, linspace_int, partial_Id_Obs,
                         round2, with_recursion, with_rk4)
from .tools.matrices import CovMat
from .tools.randvars import RV, GaussRV
from .tools.stoch import set_seed
from .tools.viz import freshfig


# Documentation generation -- exclusion
def _find_demos(as_path=False):
    "Find all model demo.py scripts."
    lst = []
    for d in (rc.dirs.dapper/"mods").iterdir():
        x = d/"demo.py"
        if x.is_file():
            x = x.relative_to(rc.dirs.DAPPER)
            if not as_path:
                x = str(x.with_suffix("")).replace("/", ".")
            lst.append(x)
    return lst
__pdoc__ = {
    "tools.remote.autoscaler": False,
    **{demo:False for demo in _find_demos()},
    "dapper.mods.KS.compare_schemes": False,
    "dapper.mods.LorenzUV.illust_LorenzUV": False,
    "dapper.mods.LorenzUV.illust_parameterizations": False,
    "dapper.mods.explore_props": False,
    "dapper.mods.QG.f90": False,
}
