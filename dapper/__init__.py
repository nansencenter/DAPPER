"""Data Assimilation with Python: a Package for Experimental Research (DAPPER).

DAPPER is a set of templates for benchmarking the performance of data assimilation (DA) methods
using synthetic/twin experiments.

See the full introduction in the [README](https://github.com/nansencenter/DAPPER)
In particular
[installation](https://github.com/nansencenter/DAPPER#installation)
and [getting started](https://github.com/nansencenter/DAPPER#getting-started)
"""

__version__ = "0.9.6"

import sys

assert sys.version_info >= (3,8), "Need Python>=3.8"

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
from .tools.math import (ens_compatible, linspace_int, Id_Obs, partial_Id_Obs, round2,
                         with_recursion, with_rk4)
from .tools.matrices import CovMat
from .tools.randvars import RV, GaussRV
from .tools.stoch import rand, randn, set_seed
from .tools.viz import freshfig


# Documentation management
# ---
# # 1. Generation:
# $ pdoc --force --html --template-dir docs -o ./docs dapper
# $ open docs/index.html
# # 2. Hosting:
# Push updated docs to github.
# In the main github settings of the repo,
# go to the "GitHub Pages" section,
# and set the source to the docs folder.
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

# This generates a lot of warnings:
# """UserWarning: __pdoc__-overriden key ... does not exist in module""".
# AFAICT that's fine. https://github.com/pdoc3/pdoc/issues/206
# Alternative: Insert this at top of each script to exclude
# >>> if __name__ != "__main__":
# >>>     raise RuntimeError("This module may only be run as script.")
# and run pdoc with --skip-errors.
__pdoc__ = {
    "tools.remote.autoscaler": False,
    **{demo:False for demo in _find_demos()},
    "dapper.mods.KS.compare_schemes": False,
    "dapper.mods.LorenzUV.illust_LorenzUV": False,
    "dapper.mods.LorenzUV.illust_parameterizations": False,
    "dapper.mods.explore_props": False,
    "dapper.mods.QG.f90": False,
}
