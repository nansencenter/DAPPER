"""Data Assimilation with Python: a Package for Experimental Research (DAPPER).

DAPPER is a set of templates for benchmarking the performance of data assimilation (DA) methods
using synthetic/twin experiments.
"""

__version__ = "0.9.6"

import sys

assert sys.version_info >= (3,8), "Need Python>=3.8"

# Profiling.
# Launch python script: $ kernprof -l -v myprog.py
# Functions decorated with 'profile' from below will be timed.
try:
    import builtins
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.



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
from .tools.math import (ens_compatible, linspace_int, partial_Id_Obs, round2,
                         with_recursion, with_rk4)
from .tools.matrices import CovMat
from .tools.randvars import RV, GaussRV
from .tools.stoch import rand, randn, set_seed
from .tools.viz import freshfig
