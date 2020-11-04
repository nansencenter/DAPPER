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



##################################
# Imports from DAPPER package
##################################
# from .dict_tools import *
from .dpr_config import rc
# from .tools.colors import *
# from .tools.utils import *
# from .tools.math import *
from .tools.math import ens_compatible, with_rk4, with_recursion, round2, linspace_int, partial_Id_Obs
from .tools.stoch import set_seed, rand, randn
from .tools.matrices import CovMat
from .tools.randvars import RV, GaussRV
from .tools.chronos import Chronology
import dapper.tools.series as series
from dapper.tools.series import UncertainQtty
from .tools.viz import freshfig
# from .tools.liveplotting import *
from .tools.magic import magic_naming, spell_out
# from .tools.localization import *
# from .tools.multiprocessing import *
# from .tools.remote.uplink import *
from .stats import register_stat
from .admin import HiddenMarkovModel, Operator, da_method, seed_and_simulate, xpList, get_param_setter
from .data_management import load_xps, xpSpace, make_label, default_styles, rel_index, discretize_cmap, default_fig_adjustments

# DA methods
from .da_methods.ensemble import EnKF, EnKS, EnRTS, SL_EAKF, LETKF, EnKF_N
from .da_methods.particle import PartFilt, OptPF, PFa, PFxN_EnKF, PFxN
from .da_methods.extended import ExtKF, ExtRTS
from .da_methods.baseline import Climatology, OptInterp, Var3D
from .da_methods.variational import Var4D, iEnKS
from .da_methods.other import LNETF, RHF
