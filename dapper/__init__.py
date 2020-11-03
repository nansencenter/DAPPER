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
import dapper.dict_tools as dtools
from .dict_tools import DotDict, NicePrint, deep_getattr, deep_hasattr
from .dpr_config import rc

# 'Tis perhaps late to issue a welcome, but the heavy libraries are below.
if rc.welcome_message:
    print("Initializing DAPPER...",flush=True)

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
from .admin import *
from .data_management import *
from .da_methods.ensemble import *
from .da_methods.particle import *
from .da_methods.extended import *
from .da_methods.baseline import *
from .da_methods.variational import *
from .da_methods.other import *

if rc.welcome_message:
    print("...Done") # ... initializing DAPPER
    print("PS: Turn off this message in your configuration: dpr_config.yaml")
