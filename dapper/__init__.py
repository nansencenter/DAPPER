"""Data Assimilation with Python: a Package for Experimental Research (DAPPER).

DAPPER is a set of templates for benchmarking the performance of data assimilation (DA) methods
using synthetic/twin experiments.
"""

__version__ = "0.9.6"

##################################
# Standard lib
##################################
import sys
import os
import itertools
import warnings
import traceback
import re
import functools
import configparser
import builtins
from pathlib import Path, PurePath
from time import sleep
from copy import deepcopy
import dataclasses as dc
from typing import Optional, Any, Union

assert sys.version_info >= (3,8), "Need Python>=3.8"


##################################
# Profiling.
##################################
# pip install line_profiler
# Launch python script: $ kernprof -l -v myprog.py
# Functions decorated with 'profile' from below will be timed.
try:
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.


##################################
# Scientific
##################################
import numpy as np
import scipy as sp
import numpy.random
import scipy.linalg as sla
import numpy.linalg as nla
import scipy.stats as ss

from scipy.linalg import svd
from numpy.linalg import eig
# eig() of scipy.linalg necessitates using np.real_if_close().
from scipy.linalg import sqrtm, inv, eigh

from numpy import \
    pi, nan, \
    log, log10, exp, sin, cos, tan, \
    sqrt, floor, ceil, \
    mean, prod, \
    diff, cumsum, \
    array, asarray, asmatrix, \
    linspace, arange, reshape, \
    eye, zeros, ones, diag, trace \
    # Don't shadow builtins: sum, max, abs, round, pow

import matplotlib as mpl
import matplotlib.pyplot as plt 

##################################
# Imports from DAPPER package
##################################
# Load rc: default settings
from .dict_tools import *
from .dpr_config import rc

# 'Tis perhaps late to issue a welcome, but the heavy libraries are below.
if rc.welcome_message:
    print("Initializing DAPPER...",flush=True)

from .tools.colors import *
from .tools.utils import *
from .tools.math import *
from .tools.stoch import *
from .tools.matrices import *
from .tools.randvars import *
from .tools.chronos import *
from .tools.series import *
from .tools.viz import *
from .tools.liveplotting import *
from .tools.localization import *
from .tools.multiprocessing import *
from .tools.remote.uplink import *
from .stats import *
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
    print("PS: Turn off this message in your configuration: dpr_config.ini")
