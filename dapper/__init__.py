"""Data Assimilation with Python: a Package for Experimental Research (DAPPER).

DAPPER is a set of templates for benchmarking the performance of data assimilation (DA) methods
using synthetic/twin experiments.
"""

__version__ = "0.9.0"

print("Initializing DAPPER...",end="", flush=True)


##################################
# Tools
##################################
import sys
assert sys.version_info >= (3,5)
import os
from time import sleep
from collections import OrderedDict
import itertools
import warnings
import traceback
import re
import functools

# Dirs
dpr        = os.path.dirname(os.path.abspath(__file__)) # DAPEPR
data_dir   = os.path.join(dpr,"..","data")              # DAPPER/data
# data_dir = os.path.join(".","data")                   # PWD/data
sample_dir = os.path.join(data_dir,"samples")           # data_dir/samples

# Pandas changes numpy's error settings. Correct.
# olderr = np.geterr()
# import pandas as pd
# np.seterr(**olderr)

# Profiling. Decorate the function you wish to time with 'profile' below
# Then launch program as: $ kernprof -l -v myprog.py
import builtins
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


##################################
# Plotting settings
##################################
import matplotlib as mpl

# user_is_patrick
import getpass
user_is_patrick = getpass.getuser() == 'pataan'

# Choose graphics backend.
from sys import platform
if user_is_patrick and platform == 'darwin':
  try:
    mpl.use('Qt5Agg') # pip install PyQt5 (and get_screen_size needs qtpy).
    import matplotlib.pyplot # Trigger (i.e. test) the actual import
  except ImportError:
    # Was prettier/stabler/faster than Qt4Agg, but Qt5Agg has caught up.
    mpl.use('MacOSX')

# terminal frontent
mpl_is_interactive=True
if 'inline' in mpl.get_backend():
  print("\nWarning: interactive/live plotting functionality is off.")
  print("Try another backend in your settings, e.g., mpl.use('Qt5Agg').")
  mpl_is_interactive=False

# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles, e.g. 'fivethirtyeight', 'bmh', 'seaborn-darkgrid'
plt.style.use(['seaborn-darkgrid',os.path.join(dpr,'tools','DAPPER.mplstyle')])


##################################
# Imports from DAPPER package
##################################
from .tools.colors import *
from .tools.utils import *
from .tools.multiprocessing import *
from .tools.math import *
from .tools.chronos import *
from .tools.stoch import *
from .tools.series import *
from .tools.matrices import *
from .tools.randvars import *
from .tools.viz import *
from .tools.liveplotting import *
from .tools.localization import *
from .tools.convenience import *
from .tools.data_management import *
from .stats import *
from .admin import *
from .da_methods.ensemble import *
from .da_methods.particle import *
from .da_methods.extended import *
from .da_methods.baseline import *
from .da_methods.variational import *
from .da_methods.other import *


print("...Done") # ... initializing DAPPER



