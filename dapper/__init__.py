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
from time import sleep
from collections import OrderedDict

assert sys.version_info >= (3,6), "Need Python>=3.6"


##################################
# Config
##################################
dirs = {}
dirs['dapper']    = os.path.dirname(os.path.abspath(__file__))
dirs['DAPPER']    = os.path.dirname(dirs['dapper'])

_rc = configparser.ConfigParser()
# Load rc files from dapper, user-home, and cwd
_rc.read(os.path.join(x,'dpr_config.ini') for x in
    [dirs['dapper'], os.path.expanduser("~"), os.curdir])
# Convert to dict
rc = {s:dict(_rc.items(s)) for s in _rc.sections() if s not in ['int','bool']}
# Parse
rc['plot']['styles'] = rc['plot']['styles'].replace('$dapper',dirs['dapper']).replace('/',os.path.sep)
for x in _rc['int' ]: rc[x] = _rc['int' ].getint(x)
for x in _rc['bool']: rc[x] = _rc['bool'].getboolean(x)

# Define paths
dirs['data_root'] = os.getcwd() if rc['dirs']['data']=="cwd" else dirs['DAPPER']
dirs['data_base'] = "dpr_data"
dirs['data']      = os.path.join(dirs['data_root'], dirs['data_base'])
dirs['samples']   = os.path.join(dirs['DAPPER']   , dirs['data_base'], "samples")

# Profiling. Decorate the function you wish to time with 'profile' below
# Then launch program as: $ kernprof -l -v myprog.py
try:
    profile = builtins.profile     # will exists if launched via kernprof
except AttributeError:
    def profile(func): return func # provide a pass-through version.

if rc['welcome_message']:
  print("Initializing DAPPER...",flush=True)


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

if user_is_patrick:
  from sys import platform
  # Try to detect notebook
  try:
    __IPYTHON__
    from IPython import get_ipython
    is_notebook_or_qt = 'zmq' in str(type(get_ipython())).lower()
  except (NameError,ImportError):
    is_notebook_or_qt = False
  # Switch backend
  if is_notebook_or_qt:
    pass # Don't change backend
  elif platform == 'darwin':
    try:
      mpl.use('Qt5Agg') # pip install PyQt5 (and get_screen_size needs qtpy).
      import matplotlib.pyplot # Trigger (i.e. test) the actual import
    except ImportError:
      # Was prettier/stabler/faster than Qt4Agg, but Qt5Agg has caught up.
      mpl.use('MacOSX')

_BE = mpl.get_backend().lower()
_LP = rc['liveplotting_enabled']
if _LP: # Check if we should disable anyway:
  _LP &= not any([_BE==x for x in ['agg','ps','pdf','svg','cairo','gdk']])
  # Also disable for inline backends, which are buggy with liveplotting
  _LP &= 'inline' not in _BE
  _LP &= 'nbagg'  not in _BE
  if not _LP:
    print("\nWarning: interactive/live plotting was requested,")
    print("but is not supported by current backend: %s."%mpl.get_backend())
    print("Try another backend in your settings, e.g., mpl.use('Qt5Agg').")
rc['liveplotting_enabled'] = _LP

# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles
plt.style.use(rc['plot']['styles'].split(","))


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


if rc['welcome_message']:
  print("...Done") # ... initializing DAPPER
  print("PS: Turn off this message in your configuration: dpr_config.ini")



