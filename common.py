# This file holds global (DAPPER-wide) imports and settings

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
    array, asarray, asmatrix, \
    linspace, arange, reshape, \
    eye, zeros, ones, diag, trace \



##################################
# Tools
##################################
import sys
assert sys.version_info >= (3,5)
import os.path
from time import sleep
from collections import OrderedDict
import warnings
import traceback
import re
import functools

# Pandas changes numpy's error settings. Correct.
olderr = np.geterr()
import pandas as pd
np.seterr(**olderr)

# Profiling
import builtins
try:
    # This will be available if launched as (e.g.)
    # (bash)$ kernprof -l -v example_1.py
    profile = builtins.profile
except AttributeError:
    # Otherwise: provide a pass-through version.
    def profile(func): return func

# Installation suggestions
def install_msg(package):
  return """
  Could not find (import) package '{0}'. Using fall-back.
  [But we recommend installing '{0}' (using pip or conda, etc...)
  to improve the functionality of DAPPER.]""".format(package)
def install_warn(import_err):
  name = import_err.args[0]
  #name = name.split('No module named ')[1]
  name = name.split("'")[1]
  warnings.warn(install_msg(name))



##################################
# Plotting settings
##################################
def user_is_patrick():
  import getpass
  return getpass.getuser() == 'pataan'

import matplotlib as mpl

# is_notebook 
try:
  from IPython import get_ipython
  is_notebook = 'zmq' in str(type(get_ipython())).lower()
except ImportError:
  is_notebook = False

# Choose graphics backend.
if is_notebook:
  mpl.use('nbAgg') # interactive
else:
  # terminal frontent
  if user_is_patrick():
    from sys import platform
    if platform == 'darwin':
      mpl.use('MacOSX') # prettier, stable, fast (notable in LivePlot)
      #mpl.use('Qt4Agg') # deprecated

      # Has geometry(placement). Causes warning
      #mpl.use('TkAgg')  
      #import matplotlib.cbook
      #warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    else:
      pass

# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles, e.g. 'fivethirtyeight', 'bmh', 'seaborn-darkgrid'
plt.style.use(['seaborn-darkgrid','tools/DAPPER.mplstyle'])



##################################
# Setup DAPPER namespace
##################################
from tools.colors import *
from tools.utils import *
from tools.math import *
from tools.chronos import *
from tools.stoch import *
from tools.series import *
from tools.matrices import *
from tools.randvars import *
from tools.viz import *
from stats import *
from tools.admin import *
from tools.convenience import *
from tools.data_management import *
from da_methods import *


