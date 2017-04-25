# This file holds global (DAPPER-wide) imports and settings

import sys
assert sys.version_info >= (3,5)
import os.path
from time import sleep
from collections import OrderedDict
import warnings
import traceback
import re
import functools

import builtins
try:
    # This will be available if launched as (e.g.)
    # (bash)$ kernprof -l -v bench_example.py
    profile = builtins.profile
except AttributeError:
    # Otherwise: provide a pass-through version.
    def profile(func): return func


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
#from scipy.linalg import eig # Necessitates np.real_if_close().
from numpy.linalg import eig
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
# Installation suggestion 
##################################
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

# Raise error on warning
#warnings.filterwarnings('error',category=RuntimeWarning)


##################################
# Interactive plotting settings
##################################
def user_is_patrick():
  import getpass
  return getpass.getuser() == 'pataan'

# Choose graphics backend.
import matplotlib as mpl
from IPython import get_ipython
if 'zmq' in str(type(get_ipython())).lower():
  # notebook frontent
  mpl.use('nbAgg') # interactive
else:
  # terminal frontent
  if user_is_patrick():
    from sys import platform
    if platform == 'darwin':
      #mpl.use('Qt4Agg') # deprecated
      #mpl.use('TkAgg')  # has geometry(placement)
      mpl.use('MacOSX') # prettier, stable, fast (notable in LivePlot)
    else:
      pass
import matplotlib.pyplot as plt
plt.ion()


# Color set up
try:
  olderr = np.geterr() # affected by seaborn (pandas?)
  import seaborn as sns
  np.seterr(**olderr)  # restore np float error treatment
  sns.set_style({'image.cmap': 'BrBG', 'legend.frameon': True})
  sns_bg = array([0.9176, 0.9176, 0.9490])
  sns.set_color_codes()
except ImportError as err:
  install_warn(err)
  #plt.style.use('ggplot') # 'fivethirtyeight', 'bmh'
  mpl.rcParams['image.cmap'] = 'BrBG'

RGBs = {'w': array([1,1,1]), 'k': array([0,0,0])}
for c in 'bgrmyc':
  RGBs[c] = array(mpl.colors.colorConverter.to_rgb(c))

# With TkAgg/Qt4Agg backend this causes warning.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# With TkAgg/Qt4Agg backend this causes warning.
#mpl.rcParams['toolbar'] = 'None'
#warnings.filterwarnings("ignore",category=UserWarning)


##################################
# Setup DAPPER namespace
##################################
from aux.utils import *
from aux.misc import *
from aux.chronos import *
from aux.stoch import *
from aux.series import *
from aux.viz import *
from aux.matrices import *
from aux.randvars import *
from stats import *
from aux.admin import *
from aux.convenience import *
from da_algos import *

