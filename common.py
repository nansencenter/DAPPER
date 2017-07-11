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
    # (bash)$ kernprof -l -v example_1.py
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
from scipy.linalg import sqrtm, inv, eigh, block_diag

from numpy import \
    pi, nan, \
    log, log10, exp, sin, cos, tan, \
    sqrt, floor, ceil, \
    mean, prod, \
    array, asarray, asmatrix, \
    linspace, arange, reshape, \
    eye, zeros, ones, diag, trace, \
    fromiter, reshape,hstack,vstack

from itertools import product

from time import time,strftime

from pandas import DataFrame,read_csv


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


##################################
# Plotting settings
##################################
def user_is_patrick():
  import getpass
  return getpass.getuser() == 'pataan'

import matplotlib as mpl

# Choose graphics backend.
try:
  from IPython import get_ipython
  is_notebook = 'zmq' in str(type(get_ipython())).lower()
except ImportError:
  is_notebook = False
if is_notebook:
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
# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles
try:
  olderr = np.geterr() # gets affected by seaborn (pandas?)
  import seaborn as sns
  np.seterr(**olderr)  # restore np float error treatment
  sns.set_style({'image.cmap': 'BrBG', 'legend.frameon': True})
  sns_bg = array([0.9176, 0.9176, 0.9490])
  sns.set_color_codes()
except ImportError as err:
  install_warn(err)
  #plt.style.use('ggplot') # 'fivethirtyeight', 'bmh', 'seaborn-darkgrid',
  mpl.rcParams['image.cmap'] = 'BrBG'

RGBs = {c: array(mpl.colors.colorConverter.to_rgb(c)) for c in 'bgrmycw'}

# With TkAgg/Qt4Agg backend this causes warning.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# With TkAgg/Qt4Agg backend this causes warning.
#mpl.rcParams['toolbar'] = 'None'
#warnings.filterwarnings("ignore",category=UserWarning)


##################################
# Setup DAPPER namespace
##################################
from tools.utils import *
from tools.misc import *
from tools.chronos import *
from tools.stoch import *
from tools.series import *
from tools.matrices import *
from tools.randvars import *
from tools.viz import *
from stats import *
from tools.admin import *
from tools.convenience import *
from da_methods import *

