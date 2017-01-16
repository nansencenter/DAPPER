import sys
assert sys.version_info >= (3,5)

import os.path

from time import sleep

from collections import OrderedDict

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
    sqrt, abs, floor, ceil, prod, \
    sum, mean, \
    linspace, arange, reshape, \
    pi, log, sin, cos, tan, sign, \
    array, asarray, matrix, asmatrix, \
    eye, zeros, ones, diag, trace

# TODO: consider set(dir(np)).intersection(set(dir(__builtin__)))
#abs all any bool complex float int max min object round str sum


##################################
# Installation suggestion 
##################################
import warnings
def install_msg(package):
  return "Could not find package '" + package + "' " + \
      "for importing and using fall-back utilities instead. " + \
      "We recommend installing '" + package + \
      "' (using pip or conda, etc...) " + \
      'to improve the functionality of DAPPER.'
def install_warn(import_err):
  name = import_err.args[0]
  #name = name.split('No module named ')[1]
  name = name.split("'")[1]
  warnings.warn(install_msg(name))

##################################
# Interactive plotting settings
##################################
import matplotlib as mpl
mpl.use('TkAgg') # has geometry(placement), but is uglier than MacOSX
import matplotlib.pyplot as plt
plt.ion()

# Color set up
try:
  import seaborn as sns
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


# With Qt4Agg backend plt.pause() causes warning. Ignore.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# With TkAgg backend this causes warning.
#mpl.rcParams['toolbar'] = 'None'



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
from aux.admin import *
from stats import *
from da_algos import *

