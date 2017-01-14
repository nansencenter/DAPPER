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


from numpy import sqrt, abs, floor, ceil, prod, \
    sum, mean, \
    linspace, arange, reshape, \
    pi, log, sin, cos, tan, sign, \
    array, asarray, matrix, asmatrix, \
    eye, zeros, ones, diag, \
    trace, \
    dot



##################################
# Interactive plotting settings
##################################
import matplotlib as mpl
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg')
plt.ion()

mpl.rcParams['toolbar'] = 'None'

# Color set up
try:
  import seaborn as sns
  sns.set_style({'image.cmap': 'BrBG', 'legend.frameon': True})
  sns_bg = array([0.9176, 0.9176, 0.9490])
  sns.set_color_codes()
except ImportError:
  # TODO: Provide suggestion to install?
  #plt.style.use('ggplot') # 'fivethirtyeight', 'bmh'
  mpl.rcParams['image.cmap'] = 'BrBG'

RGBs = {'w': array([1,1,1]), 'k': array([0,0,0])}
for c in 'bgrmyc':
  RGBs[c] = array(mpl.colors.colorConverter.to_rgb(c))


# With Qt4Agg backend plt.pause() causes warning. Ignore.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)



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

