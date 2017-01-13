import sys
assert sys.version_info >= (3,5)

import os.path

from time import sleep

from collections import OrderedDict


##################################
# Interactive plotting settings
##################################
import matplotlib as mpl
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg')
plt.ion()
from mpl_toolkits.mplot3d import Axes3D

#plt.style.use('ggplot') fivethirtyeight, bmh
import seaborn as sns


# With Qt4Agg backend plt.pause() causes warning. Ignore.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


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
# From DAPPER
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

