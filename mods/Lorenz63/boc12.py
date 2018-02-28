# Reproduce results from Fig 11 of 
# M. Bocquet and P. Sakov (2012): "Combining inflation-free and
# iterative ensemble Kalman filters for strongly nonlinear systems"
from mods.Lorenz63.sak12 import setup
# The only diff to sak12 is R: boc12 uses 1 and 8, sak12 uses 2 (and 8)

from common import *

setup.h.noise.C = CovMat(eye(3))

setup.name = os.path.relpath(__file__,'mods/')

####################
# Suggested tuning
####################
#from mods.Lorenz63.boc12 import setup ##################### Expected RMSE_a:
#cfgs += iEnKS('-N', N=3,infl=0.95)                                 # 0.20
#
# With dkObs=5:
#cfgs += iEnKS('-N', N=3)                                           # 0.15
#cfgs += iEnKS('-N', N=3,xN=1.4)                                    # 0.14
#
# With R=8*eye(3):
#cfgs += iEnKS('-N', N=3)                                           # 0.70
