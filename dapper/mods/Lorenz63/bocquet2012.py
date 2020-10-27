"""Reproduce results from Fig 11 of 
M. Bocquet and P. Sakov (2012): 'Combining inflation-free and
iterative ensemble Kalman filters for strongly nonlinear systems'"""

from dapper.mods.Lorenz63.sakov2012 import HMM
# The only diff to sakov2012 is R:
# bocquet2012 uses 1 and 8, sakov2012 uses 2 (and 8)

from dapper import *

HMM.Obs.noise.C = CovMat(eye(3))

HMM.name = Path(__file__).relative_to(rc.dirs.dapper/"mods")

####################
# Suggested tuning
####################
# from dapper.mods.Lorenz63.bocquet2012 import HMM         # rmse.a:
# HMM.t.dkObs = 25
# xps += iEnKS('Sqrt', N=10,infl=1.02,rot=True)            # 0.22
# xps += iEnKS('Sqrt', N=3, infl=1.04)                     # 0.23
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.22
# 
# HMM.t.dkObs = 5
# xps += iEnKS('Sqrt', N=10,infl=1.02,rot=True)            # 0.13
# xps += iEnKS('Sqrt', N=3, infl=1.02)                     # 0.13
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.15
# xps += iEnKS('Sqrt', N=3, xN=2.0)                        # 0.14
# 
# HMM.t.dkObs = 25 and R=8*eye(3):
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.70
