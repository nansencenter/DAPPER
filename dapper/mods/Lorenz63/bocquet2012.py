"""Reproduce results from Fig 11 of `bib.bocquet2012combining`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63.sakov2012 import HMM as _HMM

HMM = _HMM.copy()
# The only diff to sakov2012 is R:
# bocquet2012 uses 1 and 8, sakov2012 uses 2 (and 8)

HMM.Obs.noise.C = modelling.CovMat(np.eye(3))

HMM.name = HMM.name.replace("sakov", "bocquet")

####################
# Suggested tuning
####################
# from dapper.mods.Lorenz63.bocquet2012 import HMM         # rmse.a:
# HMM.tseq.dko = 25
# xps += iEnKS('Sqrt', N=10,infl=1.02,rot=True)            # 0.22
# xps += iEnKS('Sqrt', N=3, infl=1.04)                     # 0.23
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.22
#
# HMM.tseq.dko = 5
# xps += iEnKS('Sqrt', N=10,infl=1.02,rot=True)            # 0.13
# xps += iEnKS('Sqrt', N=3, infl=1.02)                     # 0.13
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.15
# xps += iEnKS('Sqrt', N=3, xN=2.0)                        # 0.14
#
# HMM.tseq.dko = 25 and R=8*eye(3):
# xps += iEnKS('Sqrt', N=3, xN=1.0)                        # 0.70
