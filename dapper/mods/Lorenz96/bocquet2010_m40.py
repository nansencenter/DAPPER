"""From [bocquet2010a][] (again), but `ndim=40` (i.e. Fig. 5 of paper)."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96 import step

tseq = modelling.Chronology(0.05, dko=1, T=4**3, BurnIn=20)

Nx = 40
Dyn = modelling.Operator(M=Nx, model=step, noise=0)

X0 = modelling.GaussRV(M=Nx, C=0.001)

jj = np.arange(0, Nx, 2)
Obs = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj), noise=1.5**2)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

####################
# Suggested tuning
####################
#                                                         # rmse.a
# xps += EnKF_N(      N=24,      rot=True ,infl=1.01)     # 0.58

# xps += iEnKS('Sqrt',N=19,Lag=2,rot=False,infl=1.04)     # 0.46
# xps += iEnKS('Sqrt',N=19,Lag=2,rot=False,xN=1.5)        # 0.45

# xps += PartFilt(N=3000,NER=0.20,reg=1.2)                # 1.01
# xps += PartFilt(N=5000,NER=0.10,reg=1.1)                # 0.88
# xps += PartFilt(N=10000,NER=0.05,reg=0.8)               # 0.57

# xps += PFxN(    N=100, xN=1000,NER=0.9,Qs=1.0)          # 1.02
# xps += PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)          # 0.62
