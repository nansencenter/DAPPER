"""From `bib.bocquet2010beyond` (again), but `ndim=40` (i.e. Fig. 5 of paper)."""
import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96.bocquet2010 import Dyn, tseq

Nx = 40
Dyn['M'] = Nx

X0 = modelling.GaussRV(M=Nx, C=0.001)

jj = np.arange(0, Nx, 2)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 1.5

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

####################
# Suggested tuning
####################
#                                                         # rmse.a
# xps += EnKF_N(      N=24,      rot=True ,infl=1.01)     # 0.38
# xps += iEnKS('Sqrt',N=19,Lag=2,rot=False,infl=1.04)     # 0.39
# xps += iEnKS('Sqrt',N=19,Lag=2,rot=False,xN=1.5)        # 0.39

# NB: Particle filter scores very sensitive to rare events => Need T>=2000.
# xps += PartFilt(N=3000,NER=0.20,reg=1.2)                # 0.77
# xps += PartFilt(N=5000,NER=0.10,reg=1.1)                # 0.72
# xps += PartFilt(N=10000,NER=0.05,reg=0.8)               # 0.45

# xps += PFxN(    N=100, xN=1000,NER=0.9,Qs=1.0)          # 0.87
# xps += PFxN(    N=100, xN=1000,NER=0.3,Qs=0.9)          # 0.72 Diverges
# xps += PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)          # 0.51
