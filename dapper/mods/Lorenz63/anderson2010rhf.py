"""Settings from `bib.anderson2010non`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63 import Tplot, dstep_dx, step, x0

tseq = modelling.Chronology(0.01, dko=12, Ko=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(C=2, mu=x0)

Obs = modelling.partial_Id_Obs(Nx, np.arange(Nx))
Obs['noise'] = 8.0

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)


####################
# Suggested tuning
####################
# Compare with Anderson's figure 10.
# Benchmarks are fairly reliable (Ko=2000):
# from dapper.mods.Lorenz63.anderson2010rhf import HMM   # rmse.a
# xps += SL_EAKF(N=20,infl=1.01,rot=True,loc_rad=np.inf) # 0.87
# xps += EnKF_N (N=20,rot=True)                          # 0.87
# xps += RHF    (N=50,infl=1.10)                         # 1.28
# xps += RHF    (N=50,infl=0.95,rot=True)                # 0.94
# xps += RHF    (N=20,infl=0.95,rot=True)                # 1.07
