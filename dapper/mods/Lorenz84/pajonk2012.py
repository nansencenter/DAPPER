"""Settings from `bib.pajonk2012deterministic`.

There is nothing to reproduce from the paper as there are no
statistically converged numbers.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63 import LPs
from dapper.mods.Lorenz84 import dstep_dx, step, x0

Nx = len(x0)
Ny = Nx

day = 0.05/6 * 24  # coz dt=0.05 <--> 6h in "model time scale"
t = modelling.Chronology(0.05, dko=1, T=200*day, BurnIn=10*day)

Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

# X0 = modelling.GaussRV(C=0.01,M=Nx) # Decreased from Pajonk's C=1.
X0 = modelling.GaussRV(C=0.01, mu=x0)

jj = np.arange(Nx)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.1

HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0, LP=LPs(jj))

####################
# Suggested tuning
####################
# xps += ExtKF(infl=2)
# xps += EnKF('Sqrt',N=3,infl=1.01)
# xps += PartFilt(reg=1.0, N=100, NER=0.4) # add reg!
# xps += PartFilt(reg=1.0, N=1000, NER=0.1) # add reg!
