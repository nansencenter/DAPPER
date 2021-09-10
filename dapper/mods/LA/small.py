"""Smaller version.

This is convenient because it's faster in liveplotting,
and convenient for debugging, e.g., `tests.test_iEnKS`.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.LA import Fmat, homogeneous_1D_cov
from dapper.mods.Lorenz96 import LPs

tseq = modelling.Chronology(dt=1, dko=5, T=300, BurnIn=-1, Tplot=100)

Nx = 100

# def step(x,t,dt):
#   return np.roll(x,1,axis=x.ndim-1)
Fm = Fmat(Nx, -1, 1, tseq.dt)


def step(x, t, dt):
    assert dt == tseq.dt
    return x @ Fm.T


Dyn = {
    'M': Nx,
    'model': step,
    'linear': lambda x, t, dt: Fm,
    'noise': 0,
}

X0 = modelling.GaussRV(mu=np.zeros(Nx), C=homogeneous_1D_cov(Nx, Nx/8, kind='Gauss'))

Ny  = 4
jj  = modelling.linspace_int(Nx, Ny)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.01


HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LPs(jj))


####################
# Suggested tuning
####################
# xps += EnKF('PertObs',N=16 ,infl=1.02)
# xps += EnKF('Sqrt'   ,N=16 ,infl=1.0)
