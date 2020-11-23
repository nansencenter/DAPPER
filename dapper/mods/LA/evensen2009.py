"""A mix of Evensen'2009 and Sakov'2008"""

# NB: Since there is no noise, and the system is stable,
#     the rmse's from this HMM go to zero as T-->infty.
#     => benchmarks largely depend on the initial error,
#     and so these absolute rmse values are not so useful
#     for quantatative evaluation of DA methods.
#     For that purpose, see mods/LA/raanes2015.py instead.

import numpy as np

import dapper as dpr
from dapper.mods.LA import Fmat, sinusoidal_sample
from dapper.mods.Lorenz96 import LPs

Nx = 1000
Ny = 4
jj = dpr.linspace_int(Nx, Ny)

tseq = dpr.Chronology(dt=1, dkObs=5, T=300, BurnIn=-1, Tplot=100)

# WITHOUT explicit matrix (assumes dt == dx/c):
# step = lambda x,t,dt: np.roll(x,1,axis=x.ndim-1)
# WITH:
Fm = Fmat(Nx, c=-1, dx=1, dt=tseq.dt)


def step(x, t, dt):
    assert dt == tseq.dt
    return x @ Fm.T


Dyn = {
    'M': Nx,
    'model': step,
    'linear': lambda x, t, dt: Fm,
    'noise': 0
}

# In the animation, it can sometimes/somewhat occur
# that the truth is outside 3*sigma !!!
# Yet this is not so implausible because sinusoidal_sample()
# yields (multivariate) uniform (random numbers) -- not Gaussian.
wnum  = 25
a = np.sqrt(5)/10
X0 = dpr.RV(M=Nx, func = lambda N: a*sinusoidal_sample(Nx, wnum, N))

Obs = dpr.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.01

HMM = dpr.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LPs(jj))


####################
# Suggested tuning
####################
# xp = EnKF('PertObs',N=100,infl=1.02)
