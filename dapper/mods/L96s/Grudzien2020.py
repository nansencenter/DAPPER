########################################################################################
########################################################################################

"""Settings as in `bib.grudzien2020numerical`.

A similar HMM is used with additive noise as a nonlinear map in various other papers,
whereas in this setting the model can be considered a random diffeomorphism, giving
a perfect-random model configuration. This uses two different solvers for the truth and
model twins respectively.

"""
########################################################################################
########################################################################################
# imports and exports

import numpy as np
from L96 import dxdt, l96s_tay2_step

import dapper.mods as modelling
from dapper.mods.integration import rk4
from dapper.mods.L96s import Tplot, dstep_dx, x0

########################################################################################
########################################################################################
## define experiment configurations

# Grudzien 2020 uses the below chronology with KObs=25000, BurnIn=5000
t = modelling.Chronology(dt=0.01, dkObs=100, KObs=11000, Tplot=Tplot, BurnIn=1000)

# set the system diffusion coefficient
diff = 0.05

# set the model state vector dimension
Nx = 10

# define the initial condition
x0 = x0(Nx)


def ensemble_step(x0, t, dt):
    return rk4(lambda t, x: dxdt(x), x0, np.nan, dt, s=diff, stages=4)


def truth_step(x0, t, dt):
    return l96s_tay2_step(x0, np.nan, dt, diff)


EnsembleDyn = {
    'M': Nx,
    'model': ensemble_step,
    'linear': dstep_dx,
    'noise': 0,
}

TruthDyn = {
    'M': Nx,
    'model': truth_step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(mu=x0, C=0.001)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 1

HMM_ensemble = modelling.HiddenMarkovModel(EnsembleDyn, Obs, t, X0)
HMM_truth = modelling.HiddenMarkovModel(TruthDyn, Obs, t, X0)


####################
# Suggested tuning
####################
