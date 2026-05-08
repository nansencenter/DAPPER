"""Settings that produce somewhat interesting/challenging DA problems."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Ikeda import LPs, Tplot, step, x0

tseq = modelling.Chronology(1, dko=1, Ko=1000, Tplot=Tplot, BurnIn=4 * Tplot)

Nx = len(x0)

Dyn = modelling.Operator(M=Nx, model=step, noise=0)

X0 = modelling.GaussRV(C=0.1, mu=x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj), noise=0.1)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################
