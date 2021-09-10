"""Settings that produce somewhat interesting/challenging DA problems.

`dt` has been chosen after noting that
using `dt` up to 0.7 does not change the chaotic properties much,
as adjudged with eye-ball and Lyapunov measures.
"""

import dapper.mods as modelling

from dapper.mods.LotkaVolterra import step, dstep_dx, x0, LP_setup, Tplot

t = modelling.Chronology(0.5, dto=10, T=1000, BurnIn=Tplot, Tplot=Tplot)

Nx = len(x0)

Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
    }

X0 = modelling.GaussRV(mu=x0, C=0.01**2)

jj = [1, 3]
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.04**2

HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0, LP=LP_setup(jj))

####################
# Suggested tuning
####################
# Not carefully tuned:
# xps += EnKF_N(N=6)
# xps += ExtKF(infl=1.02)
