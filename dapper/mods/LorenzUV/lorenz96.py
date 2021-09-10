"""As in `bib.lorenz1996predictability`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.LorenzUV import model_instance

from ..utils import rel2mods

LUV = model_instance(nU=36, J=10, F=10)
nU = LUV.nU


################
# Full
################

tseq = modelling.Chronology(dt=0.005, dto=0.05, T=4**3, BurnIn=6)


Dyn = {
    'M': LUV.M,
    'model': modelling.with_rk4(LUV.dxdt, autonom=True),
    'noise': 0,
    'linear': LUV.dstep_dx,
}

X0 = modelling.GaussRV(mu=LUV.x0, C=0.01)

R = 1.0
jj = np.arange(nU)
Obs = modelling.partial_Id_Obs(LUV.M, jj)
Obs['noise'] = R

other = {'name': rel2mods(__file__)+'_full'}
HMM_full = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, **other)


################
# Truncated
################

# Just change dt from 005 to 05
tseq = modelling.Chronology(dt=0.05, dto=0.05, T=4**3, BurnIn=6)

Dyn = {
    'M': nU,
    'model': modelling.with_rk4(LUV.dxdt_parameterized),
    'noise': 0,
}

X0 = modelling.GaussRV(mu=LUV.x0[:nU], C=0.01)

jj = np.arange(nU)
Obs = modelling.partial_Id_Obs(nU, jj)
Obs['noise'] = R

other = {'name': rel2mods(__file__)+'_trunc'}
HMM_trunc = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, **other)

####################
# Suggested tuning
####################
#                     # Expected rmse.a ("U"-vars only!)
