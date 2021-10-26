import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz05 import Model
from dapper.tools.localization import nd_Id_localization

# Sakov uses K=300000, BurnIn=1000*0.05
tseq = modelling.Chronology(0.002, dto=0.05, Ko=400, Tplot=2, BurnIn=5)

model = Model(b=8)

Dyn = {
    'M': model.M,
    'model': model.step,
    'noise': 0,
}

X0 = modelling.GaussRV(mu=model.x0, C=0.001)

jj = np.arange(model.M)  # obs_inds
Obs = modelling.partial_Id_Obs(model.M, jj)
Obs['noise'] = 1
Obs['localizer'] = nd_Id_localization((model.M,), (6,))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
