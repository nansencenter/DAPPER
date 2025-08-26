#Just using bocquet2019 settings for format and example; reproduce paper results if found

from dapper.mods.NS2D import Model
import dapper.mods as modelling
from dapper.mods.Lorenz96 import LPs
from dapper.tools.localization import nd_Id_localization
import numpy as np

System = Model(T=200, dt=0.01, nu = 1/1600)
Nx = System.Nx

tseq = modelling.Chronology(System.dt, dko=10, Ko=2 * 10**4, BurnIn=20, Tplot=0.1)

Dyn = {
    "M" : Nx**2,
    "model": System.step,
    "linear": System.dstep_dx,
    "noise": 0,
}
X0 = modelling.GaussRV(mu=np.mean(System.x0), C=0.001)

Obs = modelling.Id_Obs(Nx**2)
Obs["noise"] = 1
Obs["localizer"] = nd_Id_localization((Nx,), (4, ))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(np.arange(Nx))