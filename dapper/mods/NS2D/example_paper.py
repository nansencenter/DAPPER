#Just using bocquet2019 settings for format and example; reproduce paper results if found

from dapper.mods.NS2D import Model, LP_setup
import dapper.mods as modelling
from dapper.mods.Lorenz96 import LPs
from dapper.tools.localization import nd_Id_localization
import numpy as np

System = Model(T=3, nu = 0.1)
Nx = System.Nx

tseq = modelling.Chronology(System.dt, dko=1 , Ko=1 * 10**3, BurnIn=20, Tplot=0.1)

Dyn = {
    "M" : np.prod((128, 128)),
    "model": System.step,
    "linear": System.dstep_dx,
    "noise": 0,
}
X0 = modelling.RV(M=Dyn["M"], func=lambda N: np.tile(System.x0.flatten() + 0 * np.random.randn(N, Dyn["M"]), (N, 1)))
#X0 = lambda N: np.tile(System.x0.flatten() + 0 * np.random.randn(N, Dyn["M"]), (N, 1))
Obs = modelling.Id_Obs(Nx**2)
Obs["noise"] = 1
Obs["localizer"] = nd_Id_localization((Nx,), (4, ))

rstream = np.random.RandomState()
jj = modelling.linspace_int(128, 128)
max_offset = jj[1] - jj[0]
def obs_inds(ko):
    def random_offset():
        rstream.seed(ko)
        u = rstream.rand()
        return int(np.floor(max_offset * u))

    return jj + random_offset()

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, LP=LP_setup(obs_inds))
# HMM.liveplotters = LP_setup(obs_inds)