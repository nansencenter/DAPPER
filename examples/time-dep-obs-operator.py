# ## Illustrate how to define time-dependent observation operators (and vector lengths).
# Otherwise, this script is similar to examples/basic_1.py

# #### Imports

# %matplotlib notebook
import numpy as np

import dapper as dpr
import dapper.da_methods as da

# #### Setup
import dapper.mods as modelling
from dapper.mods.Lorenz63 import LPs, Tplot, dxdt

tseq = modelling.Chronology(0.01, dko=25, Ko=1000, Tplot=Tplot, BurnIn=4 * Tplot)


# Define a "double" Lorenz-63 system where each "half" evolves independently.
def dxdt_double(x):
    x1 = x[..., :3]
    x2 = x[..., 3:]
    f1 = dxdt(x1)
    f2 = dxdt(10 * x2) / 10
    return np.concatenate([f1, f2], -1)


Nx = 6
Dyn = {
    "M": Nx,
    "model": modelling.with_rk4(dxdt_double, autonom=True),
    "noise": 0,
}

X0 = modelling.GaussRV(C=2, mu=np.ones(Nx))


# Observe 1st and 2nd halves of state at alternating times
def Obs(ko):
    if ko % 2:
        return obs1
    else:
        return obs2


jj1 = 0 + np.arange(3)
jj2 = 3 + np.arange(3)
obs1 = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj1), noise=1)
obs2 = modelling.Operator(**modelling.partial_Id_Obs(Nx, jj2), noise=1)

HMM = modelling.HiddenMarkovModel(
    Dyn, dict(time_dependent=Obs), tseq, X0, sectors={"slow": jj1, "fast": jj2}
)

HMM.liveplotters = LPs(jj=lambda ko: jj1 if ko % 2 else jj2, params=dict())


# #### Run experiment

dpr.set_seed(3000)
xx, yy = HMM.simulate()

xp = da.EnKF("Sqrt", N=20, infl=1.02, rot=True)
# xp = da.PartFilt(N=100, reg=2.4, NER=0.3)

xp.assimilate(HMM, xx, yy, liveplots=True)

# print(xp.stats)  # ⇒ long printout
xp.stats.average_in_time()
# print(xp.avrgs)  # ⇒ long printout
print(xp.avrgs.tabulate(["rmse.a", "rmse.slow.a", "rmse.fast.a"]))
