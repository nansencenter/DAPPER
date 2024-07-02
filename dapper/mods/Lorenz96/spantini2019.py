"""Try reproducing Figure 9 of Spantini's coupling/nonlinear-transport paper."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96 import Tplot, dstep_dx, step, x0
from dapper.tools.localization import nd_Id_localization

tseq = modelling.Chronology(0.02, dto=0.4, Ko=2000, Tplot=Tplot, BurnIn=2 * Tplot)
Nx = 40
x0 = x0(Nx)
Dyn = {
    "M": Nx,
    "model": step,
    "linear": dstep_dx,
    "noise": 0,
}
X0 = modelling.GaussRV(mu=x0, C=0.1)
jj = np.arange(0, Nx, 2)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs["noise"] = 0.5
Obs["localizer"] = nd_Id_localization((Nx,), (2,), jj)
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

####################
# Suggested tuning
####################

#                                                      # rmse.a
# xps += da.Climatology()                              # 3.6
# xps += da.OptInterp()                                # 2.5
# xps += da.Var3D(B="eye", xB=4)                       # 2.3
# xps += da.EnKF ("Sqrt", N=40 , infl=1.10)            # 2.2
# xps += da.EnKF ("Sqrt", N=60 , infl=1.10)            # 1.3
# xps += da.EnKF ("Sqrt", N=100, infl=1.10)            # 0.97
# xps += da.LETKF(        N=40 , infl=1.10, loc_rad=4) # 1.0
# xps += da.LETKF(        N=60 , infl=1.02, loc_rad=4) # 0.89
# xps += da.LETKF(        N=100, infl=1.02, loc_rad=4) # 0.84
# xps += da.iEnKS("Sqrt", N=40 , infl=1.10, Lag=2)     # 0.52
# xps += da.iEnKS("Sqrt", N=60 , infl=1.10, Lag=2)     # 0.47
# xps += da.iEnKS("Sqrt", N=100, xN=1, Lag=2)          # 0.43
