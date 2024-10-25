"""Try settings of Figure 6A of Ramgraber's transport smoothing paper"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63 import Tplot, dstep_dx, step, x0

tseq = modelling.Chronology(0.05, dto=0.1, Ko=1000, Tplot=Tplot, BurnIn=4 * Tplot)
Nx = len(x0)
Dyn = {
    "M": Nx,
    "model": step,
    "linear": dstep_dx,
    "noise": 0,
}
X0 = modelling.GaussRV(C=1, M=Nx)
Obs = modelling.partial_Id_Obs(Nx, np.arange(Nx))
Obs["noise"] = 4
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)


####################
# Suggested tuning
####################

# da_method       N   xN  reg   NER  |  rmse.a
# -----------  ----  ---  ---  ----  -  ------
# Climatology                        |    7.7
# OptInterp                          |    1.72
# Persistence                        |    5.2
# Var3D                              |    1.16
# EnKF_N          3  1               |    0.50
# EnKF_N         10  1.5             |    0.43
# iEnKS          10  1.3             |    0.32
# PartFilt      100       2.4  0.3   |    0.39
# PartFilt      800       0.9  0.2   |    0.28
# PartFilt     4000       0.7  0.05  |    0.24
