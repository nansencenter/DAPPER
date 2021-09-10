"""Settings as in `bib.bocquet2019consistency`."""
import numpy as np

import dapper.mods as modelling
from dapper.mods.KS import Model, Tplot
from dapper.mods.Lorenz96 import LPs
from dapper.tools.localization import nd_Id_localization

KS = Model(dt=0.5)
Nx = KS.Nx

# nRepeat=10
tseq = modelling.Chronology(KS.dt, dko=2, Ko=2*10**4, BurnIn=2*10**3, Tplot=Tplot)

Dyn = {
    'M': Nx,
    'model': KS.step,
    'linear': KS.dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(mu=KS.x0, C=0.001)

Obs = modelling.Id_Obs(Nx)
Obs['noise'] = 1
Obs['localizer'] = nd_Id_localization((Nx,), (4,))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(np.arange(Nx))


####################
# Suggested tuning
####################

# Reproduce (top-right panel) of Fig. 4 of bocquet2019consistency    # Expected rmse.a:
# --------------------------------------------------------------------------------
# xps += LETKF(N=4 , loc_rad=15/1.82, infl=1.11,rot=True,taper='GC') # 0.18
# xps += LETKF(N=6,  loc_rad=25/1.82, infl=1.06,rot=True,taper='GC') # 0.14
# xps += LETKF(N=16, loc_rad=51/1.82, infl=1.02,rot=True,taper='GC') # 0.11
#
# Other:
# xps += Climatology()                                               # 1.3
# xps += OptInterp()                                                 # 0.5
# xps += EnKF('Sqrt', N=13,           infl=1.60,rot=True)            # 0.5
# xps += EnKF('Sqrt', N=20,           infl=1.03,rot=True)            # 0.115
