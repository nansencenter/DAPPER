"""Settings as in `bib.frei2013bridging`.

They also cite its use in the following:

`bib.bengtsson2003toward`, `bib.lei2011moment`, `bib.frei2013mixture`.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96 import dstep_dx, step
from dapper.tools.localization import nd_Id_localization

t = modelling.Chronology(0.05, dto=0.4, T=4**5, BurnIn=20)

Nx = 40
Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(M=Nx, C=0.001)

jj = 1 + np.arange(0, Nx, 2)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.5
Obs['localizer'] = nd_Id_localization((Nx,), (2,), jj)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0)


####################
# Suggested tuning
####################
# Compare to Table 1 and 3 from frei2013bridging. Note:
#  - N is too large to be very interesting.
#  - We obtain better EnKF scores than they report,
#    and use inflation and sqrt updating,
#    and don't really need localization.
# from dapper.mods.Lorenz96.frei2013bridging import HMM     # rmse.a
# xps += EnKF_N(N=400,rot=1)                                # 0.80
# xps += LETKF( N=400,rot=True,infl=1.01,loc_rad=10/1.82)   # 0.79 # short xp. only
# xps += Var3D()                                            # 2.42 # short xp. only
