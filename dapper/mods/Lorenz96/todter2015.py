"""Concerns figure 4 of `bib.todter2015second`."""

import numpy as np

import dapper.mods as modelling
import dapper.tools.randvars as RVs
from dapper.mods.Lorenz96 import step
from dapper.tools.localization import nd_Id_localization

t = modelling.Chronology(0.05, dko=2, T=4**5, BurnIn=20)

Nx = 80
Dyn = {
    'M': Nx,
    'model': step,
    'noise': 0,
}

X0 = modelling.GaussRV(M=Nx, C=0.001)

jj = np.arange(0, Nx, 2)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['localizer'] = nd_Id_localization((Nx,), (1,), jj)
# Obs['noise'] = RVs.LaplaceRV(C=1,M=len(jj))
Obs['noise'] = RVs.LaplaceParallelRV(C=1, M=len(jj))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0)

####################
# Suggested tuning
####################

#                                                          rmse.a
# xps += LETKF(N=20,rot=True,infl=1.04       ,loc_rad=5) # 0.44
# xps += LETKF(N=40,rot=True,infl=1.04       ,loc_rad=5) # 0.44
# xps += LETKF(N=80,rot=True,infl=1.04       ,loc_rad=5) # 0.43
# These scores are quite variable:
# xps += LNETF(N=40,rot=True,infl=1.10,Rs=2.5,loc_rad=5) # 0.57
# xps += LNETF(N=80,rot=True,infl=1.10,Rs=1.6,loc_rad=5) # 0.45

# In addition to standard post-analysis inflation,
# we also find the necessity of tuning the inflation (Rs) for R,
# which is not mentioned in the reference.

# In summary, compared to the reference paper:
# - Our rmse scores for the local ETKF are better
# - Our rmse scores for the local NETF
#   - with N=80 seems to be equal
#   - with N=40 is worse
#
# These results (contradict and) pretty much void the merit of the NETF,
# especially considering how the Laplace obs-error favours the NETF.
# A possible explanation is that they use T=100 (unit-less),
# which is way too short, and so they might have gotten "lucky".
# Another explanation is that our implementation is buggy,
# but this seems unlikely, especially since we reproduce the NETF
# results from Wiljes'2017.
