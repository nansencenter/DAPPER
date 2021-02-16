"""From `dapper.mods.Lorenz96.todter2015` again, but with Gaussian likelihood."""

import dapper.mods as modelling
from dapper.mods.Lorenz96.todter2015 import HMM as _HMM

HMM = _HMM.copy()
HMM.Obs.noise = modelling.GaussRV(C=HMM.Obs.noise.C)

####################
# Suggested tuning
####################

#                                                          rmse.a
# xps += LETKF(N=40,rot=True,infl=1.04       ,loc_rad=5) # 0.42
# xps += LETKF(N=80,rot=True,infl=1.04       ,loc_rad=5) # 0.42

# xps += LNETF(N=40,rot=True,infl=1.10,Rs=1.9,loc_rad=5) # 0.54
# xps += LNETF(N=80,rot=True,infl=1.06,Rs=1.4,loc_rad=5) # 0.47
