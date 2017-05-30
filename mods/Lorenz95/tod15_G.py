# Like tod15 but with Gaussian likelihood.

from common import *
from mods.Lorenz95.tod15 import setup
setup.h.noise = GaussRV(C=setup.h.noise.C)
setup.name = os.path.relpath(__file__,'mods/')

####################
# Suggested tuning
####################

#                                                          rmse_a
#cfgs += LETKF(N=40,rot=True,infl=1.04       ,loc_rad=5) # 0.42
#cfgs += LETKF(N=80,rot=True,infl=1.04       ,loc_rad=5) # 0.42

#cfgs += LNETF(N=40,rot=True,infl=1.10,Rs=1.9,loc_rad=5) # 0.54
#cfgs += LNETF(N=80,rot=True,infl=1.06,Rs=1.4,loc_rad=5) # 0.47


