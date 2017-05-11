# Concerns figure 4 of Todter and Ahrens (2015):
# "A Second-Order Exact Ensemble Square Root Filter for Nonlinear Data Assimilation"
from common import *

from mods.Lorenz95 import core
from aux.localization import partial_direct_obs_1d_loc_setup as loc

t = Chronology(0.05,dkObs=2,T=4**4,BurnIn=20)

m = 80
f = {
    'm'    : m,
    'model': core.step,
    'noise': 0
    }

X0 = GaussRV(m=m, C=0.001)

jj = arange(0,m,2)
h = partial_direct_obs_setup(m,jj)
h['loc_f'] = loc(m,jj)
#h['noise'] = LaplaceRV(C=1,m=len(jj))
h['noise'] = LaplaceParallelRV(C=1,m=len(jj))

other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,t,X0,**other)

####################
# Suggested tuning
####################

#                                                          rmse_a
#cfgs += LETKF(N=20,rot=True,infl=1.04       ,loc_rad=5) # 0.47
#cfgs += LETKF(N=40,rot=True,infl=1.04       ,loc_rad=5) # 0.45
#cfgs += LETKF(N=80,rot=True,infl=1.04       ,loc_rad=5) # 0.43
#cfgs += LNETF(N=80,rot=True,infl=1.10,Rs=1.4,loc_rad=5) # 0.43
#cfgs += LNETF(N=40,rot=True,infl=1.10,Rs=2.5,loc_rad=5) # 0.55

# Compared to the reference paper:
# - Our rmse scores for the LETKF are better
# - Our rmse scores for the NETF
#   - with N>80 are similar
#   - with N<80 are worse
# In addition to regular inflation, note that we also
# find the necessity of tuning the inflation (Rs) for R,
# which is not mentioned in the reference.

# This pretty much voids the merit of the NETF,
# especially considering that the Laplace obs-error case
# should favour the NETF.

# Caveat: maybe our implementation is buggy?
# Another explanation might be that the paper uses T=100,
# which is too short, and means that they might have gotten "lucky".


