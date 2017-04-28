# Concerns figure 5 of Todter and Ahrens (2015):
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
h['noise'] = 1.0
h['loc_f'] = loc(m,jj)

other = {'name': os.path.relpath(__file__,'mods/')}
setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
# We obtain better rmse results for the LETKF than the paper.
# But, of course, the major difference to the paper is that
# we do not use exponential observation noise, but rather Gaussian.
#cfgs += LETKF(N=20,rot=True,infl=1.04,loc_rad=5) # rmse_a = 0.46

