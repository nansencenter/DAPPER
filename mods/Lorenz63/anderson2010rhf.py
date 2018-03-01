# As in Anderson 2010 rank histogram filter



from common import *

from mods.Lorenz63.core import step, dfdx
from tools.localization import no_localization as loc

m = 3
p = m

t = Chronology(0.01,dkObs=12,T=4**5,BurnIn=4)

f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(C=1,mu=ones(m))

h = partial_direct_obs_setup(m,arange(m))
h['noise'] = 8.0
h['loc_f'] = loc(m,arange(m))

other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)


####################
# Suggested tuning
####################
# Compare with Anderson's figure 10.
# Benchmarks are fairly reliable (KObs=2000): 
#from mods.Lorenz63.anderson2010rhf import setup         # rmse_a
#cfgs += SL_EAKF(N=20,infl=1.01,rot=True,loc_rad=np.nan) # 0.87
#cfgs += EnKF_N (N=20,rot=True)                          # 0.87
#cfgs += RHF    (N=50,infl=1.10)                         # 1.28
#cfgs += RHF    (N=50,infl=0.95,rot=True)                # 0.94
#cfgs += RHF    (N=20,infl=0.95,rot=True)                # 1.07


