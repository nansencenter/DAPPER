# Settings from
# title={Second-order accurate ensemble transform particle filters},
# author={de Wiljes, Jana and Acevedo, Walter and Reich, Sebastian},

from common import *

from mods.Lorenz63.core import step, dfdx

m = 3


t = Chronology(0.01,dkObs=12,T=4**5,BurnIn=4)

f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

jj = array([0])
h = partial_direct_obs_setup(m,jj)
h['noise'] = 8
from tools.localization import no_localization
h['loc_f'] = no_localization(m,jj)

other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)

####################
# Suggested tuning
####################
# Reproduce benchmarks for NETF and ESRF (here EnKF-N)
# from left pane of Fig 1.
#from mods.Lorenz63.wiljes2017 import setup ################ Expected RMSE_a:
#cfgs += EnKF_N(N=5)                                        # 2.67
#cfgs += EnKF_N(N=30,rot=True)                              # 2.43
#cfgs += LNETF(N=40,rot=True,infl=1.02,Rs=1.0,loc_rad='NA') # 2.13
#cfgs += iEnKS('-N', N=10,Lag=2)                            # 2.10
#cfgs += PartFilt(N=35 ,reg=1.4,NER=0.3)                    # 1.86 not finely tuned



