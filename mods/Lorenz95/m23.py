# For ?
from common import *

from mods.Lorenz95.core import step, typical_init_params

#T = 4**6
T = 4**3
t = Chronology(0.05,dkObs=3,T=T,BurnIn=20)

m = 40
f = {
    'm'    : m,
    'model': step,
    'noise': GaussRV(C=3e-2 * eye(m))
    }

X0 = GaussRV(*typical_init_params(m))

p = m
h = {
    'm'    : p,
    'model': lambda x,t: x,
    'noise': GaussRV(C=0.1*eye(p)),
    'plot' : lambda y: plt.plot(y,'g')[0]
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
#config = DAC(EnKF,'Sqrt',N=40,infl=1.10,rot=False)
