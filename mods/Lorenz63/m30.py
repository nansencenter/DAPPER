# Moderate dtObs and non-0 Q.

from common import *

from mods.Lorenz63.core import step, dfdx

m = 3
p = m

#T = 4**6
T = 4**4
t = Chronology(0.01,dkObs=15,T=T,BurnIn=4)

m = 3
f = {
    'm'    : m,
    'model': lambda x,t,dt: step(x,t,dt),
    'jacob': dfdx,
    'noise': GaussRV(C=2,m=m)
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=0.5,mu=mu0)

h = {
    'm'    : p,
    'model': lambda x,t: x,
    'jacob': lambda x,t: eye(3),
    'noise': GaussRV(C=2,m=p)
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
