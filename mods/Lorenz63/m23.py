# For testing initial implementation of iEnKF

from common import *

from mods.Lorenz63.core import step

m = 3
p = m

#T = 4**6
T = 4**3
t = Chronology(0.01,0.05,T,4)

f = {
    'm'    : m,
    'model': lambda x,t,dt: step(x,t,dt),
    'noise': 0
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

h = {
    'm'    : p,
    'model': lambda x,t: x,
    'noise': GaussRV(C=2,m=p)
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################

## rmse_a = 0.205
#config.N       = 10
#config.infl    = 1.00001
#config.upd_a = 'Sqrt'
#config.rot     = False
#method      = EnKF

# rmse_a = 0.185
#config.N       = 10
#config.infl    = 1.00001
#config.upd_a = 'Sqrt'
#config.rot     = False
#config.iMax    = 10
#method      = iEnKF

