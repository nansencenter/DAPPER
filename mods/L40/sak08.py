# Reproduce results from
# table1 of sakov et al "DEnKF" (2008)

from common import *

from mods.L40.fundamentals import step, typical_init_params

#T = 4**6
T = 4**3
t = Chronology(0.05,dkObs=1,T=T,BurnIn=20)

m = 40
f = {
    'm': m,
    'model': lambda x,t,dt: step(x,t,dt),
    'noise': 0
    }


X0 = GaussRV(*typical_init_params(m))

p = m
h = {
    'm': p,
    'model': lambda x,t: x,
    'noise': GaussRV(C=1*eye(p))
    }
 
other = {'name': os.path.basename(__file__)}

params = OSSE(f,h,t,X0,**other)

# # rmse_a = 0.22
# N = 40
# #infl = 1.045 # Requires BurnIn inflation too
# infl = 1.06
# AMethod = 'PertObs'

# # rmse_a = 0.175
# N = 40
# infl = 1.01
# AMethod = 'Sqrt'
# rot = True
