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
 
other = {'name': os.path.relpath(__file__,'mods/')}

params = OSSE(f,h,t,X0,**other)



####################
# Suggested tuning
####################

#cfg.N = 40

# rmse_a = 0.22
#cfg.infl    = 1.045 # Requires BurnIn inflation too
#cfg.infl    = 1.06
#cfg.AMethod = 'PertObs non-transposed'
#method      = EnKF

# rmse_a = 0.175
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#method      = EnKF

# rmse_a = 0.18
#cfg.infl    = 1.01
#cfg.AMethod = 'DEnKF'
#method      = EnKF

# rmse_a = 0.17
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#cfg.iMax    = 10
#method      = iEnKF

