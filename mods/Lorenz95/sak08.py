# Reproduce results from
# table1 of sakov et al "DEnKF" (2008)

from common import *

from mods.Lorenz95.fundamentals import step, dfdx, typical_init_params

T = 4**5
t = Chronology(0.05,dkObs=1,T=T,BurnIn=20)

m = 40
f = {
    'm': m,
    'model': lambda x,t,dt: step(x,t,dt),
    'TLM'  : dfdx,
    'noise': 0
    }


mu0,P0 = typical_init_params(m)
X0 = GaussRV(mu0, 0.01*P0)

p = m
obsInds = equi_spaced_integers(m,p)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

H = zeros((p,m))
for i,j in enumerate(obsInds):
  H[i,j] = 1.0

#yplot = lambda y: plt.plot(y,'g*',MarkerSize=15)[0]
#yplot = lambda y: plt.plot(y,'g')[0]
def yplot(y):
  lh = plt.plot(y,'g')[0]
  #plt.pause(0.8)
  return lh

h = {
    'm': p,
    'model': hmod,
    'TLM'  : lambda x,t: H,
    'noise': GaussRV(C=1*eye(p)),
    'plot' : yplot
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)



####################
# Suggested tuning
####################


# Reproduce Sakov'2008 "deterministic"
#cfg.N = 40
# rmse_a = 0.22
#cfg.infl    = 1.06
#cfg.infl    = 1.045 # Requires BurnIn inflation too
#cfg.AMethod = 'PertObs non-transposed'
#method      = EnKF
#
# rmse_a = 0.175
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#method      = EnKF
#
# rmse_a = 0.18
#cfg.infl    = 1.01
#cfg.AMethod = 'DEnKF'
#method      = EnKF
#
# rmse_a = 0.17
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#cfg.iMax    = 10
#method      = iEnKF


#cfg.da_method = ExtKF; cfg.infl = 1.05



# Reproduce Bocquet'2015 "expanding"
# setup.t.T     = 4**4
# setup.t.dt    = 0.05
# setup.t.dkObs = 3
# #
# #cfg.N          = 20
# ##cfg.infl       = 1.02 # dkObs = 1
# ##cfg.infl       = 1.10 # dkObs = 3
# ##cfg.infl       = 1.40 # dkObs = 5
# #cfg.AMethod    = 'Sqrt'
# #cfg.rot        = False
# #cfg.da_method  = EnKF
# #
# cfg.da_method = EnKF_N
# cfg.N         = 20
# cfg.infl      = 1.0
# cfg.rot       = False
# #
# #setup.t.dt    = 0.01
# #setup.t.dkObs = 1

