# Reproduce results from
# table1 of sakov et al "DEnKF" (2008)
# This setup is also used in several other papers
# (bocquet2012"combining",bocquet2015"expanding", raanes2016"EnRTS", ...)

from common import *

from mods.Lorenz95.core import step, dfdx, typical_init_params

T = 4**5
t = Chronology(0.05,dkObs=1,T=T,BurnIn=20)

m = 40
f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }


mu0,P0 = typical_init_params(m)
X0     = GaussRV(mu0, 0.01*P0)


p  = m
jj = equi_spaced_integers(m,p)
H  = direct_obs_matrix(m,jj)
@ens_compatible
def partial_direct_obs    (x,t): return x[jj]
def partial_direct_obs_jac(x,t): return H

def yplot(y):
  lh = plt.plot(y,'g')[0]
  return lh

# Localization setup
from aux.localization import inds_and_coeffs, unravel
dIJ = unravel(arange(m), m)
oIJ = unravel(jj , m)
def locf(radius,direction,t,tag=None):
  if direction is 'x2y':
    def locf_at(i):
      return inds_and_coeffs(dIJ[:,i], oIJ, m, radius, tag=tag)
  elif direction is 'y2x':
    def locf_at(i):
      return inds_and_coeffs(oIJ[:,i], dIJ, m, radius, tag=tag)
  else: raise KeyError
  return locf_at


h = {
    'm'    : p,
    'model': partial_direct_obs,
    'jacob': partial_direct_obs_jac,
    'noise': 1, # shortcut for GaussRV(C=CovMat(eye(p)))
    'plot' : yplot,
    'loc_f': locf,
    }


other = {'name': os.path.relpath(__file__,'mods/')}
setup = OSSE(f,h,t,X0,**other)



####################
# Suggested tuning
####################

# Reproduce Sakov'2008 "deterministic"
#config = EnKF('PertObs',N=40, infl=1.06)          # rmse_a = 0.22
#config = EnKF('DEnKF',N=40, infl=1.01)            # rmse_a = 0.18
#config = EnKF('PertObs',N=28,infl=1.08)           # rmse_a = 0.24
#config = EnKF('Sqrt'   ,N=24,infl=1.013,rot=True) # rmse_a = 0.18

# Other
#config = iEnKF('Sqrt',N=40,iMax=10,infl=1.01,rot=True) # rmse_a = 0.17
#
#config = LETKF(         N=6,rot=True,infl=1.04,loc_rad=4)
#config = LETKF('approx',N=8,rot=True,infl=1.25,loc_rad=4)
#config = SL_EAKF(       N=6,rot=True,infl=1.07,loc_rad=6)
#
#config = Climatology()
#config = D3Var()
#config = ExtKF(infl=6)
#config = EnCheat('Sqrt',N=24,infl=1.02,rot=True)

# Reproduce LETKF scores from Bocquet'2011 "EnKF-N" fig 6.
#config = LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')



# Reproduce Bocquet'2015 "expanding"
# t = Chronology(0.05,dkObs=3,T=4**4,BurnIn=20)
# config = EnKF('Sqrt',N=20)
# # config.infl = 1.02 # case: dkObs=1
# # config.infl = 1.10 # case: dkObs=3
# # config.infl = 1.40 # case: dkObs=5
#
# config = EnKF_N('Sqrt',N=20)
#
# # Also try quasi-linear regime:
# t = Chronology(0.01,dkObs=1,T=4**4,BurnIn=20)
