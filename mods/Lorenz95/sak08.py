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
    'm'    : p,
    'model': hmod,
    'TLM'  : lambda x,t: H,
    'noise': GaussRV(C=1*eye(p)),
    'plot' : yplot
    }


from aux.localization import inds_and_coeffs, unravel
def loc_wrapper(radius,direction=None):
  iix = arange(m)
  dIJ = unravel(iix    , m)
  oIJ = unravel(obsInds, m)
  if direction is 'x2y':
    def locf(i):
      return inds_and_coeffs(dIJ[:,i], oIJ, m, radius)
  elif direction is 'y2x':
    def locf(i):
      return inds_and_coeffs(oIJ[:,i], dIJ, m, radius)
  else: raise KeyError
  return locf

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)
setup.locf = loc_wrapper




####################
# Suggested tuning
####################

# Reproduce Sakov'2008 "deterministic"
#config = DAC(EnKF,'PertObs',N=40, infl=1.06)          # rmse_a = 0.22
#config = DAC(EnKF,'DEnKF',N=40, infl=1.01)            # rmse_a = 0.18
#config = DAC(EnKF,'PertObs',N=28,infl=1.08)           # rmse_a = 0.24
#config = DAC(EnKF,'Sqrt'   ,N=24,infl=1.013,rot=True) # rmse_a = 0.18

#config = DAC(iEnKF,'Sqrt',N=40,iMax=10,infl=1.01,rot=True) # rmse_a = 0.17

#config = DAC(LETKF,N=6,rot=True,infl=1.04,locf=setup.locf(4,'x2y'))
#config = DAC(LETKF,'approx',N=8,rot=True,infl=1.25,locf=setup.locf(4,'x2y'))
#config = DAC(SL_EAKF,N=6,rot=True,infl=1.07,locf=setup.locf(6,'y2x'))
#
#config = DAC(Climatology)
#config = DAC(D3Var)
#config = DAC(ExtKF, infl = 1.05)
#config = DAC(EnCheat,'Sqrt',N=24,infl=1.02,rot=True)



# Reproduce Bocquet'2015 "expanding"
# setup.t.T     = 4**4
# setup.t.dt    = 0.05
# setup.t.dkObs = 3
# #
# #config.N          = 20
# ##config.infl       = 1.02 # dkObs = 1
# ##config.infl       = 1.10 # dkObs = 3
# ##config.infl       = 1.40 # dkObs = 5
# #config.upd_a    = 'Sqrt'
# #config.rot        = False
# #config.da_driver  = EnKF
# #
# config.da_driver = EnKF_N
# config.N         = 20
# config.infl      = 1.0
# config.rot       = False
# #
# #setup.t.dt    = 0.01
# #setup.t.dkObs = 1

