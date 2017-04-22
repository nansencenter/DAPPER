# Concerns figure 5 of Todter and Ahrens (2015):
# "A Second-Order Exact Ensemble Square Root Filter for Nonlinear Data Assimilation"
from common import *

from mods.Lorenz95 import core

t = Chronology(0.05,dkObs=2,T=4**3,BurnIn=20)

m = 80
f = {
    'm'    : m,
    'model': core.step,
    'noise': 0
    }

X0 = GaussRV(m=m, C=0.001)

jj = arange(0,m,2)
p  = len(jj)
@atmost_2d
def hmod(E,t): return E[:,jj]
H = direct_obs_matrix(m,jj)


from aux.localization import inds_and_coeffs, unravel
dIJ = unravel(arange(m), m)
oIJ = unravel(jj, m)
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
    'model': hmod,
    'jacob': lambda x,t: H,
    'noise': GaussRV(C=1.0*eye(p)),
    'loc_f': locf,
    }
 

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
# We obtain better rmse results for the LETKF than the paper.
# But, of course, the major difference to the paper is that
# we do not use exponential observation noise, but rather Gaussian.
#config = LETKF(N=20,rot=True,infl=1.04,loc_rad=5) # rmse_a = 0.46

