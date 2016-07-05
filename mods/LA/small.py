# Small-ish

# NB: Since there is no noise, and the system is stable,
#     the benchmarks obtained from this configuration
#     - go to zero as T-->infty
#     - are highly dependent on the initial error.

from common import *

from mods.LA.fundamentals import Fmat, homogeneous_1D_cov

m = 100
p = 4
obsInds = equi_spaced_integers(m,p)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1)

#def step(x,t,dt):
  #return np.roll(x,1,axis=x.ndim-1)
Fm = Fmat(m,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

f = {
    'm': m,
    'model': step,
    'noise': 0
    }

X0 = GaussRV(C=homogeneous_1D_cov(m,m/8,kind='Gauss'))

@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

h = {
    'm': p,
    'model': hmod,
    'noise': GaussRV(C=0.01*eye(p))
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

params = OSSE(f,h,tseq,X0,**other)



####################
# Suggested tuning
####################

#cfg.N = 100
