# Reproduce results from fig2
# of raanes'2014 "extending sqrt method to model noise"

# Lessons:
# - Multidim. multiplicative noise incorporation
#   has a tendency to go awry.
# - The main reason is that it changes the ensemble subspace,
#   away from the "model" subspace.
# - There are also some very strong, regular correlation
#   patters that arise when dt=1 (dt = c*dx).
# - It also happens if X0pat does not use centering.

# TODO: Why is rmse performance so insensitive to inflation?
# For N=30, infl=3.4 yields correct rmsV (compared to rmse),
# but the rmse performance is actually slightly worse than
# with infl=1.0, which yields hugely underestimated rmse.

from common import *

from mods.LA.core import sinusoidal_sample, Fmat

m = 1000;
p = 40;

obsInds = equi_spaced_integers(m,p)

# Burn-in allows damp*x and x+noise balance out
tseq = Chronology(dt=1,dkObs=5,T=500,BurnIn=60)

@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

def yplot(y):
  lh = plt.plot(obsInds,y,'g*',MarkerSize=8)[0]
  plt.pause(0.8)
  return lh

h = {
    'm': p,
    'model': hmod,
    'noise': GaussRV(C=0.01*eye(p)),
    'plot' : yplot,
    }


wnum  = 25
X0 = RV(sampling_func = lambda N: \
    sqrt(5)/10 * sinusoidal_sample(m,wnum,N))



# sinusoidal_sample() can be replicated by a
# covariance matrix approach.
# However, for strict equivalence one would have to use
# uniform random numbers to multiply with
# the covariance (cholesky factor).
# This approach has a prescribed covariance matrix,
# and so it can be treated in the ensemble in interesting ways,
# (ie. not just additive sampling).
# But, for efficiency, only a m-by-rank cholesky factor
# should be specified.
wnumQ     = 25
NQ        = 2000 # should be at least 2*wnumQ+1
A         = sinusoidal_sample(m,wnumQ,NQ)
A         = 1/10 * anom(A)[0] / sqrt(NQ)
Q         = A.T @ A

#TODO:
# Instead of the above, generate huge (N) sample,
# compute its cov, so that:
#Q = load average_sinusoidal_cov.datafile
#Q = GaussRV(chol = sqrtm(Q))


damp = 0.98;
Fm = Fmat(m,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

f = {
    'm': m,
    'model': lambda x,t,dt: damp * step(x,t,dt),
    'noise': GaussRV(C = Q),
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,tseq,X0,**other)


####################
# Suggested tuning
####################
## Expected rmse_a = 0.3
#cfg.N         = 30
#cfg.infl      = 3.4 # Why is rmse performance so insensitive to inflation
#cfg.AMethod   = 'PertObs'
#cfg.rot       = False
#cfg.da_method = EnKF
