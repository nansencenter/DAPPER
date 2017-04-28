# Reproduce results from fig2 of
# Raanes, Patrick Nima, Alberto Carrassi, and Laurent Bertino (2015):
# "Extending the square root method to account for additive forecast noise in ensemble methods."

# Warnings:
# - Multidim. multiplicative noise incorporation
#   has a tendency to go awry.
# - The main reason is that it changes the ensemble subspace,
#   away from the "model" subspace.
# - There are also some very strong, regular correlation
#   patters that arise when dt=1 (dt = c*dx).
# - It also happens if X0pat does not use centering.

from common import *

from mods.LA.core import sinusoidal_sample, Fmat

# Burn-in allows damp*x and x+noise balance out
tseq = Chronology(dt=1,dkObs=5,T=500,BurnIn=60)

m = 1000;
p = 40;

jj = equi_spaced_integers(m,p)
h = partial_direct_obs_setup(m,jj)
h['noise'] = 0.01


################### Noise setup ###################
# Instead of sampling model noise from sinusoidal_sample(),
# we will replicate it below by a covariance matrix approach.
# But, for strict equivalence, one would have to use
# uniform (i.e. not Gaussian) random numbers.
wnumQ = 25
fname = 'data/LA_Q_wnum' + str(wnumQ) + '_v2' + '.npz'
try:
  # Load pre-generated
  L = np.load(fname)['Left']
except FileNotFoundError:
  # First-time use
  NQ        = 20000; # Must have NQ > (2*wnumQ+1)
  A         = sinusoidal_sample(m,wnumQ,NQ)
  A         = 1/10 * anom(A)[0] / sqrt(NQ)
  Q         = A.T @ A
  U,s,_     = tsvd(Q)
  L         = U*sqrt(s)
  np.savez(fname, Left=L)

X0 = GaussRV(C=CovMat(sqrt(5)*L,'Left'))

################### Forward model ###################
damp = 0.98;
Fm = Fmat(m,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

f = {
    'm'    : m,
    'model': lambda x,t,dt: damp * step(x,t,dt),
    'jacob': lambda x,t,dt: damp * Fm,
    'noise': GaussRV(C=CovMat(L,'Left')),
    }

################### Gather ###################
other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,tseq,X0,**other)



####################
# Suggested tuning
####################

## Expected rmse_a = 0.3
#config = EnKF('PertObs',N=30,infl=3.2)
#
# But infl=1 yields approx optimal rmse, even though then rmv << rmse.
# TODO: Why is rmse so INsensitive to inflation for PertObs?
# Similar case, but with N=60: infl=1.00, and 1.80.

# Reproduce raanes'2015 "extending sqrt method to model noise":
# config = EnKF('Sqrt',fnoise_treatm='XXX',N=30,infl=1.0),
# where XXX is one of:
# - Stoch
# - Mult-1
# - Mult-m
# - Sqrt-Core
# - Sqrt-Add-Z
# - Sqrt-Dep
