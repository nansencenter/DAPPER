# Reproduce results from fig2 of
# Raanes, Patrick Nima, Alberto Carrassi, and Laurent Bertino (2015):
# "Extending the square root method to account for additive forecast noise in ensemble methods."

from dapper import *

from dapper.mods.LA.core import sinusoidal_sample, Fmat
from dapper.mods.Lorenz95.core import LPs

# Burn-in allows damp*x and x+noise balance out
tseq = Chronology(dt=1,dkObs=5,T=500,BurnIn=60,Tplot=100)

Nx = 1000;
Ny = 40;

jj = equi_spaced_integers(Nx,Ny)
Obs = partial_direct_Obs(Nx,jj)
Obs['noise'] = 0.01


################### Noise setup ###################
# Instead of sampling model noise from sinusoidal_sample(),
# we will replicate it below by a covariance matrix approach.
# But, for strict equivalence, one would have to use
# uniform (i.e. not Gaussian) random numbers.
wnumQ = 25
sample_filename = os.path.join(dirs['samples'],'LA_Q_wnum%d.npz'%wnumQ)

try:
  # Load pre-generated
  L = np.load(sample_filename)['Left']
except FileNotFoundError:
  # First-time use
  print('Did not find sample file',sample_filename,
      'for experiment initialization. Generating...')
  NQ        = 20000 # Must have NQ > (2*wnumQ+1)
  A         = sinusoidal_sample(Nx,wnumQ,NQ)
  A         = 1/10 * center(A)[0] / sqrt(NQ)
  Q         = A.T @ A
  U,s,_     = tsvd(Q)
  L         = U*sqrt(s)
  np.savez(sample_filename, Left=L)

X0 = GaussRV(C=CovMat(sqrt(5)*L,'Left'))

################### Forward model ###################
damp = 0.98;
Fm = Fmat(Nx,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

Dyn = {
    'M'    : Nx,
    'model': lambda x,t,dt: damp * step(x,t,dt),
    'jacob': lambda x,t,dt: damp * Fm,
    'noise': GaussRV(C=CovMat(L,'Left')),
    }

################### Gather ###################
HMM = HiddenMarkovModel(Dyn,Obs,tseq,X0,LP=LPs(jj))



####################
# Suggested tuning
####################

## Expected rmse_a = 0.3
# config = EnKF('PertObs',N=30,infl=3.2)
# Note that infl=1 may yield approx optimal rmse, even though then rmv << rmse.
# Why is rmse so INsensitive to inflation, especially for PertObs?

# Reproduce raanes'2015 "extending sqrt method to model noise":
# config = EnKF('Sqrt',fnoise_treatm='XXX',N=30,infl=1.0),
# where XXX is one of:
# - Stoch
# - Mult-1
# - Mult-M
# - Sqrt-Core
# - Sqrt-Add-Z
# - Sqrt-Dep

# Other notes:
# - Multidim. multiplicative noise incorporation
#   has a tendency to go awry.
#   The main reason is that it changes the ensemble subspace,
#   away from the "model" subspace.
# - There are also some very strong, regular correlation
#   patters that arise when dt=1 (dt = c*dx).
# - It also happens if X0pat does not use centering.



