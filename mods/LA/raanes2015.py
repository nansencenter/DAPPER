# Reproduce results from fig2
# of raanes'2015 "extending sqrt method to model noise"

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

m = 1000;
p = 40;

obsInds = equi_spaced_integers(m,p)

# Burn-in allows damp*x and x+noise balance out
tseq = Chronology(dt=1,dkObs=5,T=500,BurnIn=60)

@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

H = zeros((p,m))
for i,j in enumerate(obsInds):
  H[i,j] = 1.0

def yplot(y):
  lh = plt.plot(obsInds,y,'g*',ms=8)[0]
  #plt.pause(0.8)
  return lh

h = {
    'm': p,
    'model': hmod,
    'jacob': lambda x,t: H,
    'noise': GaussRV(C=0.01*eye(p)),
    'plot' : yplot,
    }



# Instead of sampling model noise from sinusoidal_sample(),
# we will replicate it below by a covariance matrix approach.
# But, for strict equivalence, one would have to use
# uniform (i.e. not Gaussian) random numbers.
wnumQ = 25
fname = 'data/LA_Q_wnum' + str(wnumQ) + '.npz'
try:
  # Load pre-generated
  Q = np.load(fname)['Q']
except FileNotFoundError:
  # First-time use
  NQ        = 20000; # Must have NQ > (2*wnumQ+1)
  A         = sinusoidal_sample(m,wnumQ,NQ)
  A         = 1/10 * anom(A)[0] / sqrt(NQ)
  Q         = A.T @ A
  np.savez(fname, Q=Q)

# TODO: Make GaussRV support init by chol,
#       test reproducibility, insert into loaded file.
#U,s,_ = tsvd(Q)
#Q12   = U*sqrt(s)
#X0    = GaussRV(C12 = sqrt(5)*Q12)

X0 = GaussRV(C = 5*Q)

damp = 0.98;
Fm = Fmat(m,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

f = {
    'm': m,
    'model': lambda x,t,dt: damp * step(x,t,dt),
    'jacob': lambda x,t,dt: damp * Fm,
    'noise': GaussRV(C = Q),
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,tseq,X0,**other)



####################
# Suggested tuning
####################

## Expected rmse_a = 0.3
#config = DAC(EnKF,'PertObs',N=30,infl=3.2)
#
# But infl=1 yields approx optimal rmse, even though then rmv << rmse.
# TODO: Why is rmse so INsensitive to inflation for PertObs?
# Similar case, but with N=60: infl=1.00, and 1.80.

# Reproduce raanes'2015 "extending sqrt method to model noise":
# config = DAC(EnKF,'Sqrt',fnoise_treatm='XXX',N=30,infl=1.0),
# where XXX is one of:
# - Stoch
# - Mult-1
# - Mult-m
# - Sqrt-Core
# - Sqrt-Add-Z
# - Sqrt-Dep
