# A mix of Evensen'2009 and Sakov'2008

# NB: Since there is no noise, and the system is stable,
#     the rmse's from this HMM go to zero as T-->infty.
#     => benchmarks largely depend on the initial error,
#     and so these absolute rmse values are not so useful
#     for quantatative evaluation of DA methods.
#     For that purpose, see mods/LA/raanes2015.py instead.

from dapper import *

from dapper.mods.LA.core import sinusoidal_sample, Fmat
from dapper.mods.Lorenz95.core import LPs

Nx = 1000
Ny = 4
jj = equi_spaced_integers(Nx,Ny)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1,Tplot=100)

# WITHOUT explicit matrix (assumes dt == dx/c):
# step = lambda x,t,dt: np.roll(x,1,axis=x.ndim-1)
# WITH:
Fm = Fmat(Nx,c=-1,dx=1,dt=tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': Fm,
    'noise': 0
    }

# In the animation, it can sometimes/somewhat occur
# that the truth is outside 3*sigma !!!
# Yet this is not so implausible because sinusoidal_sample()
# yields (multivariate) uniform (random numbers) -- not Gaussian.
wnum  = 25
X0 = RV(M=Nx, func = lambda N: sqrt(5)/10 * sinusoidal_sample(Nx,wnum,N))

Obs = partial_direct_Obs(Nx,jj)
Obs['noise'] = 0.01

HMM = HiddenMarkovModel(Dyn,Obs,tseq,X0,LP=LPs(jj))


####################
# Suggested tuning
####################
# config = EnKF('PertObs',N=100,infl=1.02)

