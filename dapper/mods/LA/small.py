# Smaller version.
# => convenient for debugging, e.g., scripts/test_iEnKS.py

from dapper import *

from dapper.mods.LA.core import Fmat, homogeneous_1D_cov
from dapper.mods.Lorenz95.core import LPs

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1,Tplot=100)

Nx = 100

# def step(x,t,dt):
  # return np.roll(x,1,axis=x.ndim-1)
Fm = Fmat(Nx,-1,1,tseq.dt)
def step(x,t,dt):
  assert dt == tseq.dt
  return x @ Fm.T

Dyn = {
    'M': Nx,
    'model': step,
    'noise': 0
    }

X0 = GaussRV(C=homogeneous_1D_cov(Nx,Nx/8,kind='Gauss'))

Ny  = 4
jj = equi_spaced_integers(Nx,Ny)
Obs  = partial_direct_Obs(Nx,jj)
Obs['noise'] = 0.01

 
HMM = HiddenMarkovModel(Dyn,Obs,tseq,X0,LP=LPs(jj))


####################
# Suggested tuning
####################
# cfgs += EnKF('PertObs',N=16 ,infl=1.02)
# cfgs += EnKF('Sqrt'   ,N=16 ,infl=1.0)

