"""Settings from Pajonk, Oliver, et al. 
'A deterministic filter for non-Gaussian Bayesian estimation—applications
to dynamical system estimation with noisy measurements.'
Physica D: Nonlinear Phenomena 241.7 (2012): 775-788.

There is nothing to reproduce from the paper as there are no
statistically converged numbers.

More interesting settings: mods.Lorenz84.harder
"""

from dapper import *
import dapper as dpr

from dapper.mods.Lorenz84.core import step, dstep_dx, x0
from dapper.mods.Lorenz63.core import LPs

Nx = len(x0)
Ny = Nx

day = 0.05/6 * 24 # coz dt=0.05 <--> 6h in "model time scale"
t = Chronology(0.05,dkObs=1,T=200*day,BurnIn=10*day)

Dyn = {
    'M'      : Nx,
    'model'  : step,
    'linear' : dstep_dx,
    'noise'  : 0
}

# X0 = dpr.GaussRV(C=0.01,M=Nx) # Decreased from Pajonk's C=1.
X0 = dpr.GaussRV(C=0.01,mu=x0)

jj = np.arange(Nx)
Obs = dpr.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.1

HMM = HiddenMarkovModel(Dyn,Obs,t,X0,LP=LPs(jj))

####################
# Suggested tuning
####################
# xps += ExtKF(infl=2)
# xps += EnKF('Sqrt',N=3,infl=1.01)
# xps += PartFilt(reg=1.0, N=100, NER=0.4) # add reg!
# xps += PartFilt(reg=1.0, N=1000, NER=0.1) # add reg!
