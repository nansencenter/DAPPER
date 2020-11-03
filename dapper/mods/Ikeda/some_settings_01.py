"""Setup parameters for twin experiments."""

from dapper import *
import dapper as dpr

from .core import step, x0, Tplot, LPs

t = Chronology(1, dkObs=1, KObs=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M'      : Nx,
    'model'  : step,
    'noise'  : 0
}

X0 = dpr.GaussRV(C=.1,mu=x0)

jj = np.arange(Nx) # obs_inds
Obs = dpr.partial_Id_Obs(Nx, jj)
Obs['noise'] = .1 # dpr.GaussRV(C=CovMat(1*eye(Nx)))

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################

