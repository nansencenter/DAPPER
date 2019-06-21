"""Reproduce results from Table 1
Sakov, Oliver, Bertino (2012):
'An Iterative EnKF for Strongly Nonlinear Systems'"""

from dapper import *

from dapper.mods.Lorenz63.core import step, dfdx, x0, Tplot, LPs

t = Chronology(0.01, dkObs=25, KObs=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(C=2,mu=x0)

jj = arange(Nx) # obs_inds
Obs = partial_direct_Obs(Nx, jj)
Obs['noise'] = 2 # GaussRV(C=CovMat(2*eye(Nx)))

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################
# from dapper.mods.Lorenz63.sak12 import HMM       # Expected RMSE_a:
# cfgs += Climatology()  # no tuning!                       # 7.6
# cfgs += OptInterp()    # no tuning!                       # 1.25
# cfgs += Var3D(infl=0.9)# tuning not strictly required     # 1.03 
# cfgs += ExtKF(infl=90) # some inflation tuning needed     # 0.87
# cfgs += EnKF('Sqrt',   N=3 ,  infl=1.30)                  # 0.82
# cfgs += EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.63
# cfgs += EnKF('PertObs',N=500, infl=0.97,rot=False)        # 0.56
# cfgs += EnKF_N(        N=3) # no tuning!                  # 0.60
# cfgs += EnKF_N(        N=10,            rot=True)         # 0.54
# cfgs += iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)         # 0.31
# cfgs += PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
# cfgs += PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
# cfgs += PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
# cfgs += PFxN(xN=1000,  N=30  ,Qs=2   ,NER=0.2)            # 0.56
