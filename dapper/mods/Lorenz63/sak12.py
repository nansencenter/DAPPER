"""Reproduce results from Table 1
Sakov, Oliver, Bertino (2012):
'An Iterative EnKF for Strongly Nonlinear Systems'"""

from dapper import *

from dapper.mods.Lorenz63.core import step, dstep_dx, x0, Tplot, LPs

t = Chronology(0.01, dkObs=25, KObs=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M'      : Nx,
    'model'  : step,
    'linear' : dstep_dx,
    'noise'  : 0
    }

X0 = GaussRV(C=2,mu=x0)

jj = arange(Nx) # obs_inds
Obs = partial_Id_Obs(Nx, jj)
Obs['noise'] = 2 # GaussRV(C=CovMat(2*eye(Nx)))

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################
# from dapper.mods.Lorenz63.sak12 import HMM       # Expected rmse.a:
# cfgs += Climatology()                                     # 7.6
# cfgs += OptInterp()                                       # 1.25
# cfgs += Var3D(xB=0.1)                                     # 1.04
# cfgs += ExtKF(infl=180)                                   # 0.92
# cfgs += EnKF('Sqrt',   N=3 ,  infl=1.30)                  # 0.80
# cfgs += EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.60
# cfgs += EnKF('PertObs',N=10,  infl=1.04)                  # 0.65
# cfgs += EnKF('PertObs',N=100, infl=1.01)                  # 0.56
# cfgs += EnKF_N(        N=3)                               # 0.60
# cfgs += EnKF_N(        N=10,            rot=True)         # 0.54
# cfgs += iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)         # 0.31
# cfgs += PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
# cfgs += PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
# cfgs += PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
# cfgs += PFxN(xN=1000,  N=30  ,Qs=2   ,NER=0.2)            # 0.56
