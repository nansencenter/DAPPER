# Settings not taken from anywhere

from dapper import *

from dapper.mods.DoublePendulum.core import step, x0, LP_setup, dstep_dx

t = Chronology(0.01,dkObs=100,T=30,BurnIn=10)

Dyn = {
    'M'     : len(x0),
    'model' : step,
    'noise' : 0,
    'linear': dstep_dx,
}

X0 = GaussRV(mu=x0,C=0.01**2)

jj = [0,2]
Obs = partial_Id_Obs(len(x0),jj)
Obs['noise'] = 0.1**2

HMM = HiddenMarkovModel(Dyn,Obs,t,X0,LP=LP_setup(jj))

####################
# Suggested tuning
####################
# from dapper.mods.DoublePendulum.dpr01 import HMM      # Expct rmse.a:

# HMM.t.dkObs = anything
# cfgs += Climatology()                                 # 5
# cfgs += OptInterp()                                   # 2.5

# HMM.t.dkObs = 7 # <-- max dkObs for tolerable performance with Var3D
# cfgs += Var3D(xB=0.1)                                 # 0.81

# HMM.t.dkObs = 10
# cfgs += ExtKF(infl=2)                                 # 0.12
# cfgs += EnKF('Sqrt', N=3  ,infl=1.03,rot=True)        # 0.12
# cfgs += EnKF('Sqrt', N=5  ,infl=1.01,rot=True)        # 0.10
# cfgs += PartFilt(    N=40  ,reg=1.0,NER=0.9)          # 0.12

# HMM.t.dkObs = 20 # <-- max dkObs for tolerable performance with ExtKF
# cfgs += ExtKF(infl=3)                                 # 0.18
# cfgs += EnKF('Sqrt', N=20 ,infl=1.02,rot=True)        # 0.15
# cfgs += PartFilt(    N=100  ,reg=1.0,NER=0.9)         # 0.13
# cfgs += PartFilt(    N=400  ,reg=1.0,NER=0.9)         # 0.12
# cfgs += PartFilt(    N=1000 ,reg=1.0,NER=0.9)         # 0.11

# HMM.t.dkObs = 30 # <-- for larger dkObs, EnKF deteriorates quickly
# cfgs += EnKF('Sqrt', N=20 ,infl=1.01,rot=True)        # 0.17
# cfgs += PartFilt(    N=100  ,reg=1.0,NER=0.9)         # 0.14
# cfgs += PartFilt(    N=400  ,reg=1.0,NER=0.9)         # 0.13
# cfgs += PartFilt(    N=1000 ,reg=1.0,NER=0.9)         # 0.12

# HMM.t.dkObs = 60
# cfgs += EnKF( 'Sqrt',N=20      ,infl=1.01,rot=True)   # 0.41
# cfgs += iEnKS('Sqrt',N=10,Lag=2,infl=1.01,rot=True)   # 0.14
# cfgs += PartFilt(    N=400  ,reg=1.0,NER=0.9)         # 0.15

# HMM.t.dkObs=100
# cfgs += EnKF('Sqrt', N=40,rot=True,infl=1.01)         # 1.9
# cfgs += iEnKS('Sqrt',N=10,Lag=1,infl=1.01,rot=True)   # 0.19
# cfgs += PartFilt(    N=400  ,reg=2.0,NER=0.9)         # 0.23
# cfgs += PartFilt(    N=1000 ,reg=1.0,NER=0.9)         # 0.19

# TODO: implement cartesian obs
