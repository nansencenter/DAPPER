# Settings not taken from anywhere

from dapper import *

from dapper.mods.DoublePendulum.core import step, x0, LP_setup

t = Chronology(0.01,dtObs=1,T=30,BurnIn=10)

Dyn = {
    'M'    : len(x0),
    'model': step,
    'noise': 0
    }

X0 = GaussRV(mu=x0,C=0.01**2)

jj = [0,2]
Obs = partial_direct_Obs(len(x0),jj)
Obs['noise'] = 0.1**2

HMM = HiddenMarkovModel(Dyn,Obs,t,X0,LP=LP_setup(jj))

####################
# Suggested tuning
####################
# config = EnKF('Sqrt',N=20   ,infl=1.02, rot=True)   # 1.66
# config = PartFilt(   N=400  ,reg=2.0,NER=0.9)       # 0.23
# config = PartFilt(   N=1000 ,reg=1.0,NER=0.9)       # 0.19
# TODO: implement cartesian obs
# TODO: implement physical viz (see demo) for liveplotting
