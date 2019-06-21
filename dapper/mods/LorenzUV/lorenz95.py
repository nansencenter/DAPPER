# As in Lorenz 1996 "Predictability..."

from dapper import *

from dapper.mods.LorenzUV.core import model_instance
LUV = model_instance(nU=36,J=10,F=10)
nU = LUV.nU






################
# Full
################

# 
t = Chronology(dt=0.005,dtObs=0.05,T=4**3,BurnIn=6)


Dyn = {
    'M'    : LUV.M,
    'model': with_rk4(LUV.dxdt,autonom=True),
    'noise': 0,
    'jacob': LUV.dfdx,
    }

X0 = GaussRV(C=0.01*eye(LUV.M))

R = 1.0
jj = arange(LUV.nU)
Obs = partial_direct_Obs(LUV.M,jj)
Obs['noise'] = R

other = {'name': rel_path(__file__,'mods/')+'_full'}
HMM_full = HiddenMarkovModel(Dyn,Obs,t,X0,**other)


################
# Truncated
################

# Just change dt from 005 to 05
t = Chronology(dt=0.05, dtObs=0.05,T=4**3,BurnIn=6)

Dyn = {
    'M'    : nU,
    'model': with_rk4(LUV.dxdt_parameterized),
    'noise': 0,
    }

X0 = GaussRV(C=0.01*eye(nU))

jj = arange(nU)
Obs = partial_direct_Obs(nU,jj)
Obs['noise'] = R
 
other = {'name': rel_path(__file__,'mods/')+'_trunc'}
HMM_trunc = HiddenMarkovModel(Dyn,Obs,t,X0,**other)

####################
# Suggested tuning
####################
#                     # Expected RMSE_a ("U"-vars only!)
