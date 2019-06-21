# Settings not taken from anywhere

from dapper import *

from dapper.mods.LotkaVolterra.core import step, dfdx, x0, LP_setup

# dt has been chosen after noting that 
# using dt up to 0.7 does not change the chaotic properties much,
# as adjudged with eye-ball and Lyapunov measures.

t = Chronology(0.5,dtObs=10,T=1000,BurnIn=10)

Nx = len(x0)

Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(mu=x0,C=0.01**2)

jj = [1,3]
Obs = partial_direct_Obs(Nx,jj)
Obs['noise'] = 0.04**2

HMM = HiddenMarkovModel(Dyn,Obs,t,X0,LP=LP_setup(jj))

####################
# Suggested tuning
####################
# Not carefully tuned:
# cfgs += EnKF_N(N=6)
# cfgs += ExtKF(infl=1.02)

