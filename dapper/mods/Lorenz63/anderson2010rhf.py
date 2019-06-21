# As in Anderson 2010 rank histogram filter



from dapper import *

from dapper.mods.Lorenz63.core import step, dfdx, x0, Tplot

t = Chronology(0.01, dkObs=12, KObs=1000, Tplot=Tplot, BurnIn=4*Tplot)

Nx = len(x0)

Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(C=2,mu=x0)

Obs = partial_direct_Obs(Nx,arange(Nx))
Obs['noise'] = 8.0


HMM = HiddenMarkovModel(Dyn,Obs,t,X0)




####################
# Suggested tuning
####################
# Compare with Anderson's figure 10.
# Benchmarks are fairly reliable (KObs=2000): 
# from dapper.mods.Lorenz63.anderson2010rhf import HMM    # rmse_a
# cfgs += SL_EAKF(N=20,infl=1.01,rot=True,loc_rad=np.inf) # 0.87
# cfgs += EnKF_N (N=20,rot=True)                          # 0.87
# cfgs += RHF    (N=50,infl=1.10)                         # 1.28
# cfgs += RHF    (N=50,infl=0.95,rot=True)                # 0.94
# cfgs += RHF    (N=20,infl=0.95,rot=True)                # 1.07


