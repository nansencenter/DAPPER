from dapper import *

from dapper.mods.Lorenz96.core import LPs
from dapper.tools.localization import Id_Obs_nd_loc_setup as loc_setup

from dapper.mods.KS.core import Model, Tplot
KS = Model(dt=0.5)
Nx = KS.Nx

# nRepeat=10
t = Chronology(KS.dt, dkObs=2, KObs=2*10**4, BurnIn=2*10**3, Tplot=Tplot)

Dyn = {
    'M'     : Nx,
    'model' : KS.step,
    'linear': KS.dstep_dx,
    'noise' : 0
    }

X0 = GaussRV(mu=KS.x0, C=0.001) 

Obs = Id_Obs(Nx)
Obs['noise'] = 1
Obs['localizer'] = loc_setup( (Nx,), (4,), periodic=True )

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)

HMM.liveplotters = LPs(np.arange(Nx))


####################
# Suggested tuning
####################

# Reproduce (top-right panel) of Fig. 4 of bocquet2019consistency     # Expected rmse.a:
# --------------------------------------------------------------------------------
# cfgs += LETKF(N=4 , loc_rad=15/1.82, infl=1.11,rot=True,taper='GC') # 0.18
# cfgs += LETKF(N=6,  loc_rad=25/1.82, infl=1.06,rot=True,taper='GC') # 0.14
# cfgs += LETKF(N=16, loc_rad=51/1.82, infl=1.02,rot=True,taper='GC') # 0.11
#
# Other:
# cfgs += Climatology()                                               # 1.3
# cfgs += OptInterp()                                                 # 0.5
# cfgs += EnKF('Sqrt', N=13,           infl=1.60,rot=True)            # 0.5
# cfgs += EnKF('Sqrt', N=20,           infl=1.03,rot=True)            # 0.115
