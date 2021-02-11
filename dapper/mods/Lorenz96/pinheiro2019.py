"""Settings from `bib.pinheiro2019efficient`."""
import dapper.mods as modelling
import dapper.mods.Lorenz96 as model
from dapper.mods.Lorenz96 import LPs, step, x0
from dapper.mods.utils import linspace_int
from dapper.tools.localization import nd_Id_localization

model.Force = 8.17
t = modelling.Chronology(0.01, dkObs=10, K=4000, Tplot=10, BurnIn=10)

Nx = 1000

Dyn = {
    'M': Nx,
    'model': step,
    'noise': 0,  # not stated
}

X0 = modelling.GaussRV(mu=x0(Nx), C=0.001)

jj = linspace_int(Nx, Nx//4, periodic=True)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.1**2
Obs['localizer'] = nd_Id_localization((Nx,), (2,), jj, periodic=True)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0)

HMM.liveplotters = LPs(jj)


# They use
# - N = 30, though this is only really stated for their PF.
# - loc_rad = 4, but don't state its definition, nor the exact tapering function.
# - infl factor 1.05 (don't state whether it's squared)
# Anyway this seem to be pretty good tuning factors.
# They find RMSE's (figure 8) hovering around 0.8 for the LETKF.
# I.e. that "the new filter consistency outperforms the LETKF".
# In DAPPER, however, the LETKF scores around 0.07 with these settings:
#                                                   # Expected rmse.a:
# xps += da.OptInterp()                                        # 0.04
# xps += da.EnKF('Sqrt', N=500, infl=1.04)                     # 0.06
# xps += da.LETKF(N=10, loc_rad=4, infl=1.05, rot=False)       # 0.07
# xps += da.LETKF(N=30, loc_rad=4, infl=1.05, rot=False)       # 0.07
