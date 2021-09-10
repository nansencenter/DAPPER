"""Settings from `bib.pinheiro2019efficient`."""
import dapper.mods as modelling
import dapper.mods.Lorenz96 as model
from dapper.mods.Lorenz96 import LPs, step, x0
from dapper.mods.utils import linspace_int
from dapper.tools.localization import nd_Id_localization

model.Force = 8.17
tseq = modelling.Chronology(0.01, dko=10, K=4000, Tplot=10, BurnIn=10)

Nx = 1000

Dyn = {
    'M': Nx,
    'model': step,
    # It's not clear from the paper whether Q=0.5 or 0.25.
    # But I'm pretty sure it's applied each dto (not dt).
    'noise': 0.25 / tseq.dto,
    # 'noise': 0.5 / t.dto,
}

X0 = modelling.GaussRV(mu=x0(Nx), C=0.001)

jj = linspace_int(Nx, Nx//4, periodic=True)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 0.1**2
Obs['localizer'] = nd_Id_localization((Nx,), (1,), jj, periodic=True)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(jj)


# Pinheiro et al. use
# - N = 30, but this is only really stated for their PF.
# - loc-rad = 4, but don't state its definition, nor the exact tapering function.
# - infl-factor 1.05, but I'm not sure if it's squared or not.
#
# They find RMSE (figure 8) hovering around 0.8 for the LETKF,
# and conclude that "the new filter consistency outperforms the LETKF".
# I cannot get the LETKF to work well with so much noise,
# but we note that optimal interpolation (a relatively unsophisticated method)
# scores around 0.56:
#                                                   # Expected rmse.a:
# xps += da.OptInterp()                                        # 0.56
#
# The following scores were obtained WITHOUT model noise.
# They can be used as an indication of how much noise there is in the system.
# xps += da.EnKF('Sqrt', N=500, infl=1.04)                     # 0.06
# xps += da.LETKF(N=10, loc_rad=4, infl=1.05, rot=False)       # 0.07
# xps += da.LETKF(N=30, loc_rad=4, infl=1.05, rot=False)       # 0.07
