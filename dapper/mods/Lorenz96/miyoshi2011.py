"""A land-ocean setup.

Refs: `bib.miyoshi2011gaussian`, which was inspired by `bib.lorenz1998optimal`.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96.sakov2008 import X0, Dyn, LPs, Nx, Tplot
from dapper.tools.localization import nd_Id_localization

# Use small dt to "cope with" ocean sector blow up
# (due to spatially-constant infl)
OneYear = 0.05 * (24/6) * 365
t = modelling.Chronology(0.005, dto=0.05, T=110*OneYear,
                         Tplot=Tplot, BurnIn=10*OneYear)

land_sites  = np.arange(Nx//2)
ocean_sites = np.arange(Nx//2, Nx)

jj = land_sites
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 1
Obs['localizer'] = nd_Id_localization((Nx,), (1,), jj)

HMM = modelling.HiddenMarkovModel(
    Dyn, Obs, t, X0,
    LP=LPs(jj),
    sectors={'land': land_sites,
             'ocean': ocean_sites},
)

####################
# Suggested tuning
####################

# Reproduce Miyoshi Figure 5
# --------------------------------------------------------------------------
# print(xps.tabulate_avrgs(["rmse.a", "rmse.land.a", "rmse.ocean.a"]))
# xps += LETKF(N=10,rot=False, infl=sqrt(1.015),loc_rad=3) # 2.1 , 0.38, 2.9

# It can be seen that Miyoshi's "Global RMSE"
# is just the average of the land and ocean RMSE's,
# which explains why this differs so much from DAPPER's
# (conventionally defined) global RMSE.
