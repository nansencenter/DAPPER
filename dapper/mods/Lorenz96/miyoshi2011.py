"""A land-ocean setup for Lorenz-96 from Miyoshi'2011: 'The Gaussian Approach to
Adaptive Covariance Inflation and Its Implementation with the LETKF'

inspired by MWR 1998 by E. N. Lorenz and K. A. Emanuel:
"Optimal Sites for Supplementary Weather Observations: Simulation with a Small Model"
"""


import numpy as np

import dapper as dpr
from dapper.mods.Lorenz96.sakov2008 import X0, Dyn, LPs, Nx, Tplot
from dapper.tools.localization import nd_Id_localization

# Use small dt to "cope with" ocean sector blow up
# (due to spatially-constant infl)
OneYear = 0.05 * (24/6) * 365
t = dpr.Chronology(0.005, dtObs=0.05, T=110*OneYear, Tplot=Tplot, BurnIn=10*OneYear)

land_sites  = np.arange(Nx//2)
ocean_sites = np.arange(Nx//2, Nx)

jj = land_sites
Obs = dpr.partial_Id_Obs(Nx, jj)
Obs['noise'] = 1
Obs['localizer'] = nd_Id_localization((Nx,), (1,), jj)

HMM = dpr.HiddenMarkovModel(
    Dyn, Obs, t, X0,
    LP=LPs(jj),
    sectors={'land': land_sites, 'ocean': ocean_sites}
)

####################
# Suggested tuning
####################

# Reproduce Miyoshi Figure 5                              # rmse.a  rmse.land.a  rmse.ocean.a
# ---------------------------------------------------------------------------------------------
# xps += LETKF(N=10,rot=False,infl=sqrt(1.015),loc_rad=3) # 2.1     0.38         2.9

# It can be seen that Miyoshi's "Global RMSE" is just the average of the land and ocean RMSE's,
# which explains why this differs so much from DAPPER's (conventionally defined) global RMSE.
