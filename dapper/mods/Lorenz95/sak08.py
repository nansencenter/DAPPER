# This HMM is also used (with small variations) in many DA papers.
# First (exact) use: Ott et al (2004) "A local EnKF for atmospheric DA" ?
# See list below of other papers that use it.

from dapper import *

from dapper.mods.Lorenz95.core import step, dfdx, x0, Tplot, LPs
from dapper.tools.localization import partial_direct_obs_nd_loc_setup as loc_setup

t = Chronology(0.05, dkObs=1, KObs=1000, Tplot=Tplot, BurnIn=2*Tplot)

Nx = 40
x0 = x0(Nx)

Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(mu=x0, C=0.001) 

jj = arange(Nx) # obs_inds
Obs = partial_direct_Obs(Nx, jj)
Obs['noise'] = 1
Obs['localizer'] = loc_setup( (Nx,), (2,), jj, periodic=True )

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################

# Reproduce Table1 of Sakov'2008 "deterministic"                # Expected RMSE_a:
# --------------------------------------------------------------------------------
# cfgs += EnKF('PertObs'        ,N=40, infl=1.06)               # 0.22
# cfgs += EnKF('DEnKF'          ,N=40, infl=1.01)               # 0.18
# cfgs += EnKF('PertObs'        ,N=28, infl=1.08)               # 0.24
# cfgs += EnKF('Sqrt'           ,N=24, infl=1.013,rot=True)     # 0.18

# Other analysis schemes:
# cfgs += EnKF('Serial'         ,N=28, infl=1.02,rot=True)      # 0.18
# cfgs += EnKF('Serial ESOPS'   ,N=28, infl=1.02)               # 0.18
# cfgs += EnKF('Serial Stoch'   ,N=28, infl=1.08)               # 0.24
# EnKF-N
# cfgs += EnKF_N(N=24,rot=True) # no tuning!                    # 0.21
# cfgs += EnKF_N(N=24,rot=True,xN=2.0)                          # 0.18
# Baseline methods
# cfgs += Climatology()                                         # 3.6 
# cfgs += OptInterp()                                           # 0.95 
# cfgs += Var3D_Lag(infl=0.5)
# cfgs += Var3D(infl=1.05)                                      # 0.41 
# cfgs += ExtKF(infl=10)                                        # 0.24 

# Reproduce LETKF scores from Bocquet'2011 "EnKF-N" fig 6:
# --------------------------------------------------------------------------------
# cfgs += LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')  # 
# Other localized:
# cfgs += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4)      # 0.22
# cfgs += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)      # 0.23

# Reproduce Table 3 (IEnKF) from sakov2012iterative
# --------------------------------------------------------------------------------
# HMM.t.dkObs = 12
# cfgs += iEnKS('Sqrt' ,N=25,Lag=1,nIter=10,infl=1.2,rot=1)     # 0.46

# Reproduce Fig 3 of Bocquet'2015 "expanding"
# --------------------------------------------------------------------------------
# cfgs += EnKF('Sqrt',N=20,rot=True,infl=1.04)                  # 0.20
# # use infl=1.10 with dkObs=3
# # use infl=1.40 with dkObs=5
# cfgs += EnKF_N(N=20)                                          # 0.24
# cfgs += EnKF_N(N=20,xN=2)                                     # 0.19
# # Also try quasi-linear regime:
# t = Chronology(0.01,dkObs=1,...)

# Reproduce Bocquet/Sakov'2013 "Joint...", Fig 4, i.e. dtObs=0.2:
# cfgs += iEnKS('Sqrt', N=20, Lag=4, xN=2) # 0.31
# cfgs += Var4D(Lag=1,xB=0.2)              # 0.46
# cfgs += Var4D(Lag=2,xB=0.1)              # 0.39
# cfgs += Var4D(Lag=4,xB=0.02)             # 0.37
# Cannot reproduce Fig4's reported 4D-Var scores for L>4. Example:
# cfgs += Var4D(Lag=6,xB=0.015)            # 0.385 Boc13 reports 0.33

# Tests with the Particle filter, with N=3000, KObs=10'000.
# da_method  NER  reg  |  rmse_a   rmv_a
# --------- ----  ---  -  ------  ------
# PartFilt  0.05  1.2  |  0.35    0.40  
# PartFilt  0.05  1.6  |  0.41    0.45  
# PartFilt  0.5   0.7  |  0.26    0.29  
# PartFilt  0.5   0.9  |  0.30    0.34  
# PartFilt  0.5   1.2  |  0.36    0.40  
# Using NER=0.9 yielded rather poor results.


