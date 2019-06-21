# Reproduce experiments from 
# "Application of a hybrid EnKF-OI to ocean forecasting"
# by F. Counillon, P. Sakov, and L. Bertino (2009).

from dapper import *
from dapper.mods.QG.sak08 import HMM
from dapper.mods.QG.core import model_config

dt = 1.25 * 10 # 10 steps between obs (also requires dkObs=1)
HMM.t = Chronology(dt=dt,dkObs=1,T=1000*dt,BurnIn=10*dt)

HMM.Dyn.model = model_config("cou09_ens"  ,{"dtout":dt, 'RKH2':2.0e-11}).step
truth_model   = model_config("cou09_truth",{"dtout":dt}                ).step

####################
# Suggested tuning
####################
# Reproduce Table 1 results.
# - Note that Counillon et al:
#    - Report forecast rmse's (but they are pretty close to analysis rmse anyways).
#    - Use enkf-matlab which has a bug which cause them to report the
#      wrong localization radius (see mods/QG/sak08.py).
#      Eg. enkf-matlab radius 15 (resp 25) corresponds to DAPPER radius 10.6 (resp 17.7).

# R = 17.7 # equiv. to R=25 in enkf-matlab
# from dapper.mods.QG.cou09 import HMM  , truth_model        # Expected RMSE_f:
# cfgs += LETKF(mp=True, N=25,infl=1.15,taper='Gauss',loc_rad=R)      # 1.11
# cfgs += LETKF(mp=True, N=15,infl=1.35,taper='Gauss',loc_rad=R)      # 1.2
#
# - My N=15 rmse << rmse_from_paper. But I only tested a single repetition => maybe I got lucky.
#
# - Use this to turn on/off the truth-model before/after truth simulation:
#   with set_tmp(HMM.Dyn,'model',truth_model):
#     xx,yy = simulate(HMM)

