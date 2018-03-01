# Reproduce results from Table 1 of Sakov et al "DEnKF" (2008)
# This setup is also used in several other papers
# (bocquet2012"combining",bocquet2015"expanding", raanes2016"EnRTS", ...)

from common import *

from mods.Lorenz95.core import step, dfdx, typical_init_params
from tools.localization import partial_direct_obs_1d_loc_setup as loc

t = Chronology(0.05,dkObs=1,T=4**5,BurnIn=20)

m = 40
f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(m=m, C=0.001) 
#X0 = GaussRV(*typical_init_params(m))

h = {
    'm'    : m,
    'model': Id_op(),
    'jacob': Id_mat(m),
    'noise': 1, # abbrev GaussRV(C=CovMat(eye(m)))
    'plot' : lambda y: plt.plot(y,'g')[0],
    'loc_f': loc(m,arange(m)),
    }


other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,t,X0,**other)



####################
# Suggested tuning
####################
# Reproduce Sakov'2008 "deterministic"                    # Expected RMSE_a:
#cfgs += EnKF('PertObs',N=40, infl=1.06)                  # 0.22
#cfgs += EnKF('DEnKF'  ,N=40, infl=1.01)                  # 0.18
#cfgs += EnKF('PertObs',N=28, infl=1.08)                  # 0.24
#cfgs += EnKF('Sqrt'   ,N=24, infl=1.013,rot=True)        # 0.18

# Other
#cfgs += EnKF_N(N=24,rot=True) # no tuning!               # 0.21
#cfgs += EnKF_N(N=24,rot=True,xN=2.0)                     # 0.18
#cfgs += iEnKS('Sqrt',N=40,infl=1.01,rot=True)            # 0.17

# Localized
#cfgs += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4) # 0.22
#cfgs += LETKF(approx=1,N=8,rot=True,infl=1.25,loc_rad=4) # 0.36
#cfgs += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6) # 0.23
# Reproduce LETKF scores from Bocquet'2011 "EnKF-N" fig 6:
#cfgs += LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')

# Other
#cfgs += Climatology()                                    # 3.6 
#cfgs += OptInterp()                                      # 0.95 
#cfgs += Var3D_Lag(infl=0.5)
#cfgs += Var3D(infl=1.05)                                 # 0.41 
#cfgs += ExtKF(infl=10)                                   # 0.24 


# Reproduce Fig 3 of Bocquet'2015 "expanding"
#cfgs += EnKF('Sqrt',N=20,rot=True,infl=1.04)             # 0.20
# # use infl=1.10 with dkObs=3
# # use infl=1.40 with dkObs=5
# cfgs += EnKF_N(N=20)                                    # 0.24
# cfgs += EnKF_N(N=20,xN=2)                               # 0.19
# # Also try quasi-linear regime:
# t = Chronology(0.01,dkObs=1,T=4**4,BurnIn=20)

# Reproduce Fig 7a of Bocquet'2014 "an iterative EnKS"
# Similar settings also in Fig 7.3 of DA book by Asch, Bocquet, Nodet.
#cfgs += iEnKS('-N', N=20,Lag=10,xN=2.0)                  # 0.163
#cfgs += iEnKS('-N', N=20,Lag=20,xN=2.0)                  # 0.160 
# using setup.t.dkObs = 8:
#cfgs += iEnKS('-N', N=20,Lag=2,xN=1.0)                   # 0.39
#
# Fig 2 (smoother averages):
#cfgs += iEnKS('-N', N=20,Lag=10,xN=2.0)                  # 0.163
# The analysis-time smoother averages can be computed as
# mean(s[indx].rmse.u[setup.t.kkObs_BI])
# while universal-time averages are given by 
#print_averages(cfgs,avrgs,[],['rmse_u'])


# Tests with the Particle filter, with N=3000, KObs=10'000.
# da_method  NER  reg  |  rmse_a   rmv_a
# --------- ----  ---  -  ------  ------
# PartFilt  0.05  1.2  |  0.35    0.40  
# PartFilt  0.05  1.6  |  0.41    0.45  
# PartFilt  0.5   0.7  |  0.26    0.29  
# PartFilt  0.5   0.9  |  0.30    0.34  
# PartFilt  0.5   1.2  |  0.36    0.40  
# Using NER=0.9 yielded rather poor results.
