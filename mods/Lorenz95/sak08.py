# Reproduce results from Table 1 of Sakov et al "DEnKF" (2008)
# This setup is also used in several other papers
# (bocquet2012"combining",bocquet2015"expanding", raanes2016"EnRTS", ...)

from common import *

from mods.Lorenz95.core import step, dfdx, typical_init_params
from aux.localization import partial_direct_obs_1d_loc_setup as loc

t = Chronology(0.05,dkObs=1,T=4**5,BurnIn=20)

m = 40
f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(*typical_init_params(m))

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
#cfgs += iEnKF('Sqrt',N=40,iMax=10,infl=1.01,rot=True)    # 0.17

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


# Reproduce Bocquet'2015 "expanding"
# t = Chronology(0.05,dkObs=3,T=4**4,BurnIn=20)
# config = EnKF('Sqrt',N=20)
# # config.infl = 1.02 # case: dkObs=1
# # config.infl = 1.10 # case: dkObs=3
# # config.infl = 1.40 # case: dkObs=5
#
# config = EnKF_N('Sqrt',N=20)
#
# # Also try quasi-linear regime:
# t = Chronology(0.01,dkObs=1,T=4**4,BurnIn=20)
