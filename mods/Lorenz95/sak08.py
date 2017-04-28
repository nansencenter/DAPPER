# Reproduce results from
# table1 of sakov et al "DEnKF" (2008)
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
setup = OSSE(f,h,t,X0,**other)



####################
# Suggested tuning
####################

# Reproduce Sakov'2008 "deterministic"
#config = EnKF('PertObs',N=40, infl=1.06)           # rmse_a = 0.22
#config = EnKF('DEnKF'  ,N=40, infl=1.01)           # rmse_a = 0.18
#config = EnKF('PertObs',N=28, infl=1.08)           # rmse_a = 0.24
#config = EnKF('Sqrt'   ,N=24, infl=1.013,rot=True) # rmse_a = 0.18

# Other
#config = iEnKF('Sqrt',N=40,iMax=10,infl=1.01,rot=True) # rmse_a = 0.17
#
#config = LETKF(         N=6,rot=True,infl=1.04,loc_rad=4)
#config = LETKF(approx=1,N=8,rot=True,infl=1.25,loc_rad=4)
#config = SL_EAKF(       N=6,rot=True,infl=1.07,loc_rad=6)
#
#config = Climatology()
#config = D3Var()
#config = ExtKF(infl=6)
#config = EnCheat('Sqrt',N=24,infl=1.02,rot=True)

# Reproduce LETKF scores from Bocquet'2011 "EnKF-N" fig 6.
#config = LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')



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
