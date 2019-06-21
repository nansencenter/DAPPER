# As in:
#  - Frei, Marco, and Hans R. Künsch.
#    "Bridging the ensemble Kalman and particle filters."
#    Biometrika 100.4 (2013): 781-800.
# who aloso cite its use in:
#  - BENGTSSON, T., SNYDER, C. & NYCHKA, D. (2003).
#    "Toward a nonlinear ensemble filter for high-dimensional systems."
#    J. Geophys. Res. 108, 8775.
#  - LEI, J. & BICKEL, P. (2011).
#    "A moment matching ensemble filter for nonlinear non-Gaussian data assimilation."
#    Mon. Weather Rev. 139, 3964–73
#  - FREI, M. & KUNSCH H. R. (2013).
#    "Mixture ensemble Kalman filters"
#    Comp. Statist. Data Anal. 58, 127–38.


from dapper import *

from dapper.mods.Lorenz95.core import step, dfdx
from dapper.tools.localization import partial_direct_obs_nd_loc_setup as loc_setup

t = Chronology(0.05,dtObs=0.4,T=4**5,BurnIn=20)

Nx = 40
Dyn = {
    'M'    : Nx,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(M=Nx, C=0.001)

jj = 1 + arange(0,Nx,2)
Obs = partial_direct_Obs(Nx,jj)
Obs['noise'] = 0.5
Obs['localizer'] = loc_setup( (Nx,), (2,), jj, periodic=True )

HMM = HiddenMarkovModel(Dyn,Obs,t,X0)



####################
# Suggested tuning
####################
# Compare to Table 1 and 3 from frei2013bridging. Note:
#  - N is too large to be very interesting.
#  - We obtain better EnKF scores than they report,
#    and use inflation and sqrt updating,
#    and don't really need localization.
# from dapper.mods.Lorenz95.frei2013bridging import HMM      # rmse_a
# cfgs += EnKF_N(N=400,rot=1)                                # 0.80
# cfgs += LETKF(N=400,rot=True,infl=1.01,loc_rad=10/1.82)    # 0.79 # short experiment only
# cfgs += Var3D(infl=0.8)                                    # ≈2.5 # short experiment only


