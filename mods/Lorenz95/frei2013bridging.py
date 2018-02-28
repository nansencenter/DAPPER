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


from common import *

from mods.Lorenz95.core import step, dfdx
from tools.localization import partial_direct_obs_1d_loc_setup as loc

t = Chronology(0.05,dtObs=0.4,T=4**5,BurnIn=20)

m = 40
f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

X0 = GaussRV(m=m, C=0.001)

jj = 1 + arange(0,m,2)
h = partial_direct_obs_setup(m,jj)
h['noise'] = 0.5
h['loc_f'] = loc(m,jj)

other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,t,X0,**other)



####################
# Suggested tuning
####################
# Compare to Table 1 and 3 from frei2013bridging. Note:
#  - N is too large to be very interesting.
#  - We obtain better EnKF scores than they report,
#    and use inflation and sqrt updating,
#    and don't really need localization.
#from mods.Lorenz95.frei2013bridging import setup           # rmse_a
#cfgs += EnKF_N(N=400,rot=1)                                # 0.80
#cfgs += LETKF(N=400,rot=True,infl=1.01,loc_rad=10*1.82)    # 0.79 # short experiment only
#cfgs += Var3D(infl=0.8)                                    # ≈2.5 # short experiment only


