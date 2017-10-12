# Uses nX, J, F as in core.py, which is taken from Wilks2005.

from common import *

from mods.LorenzXY.core import nX,m,dxdt,dfdx,plot_state

# Wilks2005 uses dt=1e-4 with RK4 for the full model,
# and dt=5e-3 with RK2 for the forecast/truncated model.
# As berry2014linear notes, this is possible coz
# "numerical stiffness disappears when fast processes are removed".

#t = Chronology(dt=0.001,dtObs=0.05,T=4**3,BurnIn=6) # allows using rk2
t = Chronology(dt=0.005,dtObs=0.05,T=4**3,BurnIn=6)  # requires rk4

f = {
    'm'    : m,
    'model': with_rk4(dxdt,autonom=True,order=4),
    'noise': 0,
    'jacob': dfdx,
    'plot' : plot_state
    }

X0 = GaussRV(C=0.01*eye(m))

h = partial_direct_obs_setup(m,arange(nX))
h['noise'] = 0.1

other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)


####################
# Suggested tuning
####################
#                                                         # Expected RMSE_a:
#cfgs += Climatology()                                    # 0.93
#cfgs += Var3D()                                          # 0.38
#cfgs += EnKF_N(N=20)                                     # 0.27
