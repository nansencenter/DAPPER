# Uses nU, J, F as in core.py, which is taken from Wilks2005.
# Obs settings taken from different places (=> quasi-linear regime).

from common import *
from mods.LorenzUV.core import model_instance
LUV = model_instance()
nU = LUV.nU

# Wilks2005 uses dt=1e-4 with RK4 for the full model,
# and dt=5e-3 with RK2 for the forecast/truncated model.
# As berry2014linear notes, this is possible coz
# "numerical stiffness disappears when fast processes are removed".

################
# Full
################

#t = Chronology(dt=0.001,dtObs=0.05,T=4**3,BurnIn=6) # allows using rk2
t = Chronology(dt=0.005,dtObs=0.05,T=4**3,BurnIn=6)  # requires rk4


f = {
    'm'    : LUV.m,
    'model': with_rk4(LUV.dxdt,autonom=True),
    'noise': 0,
    'jacob': LUV.dfdx,
    'plot' : LUV.plot_state
    }

X0 = GaussRV(C=0.01*eye(LUV.m))

R = 0.1
h = partial_direct_obs_setup(LUV.m,arange(LUV.nU))
h['noise'] = R

other = {'name': rel_path(__file__,'mods/')+'_full'}
setup_full = TwinSetup(f,h,t,X0,**other)


################
# Truncated
################

# Just change dt from 005 to 05
t = Chronology(dt=0.05, dtObs=0.05,T=4**3,BurnIn=6)

f = {
    'm'    : nU,
    'model': with_rk4(LUV.dxdt_parameterized),
    'noise': 0,
    }

X0 = GaussRV(C=0.01*eye(nU))

h = partial_direct_obs_setup(nU,arange(nU))
h['noise'] = R
 
other = {'name': rel_path(__file__,'mods/')+'_trunc'}
setup_trunc = TwinSetup(f,h,t,X0,**other)


def polynom_prmzt(t,x,order):
  """
  Polynomial (deterministic) parameterization of fast variables (Y).
  
  NB: Only valid for system settings of Wilks'2005.

  Note: In order to observe an improvement in DA performance w
        higher orders, the EnKF must be reasonably tuned with 
        There is very little improvement gained above order=1.
  """
  if   order==4:
    # From Wilks
    d -= 0.262 + 1.45*x - 0.0121*x**2 - 0.00713*x**3 + 0.000296*x**4
  elif order==3:
    # From Arnold
    d -= 0.341 + 1.30*x - 0.0136*x**2 - 0.00235*x**3
  elif order==1:
    # From me -- see AdInf/illust_parameterizations.py
    d -= 0.74 + 0.82*x
  elif order==0:
    # From me -- see AdInf/illust_parameterizations.py
    d -= 3.82
  elif order==-1:
    # Leave as dxdt_trunc
    pass
  else:
    raise NotImplementedError
  return d




####################
# Suggested tuning
####################
#                                                         # Expected RMSE_a:
#cfgs += Climatology()                                    # 0.93
#cfgs += Var3D()                                          # 0.38
#cfgs += EnKF_N(N=20)                                     # 0.27
