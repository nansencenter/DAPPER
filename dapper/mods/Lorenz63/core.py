# "Lorenz-63"  model. Classic exhibitor of chaos.
# Phase-plot looks like a butterfly.
# See demo.py for more info.


import numpy as np
from dapper.tools.math import with_rk4, is1d, ens_compatible, integrate_TLM

# Constants
sig = 10.0; rho = 28.0; beta = 8.0/3

# Dynamics: time derivative.
@ens_compatible
def dxdt(x):
  d     = np.zeros_like(x)
  x,y,z = x
  d[0]  = sig*(y - x)
  d[1]  = rho*x - y - x*z
  d[2]  = x*y - beta*z
  return d

# Dynamics: time step integration.
step = with_rk4(dxdt,autonom=True)

# Time span for plotting. Typically: ≈10 * "system time scale".
Tplot = 4.0

# Example initial state.
# Specifics are usually not important coz system is chaotic,
# and we employ a BurnIn before averaging statistics.
# But it's often convenient to give a point on the attractor,
# or at least its basin, or at least ensure that it's "physical".
x0 = np.array([1.509, -1.531, 25.46])


################################################
# OPTIONAL (not necessary for EnKF or PartFilt):
################################################
def TLM(x):
  """Tangent linear model"""
  x,y,z = x
  A = np.array(
      [[-sig , sig , 0],
      [rho-z , -1  , -x],
      [y     , x   , -beta]])
  return A

def dfdx(x,t,dt):
  """Integral of TLM. Jacobian of step."""
  return integrate_TLM(TLM(x),dt,method='approx')


################################################
# Add some non-default liveplotters
################################################
import dapper.tools.liveplotting as LP
params = dict(labels='xyz', Tplot=1)
def LPs(jj=None,params=params): return [
    (14, 1, LP.sliding_marginals(jj, zoomy=0.8, **params)) ,
    (13, 1, LP.phase3d(jj, **params)                     ) ,
    ]
