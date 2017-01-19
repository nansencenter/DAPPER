import numpy as np
from aux.misc import rk4, is1d
from common import ens_compatible, integrate_TLM

# Constants
sig = 10.0; rho = 28.0; beta = 8.0/3

@ens_compatible
def dxdt(x):
  d     = np.zeros_like(x)
  x,y,z = x
  d[0]  = sig*(y - x)
  d[1]  = rho*x - y - x*z
  d[2]  = x*y - beta*z
  return d

def step(x0, t0, dt):
    return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)

def TLM(x):
  """Tangent linear model"""
  assert is1d(x)
  x,y,z = x
  TLM=np.array(
      [[-sig, sig, 0],
      [rho-z, -1, -x],
      [y, x, -beta]])
  return TLM

def dfdx(x,t,dt):
  """Integral of TLM. Jacobian of step."""
  return integrate_TLM(TLM(x),dt,method='approx')
