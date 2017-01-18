import numpy as np
from aux.misc import rk4, is1d
from common import ens_compatible, integrate_TLM

# Constants
sig = 10.0; rho = 28.0; beta = 8.0/3

@ens_compatible
def dxdt(x):
  """
  Same as dxdt(), but with transposing in wrapper.
  """
  d    = np.zeros_like(x)
  d[0] = sig*(x[1] - x[0])
  d[1] = rho*x[0] - x[1] - x[0]*x[2]
  d[2] = x[0]*x[1] - beta*x[2]
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
