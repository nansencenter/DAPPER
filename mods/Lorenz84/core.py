# "Lorenz-84"  model.
# Like Lorenz-63, the Lorenz-84 is also a strange attractor of dimension 3,
# but it has a more complex geometry.
# 
# Refs:
# E.N. Lorenz, Irregularity: a fundamental property of the atmosphere,
#   Tellus A 36 (1984) 98–110.
# E.N. Lorenz, Alook at some details of the growth of initial uncertainties,
#   Tellus A 57 (2005) 1–11

import numpy as np
from aux.misc import rk4, is1d
from common import ens_compatible, integrate_TLM

# Constants
a = 0.25;  b = 4; F = 8.0; G = 1.23;
#G = 1.0

@ens_compatible
def dxdt(x):
  d     = np.zeros_like(x)
  x,y,z = x
  d[0]  = - y**2 - z**2 - a*x + a*F
  d[1]  = x*y - b*x*z - y + G
  d[2]  = b*x*y + x*z - z
  return d


def step(x0, t0, dt):
    return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)

def TLM(x):
  """Tangent linear model"""
  assert is1d(x)
  x,y,z = x
  TLM=np.array(
      [[-a   , -2*y , -2*z],
      [y-b*z , -1   , -b*x],
      [b*y+z , b*x  , x-1]])
  return TLM

def dfdx(x,t,dt):
  """Integral of TLM. Jacobian of step."""
  return integrate_TLM(TLM(x),dt,method='approx')


