import numpy as np
from aux.misc import rk4, is1d
from common import atmost_2d

@atmost_2d
def dxdt(x):
  fx = np.zeros_like(x)
  fx[:,0] = 10.0*(x[:,1] - x[:,0])
  fx[:,1] = 28.0*x[:,0] - x[:,1] - x[:,0]*x[:,2]
  fx[:,2] = x[:,0]*x[:,1] - (8.0/3)*x[:,2]
  return fx

def dfdx(x,t,dt):
  """
  Jacobian of x + dt*dxdt.
  """
  assert is1d(x)
  m  = len(x)
  F  = np.eye(3) + dt*np.array([
    [-10, 10, 0],
    [28-x[2], -1, -x[0]],
    [x[1], x[0], -8/3]])
  return F

def step(x0, t0, dt):
    return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)

