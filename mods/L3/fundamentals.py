import numpy as np
from misc import rk4

def dxdt(x):
  if np.ndim(x) is 1:
    x = x.reshape((1,len(x)))
  fx = np.zeros_like(x)
  fx[:,0] = 10.0*(x[:,1] - x[:,0])
  fx[:,1] = 28.0*x[:,0] - x[:,1] - x[:,0]*x[:,2]
  fx[:,2] = x[:,0]*x[:,1] - (8.0/3)*x[:,2]
  if fx.shape[0] is 1:
    fx = fx.squeeze()
  return fx

def step(x0, t0, dt):
    return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)

