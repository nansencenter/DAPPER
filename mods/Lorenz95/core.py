# Note how everything is ndim-agnostic.
# TLM is also agnostic about forcing.

import numpy as np
from scipy.linalg import circulant
from aux.misc import rk4, integrate_TLM, is1d

  
def dxdt(x,Force):
  a = x.ndim-1
  s = lambda x,n: np.roll(x,-n,axis=a)
  return np.multiply(s(x,1)-s(x,-2), s(x,-1)) - x + Force

def step(x0, t, dt, Force=8.0):
  return rk4(lambda t,x: dxdt(x,Force), x0, np.nan, dt)


def TLM(x):
  """Tangent linear model"""
  assert is1d(x)
  m    = len(x)
  TLM  = np.zeros((m,m))
  md   = lambda i: np.mod(i,m)
  for i in range(m):
    TLM[i,i]       = -1.0
    TLM[i,   i-2 ] = -x[i-1]
    TLM[i,md(i+1)] = +x[i-1]
    TLM[i,   i-1 ] = x[md(i+1)]-x[i-2]
  return TLM

def dfdx(x,t,dt):
  """Integral of TLM. Jacobian of step."""
  resolvent = np.eye(len(x)) + dt*TLM(x) # Approximate
  #resolvent = integrate_TLM(TLM(x),dt)   # Exact
  return resolvent

def typical_init_params(m):
  """
  Approximate (3 degrees of acf of) climatology.
  Obtained for F=8, m=40.
  """
  mu0 = 2.34*np.ones(m)
  # Auto-cov-function
  acf = lambda i: 0.0 + 14*(i==0) + 0.9*(i==1) - 4.7*(i==2) - 1.2*(i==3)
  P0  = circulant(acf(periodic_distance_range(m)))
  return mu0, P0

def periodic_distance_range(m):
  return np.minimum(np.arange(m),np.arange(m,0,-1))
  #return np.roll(np.abs(np.arange(m) - m//2), (m+1)//2)
  #return np.concatenate((range((m+1)//2), range(m//2,0,-1)))


