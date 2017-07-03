# "Lorenz-95" (or 96) model.
# 
# A summary for the purpose of DA is provided in
# section 3.5 of thesis found at
# ora.ox.ac.uk/objects/uuid:9f9961f0-6906-4147-a8a9-ca9f2d0e4a12
#
# A more detailed summary is given in Chapter 11 of 
# Majda, Harlim: Filtering Complex Turbulent Systems"
#
# Note: implementation is ndim-agnostic.
#
# Note: the model integration is unstable (--> infinity)
# in the presence of large peaks in amplitude,
# Example: x = [0,-30,0,30]; step(x,dt=0.05,recursion=4).
# This may be occasioned by the Kalman analysis update,
# especially if the system is only partially observed.
# Is this effectively a CFL condition? Could be addressed by:
#  - post-processing,
#  - modifying the step() function, e.g.:
#    - crop amplitude
#    - or lowering dt
#    - using an implicit time stepping scheme instead of rk4

import numpy as np
from scipy.linalg import circulant
from tools.misc import rk4, integrate_TLM, is1d

Force           = 8.0
prevent_blow_up = False

def dxdt(x):
  a = x.ndim-1
  s = lambda x,n: np.roll(x,-n,axis=a)
  return np.multiply(s(x,1)-s(x,-2), s(x,-1)) - x + Force

def step(x0, t, dt):

  #if prevent_blow_up:
    #clip      = abs(x0)>30
    #x0[clip] *= 0.1

  return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)


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
  # method='analytic' is a substantial upgrade for Lor95 
  return integrate_TLM(TLM(x),dt,method='analytic')


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


