####################################
# Lorenz95 two-scale/layer version
####################################
# See Wilks 2005 "Effects of stochastic parametrizations in the Lorenz '96 system"
# X:  large amp, low frequency vars: convective events
# Y:  small amp, high frequency vars: large-scale synoptic events
#
# Typically, the DA system will only use the truncated system
# (containing only the X variables),
# where the Y's are parameterized as model noise,
# while the truth is simulated by the full system.
#
# Stochastic parmateterization (Todo):
# Wilks: benefit of including stochastic noise negligible
# unless its temporal auto-corr is taken into account (as AR(1))
# (but spatial auto-corr can be neglected).
# But AR(1) noise is technically difficult because DAPPER
# is built around the Markov assumption. Possible work-around:
#  - Don't use standard dxdt + rk4
#  - Use persistent variables


import numpy as np
from numpy import arange
from dapper.tools.math import rk4, is1d
from matplotlib import pyplot as plt
import dapper.tools.liveplotting as LP

# Shift elements
s = lambda x,n: np.roll(x,-n,axis=-1)


class model_instance():
  """
  Use OOP to facilitate having multiple parameter settings simultaneously.
  Default parameters from Wilks'2005.
  """
  def __init__(self,nU=8,J=32,F=20,h=1,b=10,c=10):
    
    # System size
    self.nU = nU       # num of X
    self.J  = J        # num of Y per X  
    self.M  = (J+1)*nU # => Total state length

    # Other parameters
    self.F  = F  # forcing
    self.h  = h  # coupling constant
    self.b  = b  # Spatial scale ratio
    self.c  = c  # time scale ratio

    # Indices for coupling
    self.iiX = (arange(J*nU)/J).astype(int)
    self.iiY = arange(J*nU).reshape((nU,J))


  def dxdt_trunc(self,x):
    """
    Truncated dxdt: slow variables (X) only.
    Same as "uncoupled" Lorenz-95.
    """
    assert x.shape[-1] == self.nU
    return -(s(x,-2)-s(x,1))*s(x,-1) - x + self.F


  def dxdt(self,x):
    """Full (coupled) dxdt."""
    # Split into X,Y
    nU,J,h,b,c = self.nU,self.J,self.h,self.b,self.c
    X  = x[...,:nU]
    Y  = x[...,nU:]
    assert Y.shape[-1] == J*X.shape[-1]
    d  = np.zeros_like(x)

    # dX/dt
    d[...,:nU] = self.dxdt_trunc(X)
    # Couple Y-->X
    for i in range(nU):
      d[...,i] += -h*c/b * np.sum(Y[...,self.iiY[i]],-1)

    # dY/dt
    d[...,nU:] = -c*b*(s(Y,2)-s(Y,-1))*s(Y,1) - c*Y
    # Couple X-->Y
    d[...,nU:] += h*c/b * X[...,self.iiX]

    return d


  def dxdt_parameterized(self,t,x):
    """
    Truncated dxdt with parameterization of fast variables (Y)
    """
    d  = self.dxdt_trunc(x)
    d -= self.prmzt(t,x) # must (of course) be set first
    return d


  def dfdx(self):
    """
    Jacobian of x + dt*dxdt.
    """
    nU,J,h,b,c = self.nU,self.J,self.h,self.b,self.c
    assert is1d(x)
    F = np.zeros((ndim(),ndim()))

    # X
    md = lambda i: np.mod(i,nU)
    for i in range(nU):
      # wrt. X
      F[i,i]         = - dt + 1
      F[i,md(i-2)]   = - dt * x[md(i-1)]
      F[i,md(i+1)]   = + dt * x[md(i-1)]
      F[i,md(i-1)]   =   dt *(x[md(i+1)]-x[md(i-2)])
      # wrt. Y
      F[i,nU+self.iiY[i]] = dt * -h*c/b
    # Y
    md = lambda i: nU + np.mod(i-nU,nU*J)
    for i in range(nU,(J+1)*nU):
      # wrt. Y
      F[i,i]         = -dt*c + 1
      F[i,md(i-1)]   = +dt*c*b * x[md(i+1)]
      F[i,md(i+1)]   = -dt*c*b * (x[md(i+2)]-x[md(i-1)])
      F[i,md(i+2)]   = -dt*c*b * x[md(i+1)]
      # wrt. X
      F[i,self.iiX[i-nU]] = dt * h*c/b
    return F

  def LPs(self,jj):
    return [ ( 11, 1, LP.spatial1d(jj,dims=arange(self.nU)) ) ]

