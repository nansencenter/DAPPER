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
from tools.math import rk4, is1d

# Parameters
nX= 8  # of X
J = 32 # of Y per X 
m = (J+1)*nX # total state length
h = 1  # coupling constant
F = 20 # forcing
b = 10 # Spatial scale ratio
c = 10 # time scale ratio
#c = 4 more difficult to parameterize (less scale separation)

check_parameters = True

# Shift elements
s = lambda x,n: np.roll(x,-n,axis=-1)

# Indices of X and Y variables in state
iiX = (arange(J*nX)/J).astype(int)
iiY = arange(J*nX).reshape((nX,J))


def dxdt_trunc(x):
  """
  Truncated dxdt: slow variables (X) only.
  Same as "uncoupled" Lorenz-95.
  """
  assert x.shape[-1] == nX
  return -(s(x,-2)-s(x,1))*s(x,-1) - x + F


def dxdt(x):
  """Full (coupled) dxdt."""
  # Split into X,Y
  X = x[...,:nX]
  Y = x[...,nX:]
  assert Y.shape[-1] == J*X.shape[-1]
  d = np.zeros_like(x)

  # dX/dt
  d[...,:nX] = dxdt_trunc(X)
  # Couple Y-->X
  for i in range(nX):
    d[...,i] += -h*c/b * np.sum(Y[...,iiY[i]],-1)

  # dY/dt
  d[...,nX:] = -c*b*(s(Y,2)-s(Y,-1))*s(Y,1) - c*Y
  # Couple X-->Y
  d[...,nX:] += h*c/b * X[...,iiX]

  return d



# Order of deterministic error parameterization.
# Note: In order to observe an improvement in DA performance when using
#       higher orders, the EnKF must be reasonably tuned with inflation.
#       There is very little improvement gained above order=1.
detp_order = 'UNSET' # set from outside

def dxdt_detp(t,x):
  """
  Truncated dxdt with
    polynomial (deterministic) parameterization of fast variables (Y)
  """
  d = dxdt_trunc(x)
  
  if check_parameters:
    assert np.all([nX==8,J==32,F==20,c==10,b==10,h==1]), \
        """
        The parameterizations have been tuned (by Wilks)
        for specific param values. These are not currently in use.
        """

  if hasattr(detp_order, "__call__"):
    # Custom parameterization function
    d -= detp_order(t,x)
  elif detp_order==4:
    # From Wilks
    d -= 0.262 + 1.45*x - 0.0121*x**2 - 0.00713*x**3 + 0.000296*x**4
  elif detp_order==3:
    # From Arnold
    d -= 0.341 + 1.30*x - 0.0136*x**2 - 0.00235*x**3
  elif detp_order==1:
    # From me -- see AdInf/illust_parameterizations.py
    d -= 0.74 + 0.82*x
  elif detp_order==0:
    # From me -- see AdInf/illust_parameterizations.py
    d -= 3.82
  elif detp_order==-1:
    # Leave as dxdt_trunc
    pass
  else:
    raise NotImplementedError
  return d


def dfdx(x,t,dt):
  """
  Jacobian of x + dt*dxdt.
  """
  assert is1d(x)
  F  = np.zeros((m,m))
  # X
  md = lambda i: np.mod(i,nX)
  for i in range(nX):
    # wrt. X
    F[i,i]         = - dt + 1
    F[i,md(i-2)]   = - dt * x[md(i-1)]
    F[i,md(i+1)]   = + dt * x[md(i-1)]
    F[i,md(i-1)]   =   dt *(x[md(i+1)]-x[md(i-2)])
    # wrt. Y
    F[i,nX+iiY[i]] = dt * -h*c/b
  # Y
  md = lambda i: nX + np.mod(i-nX,nX*J)
  for i in range(nX,(J+1)*nX):
    # wrt. Y
    F[i,i]         = -dt*c + 1
    F[i,md(i-1)]   = +dt*c*b * x[md(i+1)]
    F[i,md(i+1)]   = -dt*c*b * (x[md(i+2)]-x[md(i-1)])
    F[i,md(i+2)]   = -dt*c*b * x[md(i+1)]
    # wrt. X
    F[i,iiX[i-nX]] = dt * h*c/b
  return F


from matplotlib import pyplot as plt
def plot_state(x):
  circX = np.mod(arange(nX+1)  ,nX)
  circY = np.mod(arange(nX*J+1),nX*J) + nX
  lhX   = plt.plot(arange(nX+1)    ,x[circX],'b',lw=3)[0]
  lhY   = plt.plot(arange(nX*J+1)/J,x[circY],'g',lw=2)[0]
  ax    = plt.gca()
  ax.set_xticks(arange(nX+1))
  ax.set_xticklabels([(str(i) + '/\n' + str(i*J)) for i in circX])
  ax.set_ylim(-5,15)
  def setter(x):
    lhX.set_ydata(x[circX])
    lhY.set_ydata(x[circY])
  return setter



