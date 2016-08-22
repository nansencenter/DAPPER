import numpy as np
from scipy.linalg import circulant
from misc import rk4, is1d, atmost_2d

from mods.Lorenz95.core import lr


####################################
# Lorenz95 two-scale/layer version
####################################
# See Wilks 2005 "effects..." and Hanna Arnold's thesis.
# X:  large amp, low frequency vars: convective events
# Y:  small amp, high frequency vars: large-scale synoptic events

# Typically, the DA system will only use the truncated system
# (containing only the X variables),
# where the Y's are parameterized as model noise,
# while the truth is simulated by the full system.
#
# See berry2014linear for EnKF application.
# Typically dt0bs = 0.01 and dt = dtObs/10.
# But for EnKF (full system) they use dt = dtObs coz
# "numerical stiffness disappears when fast processes are removed".
#
# Also see mitchell2014

nX= 8  # of X
J = 32 # of Y per X 
m = (J+1)*nX # total state length
h = 1  # coupling constant
F = 20 # forcing
b = 10 # Spatial scale ratio
c = 10 # time scale ratio
#c = 4 more difficult to parameterize (less scale separation)

iiX = (np.arange(J*nX)/J).astype(int)
iiY = np.arange(J*nX).reshape((nX,J))
@atmost_2d
def dxdt(x):
  a = 1 # axis

  # Split into X,Y
  X = x[:,:nX]
  Y = x[:,nX:]
  assert Y.shape[1] == J*X.shape[1]

  d = np.zeros_like(x)
  # dX/dt -- same as "uncoupled" Lorenz-95
  d[:,:nX] = np.multiply(lr(X,1,a)-lr(X,-2,a),lr(X,-1,a)) - X + F
  # Add in coupling from Y vars
  for i in range(nX):
    d[:,i] += -h*c/b * np.sum(Y[:,iiY[i]],1)
  # dY/dt
  d[:,nX:] = -c*b*np.multiply(lr(Y,2,a)-lr(Y,-1,a),lr(Y,1,a)) - c*Y \
      + h*c/b * X[:,iiX]
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


def step(x0, t, dt):
  return rk4(lambda t,x: dxdt(x),x0,np.nan,dt)


from mods.Lorenz95.core import typical_init_params
mu0,_P0 = typical_init_params(nX)
mu0 = np.hstack([mu0, np.zeros(nX*J)])
P0 = np.eye(m)
P0[:nX,:nX] = _P0
