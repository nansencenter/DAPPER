import numpy as np
from scipy.linalg import circulant
from misc import rk4, is1d, atmost_2d

from mods.Lorenz95.fundamentals import lr


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

  # Split X,Y
  X = x[:,:nX]
  Y = x[:,nX:]
  assert Y.shape[1] is J*X.shape[1]

  d = np.zeros_like(x)
  # X vars
  d[:,:nX] = np.multiply(lr(X,1,a)-lr(X,-2,a),lr(X,-1,a)) - X + F
  for i in range(nX):
    d[:,i] += -h*c/b * np.sum(Y[:,iiY[i]],1)
  # Y vars
  d[:,nX:] = -c*b*np.multiply(lr(Y,2,a)-lr(Y,-1,a),lr(Y,1,a)) - c*Y \
      + h*c/b * X[:,iiX]
  return d


def step(x0, t, dt):
  return rk4(lambda t,x: dxdt(x),x0,np.nan,dt)


from mods.Lorenz95.fundamentals import typical_init_params
mu0,_P0 = typical_init_params(nX)
mu0 = np.hstack([mu0, np.zeros(nX*J)])
P0 = np.eye(m)
P0[:nX,:nX] = _P0
