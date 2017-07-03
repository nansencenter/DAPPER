####################################
# Lorenz95 two-scale/layer version
####################################
# See Wilks 2005 "effects..."
# X:  large amp, low frequency vars: convective events
# Y:  small amp, high frequency vars: large-scale synoptic events

# Typically, the DA system will only use the truncated system
# (containing only the X variables),
# where the Y's are parameterized as model noise,
# while the truth is simulated by the full system.
#
# See berry2014linear for EnKF application.
# Typically dt0bs = 0.01 and dt = dtObs/10 for truth.
# But for EnKF they use dt = dtObs coz
# "numerical stiffness disappears when fast processes are removed".
#
# Wilks2005 uses dt=1e-4 with RK4 for the full model,
# and dt=5e-3 with RK2 for the forecast/truncated model.
#
# Also see mitchell2014 and Hanna Arnold's thesis.

import numpy as np
from scipy.linalg import circulant
from tools.misc import rk4, is1d, atmost_2d

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
  # Split into X,Y
  X = x[:,:nX]
  Y = x[:,nX:]
  assert Y.shape[1] == J*X.shape[1]

  s = lambda x,n: np.roll(x,-n,axis=-1)

  d = np.zeros_like(x)
  # dX/dt -- same as "uncoupled" Lorenz-95
  d[:,:nX] = np.multiply(s(X,1)-s(X,-2),s(X,-1)) - X + F
  # Add in coupling from Y vars
  for i in range(nX):
    d[:,i] += -h*c/b * np.sum(Y[:,iiY[i]],1)
  # dY/dt
  d[:,nX:] = -c*b*np.multiply(s(Y,2)-s(Y,-1),s(Y,1)) - c*Y \
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


@atmost_2d
def dxdt_trunc(x):
  "truncated dxdt: slow variables (X) only"
  assert x.shape[1] == nX
  return np.multiply(s(x,1)-s(x,-2),s(x,-1)) - x + F

def dxdt_det(x):
  """
  Truncated dxdt: slow variables (X) only.
  Deterministic parameterization of fast variables (Y)
  """
  d = dxdt_trunc(x) #@atmost_2d included here
  # Parameterization tuned (by Wilks) for following values.
  assert np.all([nX==8,J==32,F==20,c==10,b==10,h==1])
  d -= 0.262 + 1.45*x - 0.0121*x**2 - 0.00713*x**3 + 0.000296*x**4
  return d

def dxdt_bad(x):
  """
  Truncated dxdt: slow variables (X) only.
  Y parameterized by constant forcing. Should be worse that dxdt_det()
  """
  d = dxdt_trunc(x) #@atmost_2d included here
  assert np.all([nX==8,J==32,F==20,c==10,b==10,h==1])
  d -= 5.5
  return d


#@atmost_2d
#def dxdt_ar1(x):
  # Wilks: benefit of including stochastic noise negligible
  # unless its temporal auto-corr is taken into account (as AR(1))
  # (but spatial auto-corr can be neglected).
  #
  # But, using "persistent variables" to get autocorrelation
  # wont work, coz RK4 calls dxdt multiple (4) times,
  # thus generating new (but correlated) noise instances for
  # the same time step.
  # Moreover, I the noise should scale with sqrt(dt),
  # which won't happen if you put it into dxdt.
  # => stochastic parameterizations is the remit of add_noise().

  #phi_1 = 0.984
  #phi_c = (1-phi**2)**0.5
  #sig   = 1.99

  #d  = dxdt_d(x)
  #w  = phi*w + sig*phi_c*randn(x.shape)
  #d += w

  #return d
