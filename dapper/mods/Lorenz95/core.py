# "Lorenz-95" (or 96) model. For a deeper introduction, see
# "DAPPER/tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"
#
# Note: implementation is ndim-agnostic.

import numpy as np
from dapper.tools.math import is1d, rk4, integrate_TLM

Force = 8.0

Tplot = 10

x0 = lambda M: 2.3*np.ones(M)

def shift(x,n):
  return np.roll(x,-n,axis=-1)

def dxdt_autonomous(x):
  return (shift(x,1)-shift(x,-2))*shift(x,-1) - x

def dxdt(x):
  return dxdt_autonomous(x) + Force

def step(x0, t, dt):
  return rk4(lambda t,x: dxdt(x), x0, np.nan, dt)


################################################
# OPTIONAL (not required by EnKF or PartFilt):
################################################
def d2x_dtdx(x):
  assert is1d(x)
  M = len(x)
  F = np.zeros((M,M))
  md = lambda i: np.mod(i,M) # modulo

  for i in range(M):
    F[i,i]       = -1.0
    F[i,   i-2 ] = -x[i-1]
    F[i,md(i+1)] = +x[i-1]
    F[i,   i-1 ] = x[md(i+1)]-x[i-2]

  return F

# dstep_dx = FD_Jac(step)
def dstep_dx(x,t,dt):
  # For L95, method='analytic' >> 'approx'
  return integrate_TLM(d2x_dtdx(x),dt,method='analytic')


################################################
# Add some non-default liveplotters
################################################
import dapper.tools.liveplotting as LP
def LPs(jj=None): return [
    (11, 1, LP.spatial1d(jj)  ),
    (12, 1, LP.correlations   ),
    (15, 0, LP.spectral_errors),
    (13, 0, LP.phase3d(jj)    ),
    (14, 0, LP.sliding_marginals(jj)) ,
    ]


