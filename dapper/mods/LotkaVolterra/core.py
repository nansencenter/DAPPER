# Parameters set so as to have chaotic dynamics.
# Refs: wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
#       Vano et al (2006): "Chaos in low-dimensional Lotka-Volterra models of competition".

from dapper import *

Nx = 4

# "growth" coefficients
r = array([1 , 0.72 , 1.53 , 1.27])

# "interaction" coefficients
A = array([
  [ 1    , 1.09 , 1.52 , 0    ] ,
  [ 0    , 1    , 0.44 , 1.36 ] ,
  [ 2.33 , 0    , 1    , 0.47 ] ,
  [ 1.21 , 0.51 , 0.35 , 1    ]
  ])

x0 = 0.25*ones(Nx)

def dxdt(x):
  return (r*x) * (1 - x@A.T)

step = with_rk4(dxdt,autonom=True)


def TLM(x):
  return diag(r - r*(A@x)) - (r*x)[:,None]*A 
def dfdx(x,t,dt):
  return integrate_TLM(TLM(x),dt,method='approx')


from dapper.mods.Lorenz63.core import LPs as L63_LPs
LP_setup = lambda jj: L63_LPs(jj, params=dict())

