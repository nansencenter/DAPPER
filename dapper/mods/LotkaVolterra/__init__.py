"""The generalized predator-prey model, with settings for chaotic dynamics.

Refs:
[Wiki](https://en.wikipedia.org/wiki/Competitive_Lotka-Volterra_equations),
`bib.vano2006chaos`.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.integration import integrate_TLM
from dapper.mods.Lorenz63 import LPs

Nx = 4

# "growth" coefficients
r = np.array([1, 0.72, 1.53, 1.27])

# "interaction" coefficients

A = np.array(
    [[1.  , 1.09, 1.52, 0.  ],
     [0.  , 1.  , 0.44, 1.36],
     [2.33, 0.  , 1.  , 0.47],
     [1.21, 0.51, 0.35, 1.  ]])

x0 = 0.25*np.ones(Nx)
Tplot = 100


def dxdt(x):
    return (r*x) * (1 - x@A.T)


step = modelling.with_rk4(dxdt, autonom=True)


def d2x_dtdx(x):
    return np.diag(r - r*(A@x)) - (r*x)[:, None]*A


def dstep_dx(x, t, dt):
    return integrate_TLM(d2x_dtdx(x), dt, method='approx')


def LP_setup(jj): return LPs(jj, params=dict())
