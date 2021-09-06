"""A chaotic system of size 3, like Lorenz-63, but with +complex geometry.

Refs: `bib.lorenz1984irregularity`, `bib.lorenz2005look`
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.integration import integrate_TLM

# Constants
a = 0.25
b = 4
F = 8.0
G = 1.23
# G = 1.0


@modelling.ens_compatible
def dxdt(x):
    x, y, z = x
    dx = - y**2 - z**2 - a*x + a*F
    dy = x*y - b*x*z - y + G
    dz = b*x*y + x*z - z
    return np.array([dx, dy, dz])


step = modelling.with_rk4(dxdt, autonom=True)

x0 = np.array([1.65, 0.49, 1.21])


def d2x_dtdx(x):
    x, y, z = x
    Mat = np.array(
        [[-a   , -2*y, -2*z],
         [y-b*z, x-1 , -b*x],
         [b*y+z, b*x , x-1]])
    return Mat


def dstep_dx(x, t, dt):
    return integrate_TLM(d2x_dtdx(x), dt, method='approx')
