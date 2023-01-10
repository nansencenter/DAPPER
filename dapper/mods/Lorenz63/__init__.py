"""The classic exhibitor of chaos, consisting of 3 coupled ODEs.

The ODEs are derived by modelling, with many simplifications,
the fluid convection between horizontal plates with different temperatures.

Its phase-plot (with typical param settings) looks like a butterfly.

See demo.py for more info.
"""


import numpy as np

import dapper.mods as modelling

from .extras import LPs, d2x_dtdx, dstep_dx

# Constants
sig = 10.0
rho = 28.0
beta = 8.0/3

# Suggested values
x0 = np.array([1.509, -1.531, 25.46])
Tplot = 4.0


@modelling.ens_compatible
def dxdt(x):
    """Evolution equation (coupled ODEs) specifying the dynamics."""
    x, y, z = x
    dx = sig*(y - x)
    dy = rho*x - y - x*z
    dz = x*y - beta*z
    return np.array([dx, dy, dz])


step = modelling.with_rk4(dxdt, autonom=True)
