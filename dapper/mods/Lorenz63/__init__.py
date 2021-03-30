"""The classic exhibitor of chaos, consisting of 3 coupled ODEs.

The ODEs are derived by modelling, with many simplifications,
the fluid convection between horizontal plates with different temperatures.

Its phase-plot (with typical param settings) looks like a butterfly.

See demo.py for more info.
"""


import numpy as np

import dapper.mods as modelling

from .extras import LPs, d2x_dtdx, dstep_dx

__pdoc__ = {"demo": False}

# Constants
sig = 10.0
rho = 28.0
beta = 8.0/3


@modelling.ens_compatible
def dxdt(x):
    """Evolution equation (coupled ODEs) specifying the dynamics."""
    d     = np.zeros_like(x)
    x, y, z = x
    d[0]  = sig*(y - x)
    d[1]  = rho*x - y - x*z
    d[2]  = x*y - beta*z
    return d


step = modelling.with_rk4(dxdt, autonom=True)

Tplot = 4.0

x0 = np.array([1.509, -1.531, 25.46])
