"""A 1D emulator of chaotic atmospheric behaviour.

`bib.lorenz1996predictability`

For a short introduction, see

- `demo` and
- "Dynamical systems, chaos, Lorenz.ipynb" from the DA-tutorials

Note: the implementation is `ndim`-agnostic.
"""

import numpy as np

from dapper.mods.integration import rk4

from .extras import LPs, d2x_dtdx, dstep_dx

__pdoc__ = {"demo": False}

Force = 8.0
Tplot = 10


def x0(M):
    return np.eye(M)[0]


def shift(x, n):
    return np.roll(x, -n, axis=-1)


def dxdt_autonomous(x):
    return (shift(x, 1)-shift(x, -2))*shift(x, -1) - x


def dxdt(x):
    return dxdt_autonomous(x) + Force


def step(x0, t, dt):
    return rk4(lambda t, x: dxdt(x), x0, np.nan, dt)
