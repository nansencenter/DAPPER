"""A 1D emulator of chaotic atmospheric behaviour.

`bib.lorenz1996predictability`

For a short introduction, see

- `demo` and
- "Dynamical systems, chaos, Lorenz.ipynb" from the DA-tutorials

Note: the implementation is `len(x)`-agnostic.
"""

import numpy as np

from dapper.mods.integration import rk4

from .extras import LPs, d2x_dtdx, dstep_dx

Force = 8.0
Tplot = 10


def x0(M):
    x = np.zeros(M)
    x[0] = 1
    return x


def shift(x, n):
    return np.roll(x, -n, axis=-1)


def dxdt_autonomous(x):
    return (shift(x, 1)-shift(x, -2))*shift(x, -1) - x


def dxdt(x):
    return dxdt_autonomous(x) + Force


def step(x0, t, dt):
    return rk4(lambda x, t: dxdt(x), x0, np.nan, dt)
