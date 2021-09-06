"""The "Ikeda map" is a discrete-time dynamical system of size 2.

Source: [Wiki](https://en.wikipedia.org/wiki/Ikeda_map) and Colin Grudzien.

See `demo` for more info.
"""


import numpy as np
from numpy import cos, sin

import dapper.mods as modelling
import dapper.tools.liveplotting as LP

# Constant 0.6 <= u <= 1.
u = 0.9

x0 = np.zeros(2)
Tplot = 10.0


@modelling.ens_compatible
def step(x, _t, _dt):
    s, t, x1, y1 = aux(*x)
    return 1+u*x1, u*y1


def aux(x, y):
    """Comps used both by step and its jacobian."""
    s = 1 + x**2 + y**2
    t = 0.4 - 6 / s
    # x1= x*cos(t) + y*cos(t) # Colin's mod
    x1 = x*cos(t) - y*sin(t)
    y1 = x*sin(t) + y*cos(t)
    return s, t, x1, y1


def dstep_dx(x, _t, _dt):
    s, t, x1, y1 = aux(*x)
    x, y = x

    dt_x = 12/s**2 * x
    dt_y = 12/s**2 * y

    dx_x = -y1*dt_x + cos(t)
    dy_x = +x1*dt_x + sin(t)
    dx_y = -y1*dt_y - sin(t)
    dy_y = +x1*dt_y + cos(t)

    return u * np.array([[dx_x, dx_y], [dy_x, dy_y]])


# Liveplotting
params = dict(labels='xy')


def LPs(jj=None, params=params): return [
    (1, LP.sliding_marginals(obs_inds=jj, zoomy=0.8, **params)),
    (1, LP.phase_particles(
        is_3d=False, obs_inds=jj, zoom=0.8, Tplot=0, **params)),
]
