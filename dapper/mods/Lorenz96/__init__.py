"""A 1D emulator of chaotic atmospheric behaviour.

`bib.lorenz1996predictability`

For a short introduction, see

- `demo` and
- "Dynamical systems, chaos, Lorenz.ipynb" from the DA-tutorials

Note: the implementation is `ndim`-agnostic.
"""

import numpy as np

import dapper.tools.liveplotting as LP
from dapper.mods.integration import integrate_TLM, rk4

__pdoc__ = {"demo": False}

Force = 8.0

Tplot = 10


def x0(M): return np.eye(M)[0]


def shift(x, n):
    return np.roll(x, -n, axis=-1)


def dxdt_autonomous(x): return (shift(x, 1)-shift(x, -2))*shift(x, -1) - x
def dxdt(x): return dxdt_autonomous(x) + Force


def step(x0, t, dt):
    return rk4(lambda t, x: dxdt(x), x0, np.nan, dt)


################################################
# OPTIONAL (not required by EnKF or PartFilt):
################################################
def d2x_dtdx(x):
    M = len(x)
    F = np.zeros((M, M))
    def md(i): return np.mod(i, M)  # modulo

    for i in range(M):
        F[i, i]       = -1.0
        F[i,   i-2]   = -x[i-1]
        F[i, md(i+1)] = +x[i-1]
        F[i,   i-1]   = x[md(i+1)]-x[i-2]

    return F

# dstep_dx = FD_Jac(step)


def dstep_dx(x, t, dt):
    # For L96, method='analytic' >> 'approx'
    return integrate_TLM(d2x_dtdx(x), dt, method='analytic')


################################################
# Add some non-default liveplotters
################################################
def LPs(jj=None): return [
    (11, 1, LP.spatial1d(jj)),
    (12, 1, LP.correlations),
    (15, 0, LP.spectral_errors),
    (13, 0, LP.phase_particles(True, jj)),
    (14, 0, LP.sliding_marginals(jj)),
]
