"""A 1D emulator of chaotic atmospheric behaviour.

From "Predictability -- a problem partly solved" by E. N. Lorenz (1996).
Proc. Seminar on Predictability, Vol. 1, ECMWF, Reading, Berkshire, UK, 1-18.

For a short introduction, see
- demo.py and
- "Dynamical systems, chaos, Lorenz.ipynb" from the DA-tutorials

Note: the implementation is ndim-agnostic.
"""

import numpy as np

import dapper.tools.liveplotting as LP
from dapper.tools.math import integrate_TLM, is1d, rk4, RK4_adj, RK4_linear

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
    # assert is1d(x)
    M = x.shape[-1]
    N = len(x)
    F = np.zeros((N, M, M))
    def md(i): return np.mod(i, M)  # modulo

    for i in range(M):
        F[..., i, i]       = -1.0
        F[..., i,   i-2]   = -x[..., i-1]
        F[..., i, md(i+1)] = +x[..., i-1]
        # print(md(i+2))
        F[..., i,   i-1]   = x[..., md(i+1)]-x[..., i-2]

    return F

# dstep_dx = FD_Jac(step)
def d2x_dtdx_adjoint(X):
    
    x, dx = X[0], X[1]
    M = x.shape[-1]
    dx_new = np.zeros_like(x)
    def md(i): return np.mod(i, M)  # modulo

    for i in range(M):
        dx_new[...,i] = dx[...,i-1]*x[...,i-2] - dx[...,i] + dx[...,md(i+1)]*(x[...,md(i+2)] - x[...,i-1]) - dx[...,md(i+2)]*x[...,md(i+1)]
    return dx_new

def adjoint_step(x0, t, dt):
    return RK4_adj(lambda t, x: dxdt(x), lambda t, x: d2x_dtdx_adjoint(x), x0, np.nan, dt)

def linear_step(x, M0, t, dt):
    return RK4_linear(lambda t, x: dxdt(x), lambda t, x, M: d2x_dtdx(x)@M, x, M0, t, dt)
    # return integrate_TLM(d2x_dtdx(x), dt, method='rk4')

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
