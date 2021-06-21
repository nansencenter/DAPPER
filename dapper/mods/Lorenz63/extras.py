"""Extra functionality (not necessary for the EnKF or the particle filter)."""

import numpy as np

import dapper.mods.Lorenz63 as core
import dapper.tools.liveplotting as LP
from dapper.mods.integration import integrate_TLM


def d2x_dtdx(x):
    """Tangent linear model (TLM). I.e. the Jacobian of dxdt(x)."""
    x, y, z = x
    sig, rho, beta = core.sig, core.rho, core.beta
    A = np.array(
        [[-sig, sig, 0],
         [rho-z, -1, -x],
         [y, x, -beta]])
    return A


def dstep_dx(x, t, dt):
    """Compute resolvent (propagator) of the TLM. I.e. the Jacobian of `step(x)`."""
    return integrate_TLM(d2x_dtdx(x), dt, method='approx')


# Add some non-default liveplotters
params = dict(labels='xyz', Tplot=1)


def LPs(jj=None, params=params): return [
    (1, LP.correlations),
    (1, LP.sliding_marginals(jj, zoomy=0.8, **params)),
    (1, LP.phase_particles(is_3d=True, obs_inds=jj, **params)),
]
