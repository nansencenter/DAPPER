"""Extra functionality (not necessary for the EnKF or the particle filter)."""

import numpy as np

import dapper.tools.liveplotting as LP
from dapper.mods.integration import integrate_TLM


def d2x_dtdx(x):
    """Tangent linear model (TLM). I.e. the Jacobian of dxdt(x)."""
    M = len(x)
    F = np.zeros((M, M))
    def md(i): return np.mod(i, M)  # modulo

    for i in range(M):
        F[i   , i]    = -1.0
        F[i   , i-2]  = -x[i-1]
        F[i, md(i+1)] = +x[i-1]
        F[i   , i-1]  = x[md(i+1)]-x[i-2]

    return F


# dstep_dx = FD_Jac(step)
def dstep_dx(x, t, dt):
    """Compute resolvent (propagator) of the TLM. I.e. the Jacobian of `step(x)`."""
    # For L96, method='analytic' >> 'approx'
    return integrate_TLM(d2x_dtdx(x), dt, method='analytic')


# Add some non-default liveplotters
def LPs(jj=None): return [
    (1, LP.spatial1d(jj)),
    (1, LP.correlations),
    (0, LP.spectral_errors),
    (0, LP.phase_particles(True, jj)),
    (0, LP.sliding_marginals(jj)),
]
