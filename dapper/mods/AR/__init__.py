"""Linear, multivariate auto-regressive model, AR(1), with coupled components.

$$ x_{k+1, i} = \\phi \\, x_{k, i} + c \\, x_{k, i+1} + q_{k, i} \\,, $$

with $x_{k, D} := 0$. This is the discrete-time version of the damped chain

$$ \\frac{d x_i}{d t} = x_{i+1} - \\beta x_i + \\frac{d q_i}{d t} \\,, $$

i.e. a noisy, damped generalisation of $d^D x / dt^D = 0$,
as in "T5 - Multivariate Kalman filter" of the
[DA-tutorials](https://github.com/nansencenter/DA-tutorials).

The transition matrix `Mmat(D, phi, c)` is upper bidiagonal.
For `c=0` the components decouple into independent (scalar) AR(1) processes
(and `phi=1` then recovers `mods.Id`).
For `c≠0` it is a single Jordan block (all eigenvalues equal `phi`),
i.e. non-normal, yielding transient growth despite being stable (if `|phi|<1`).

Since everything is linear-Gaussian, the exact (KF) solution is available,
and (for `|phi|<1`) so are the stationary distribution
and the steady-state KF covariances:

- `stationary_cov`: solves the discrete Lyapunov equation $C = M C M^T + Q$.
- `steady_KF_covs`: solves the discrete algebraic Riccati equation.

These make for analytic tests of DA methods (see `tests/test_AR.py`).
"""

import numpy as np
import scipy.linalg as sla


def Mmat(D, phi=0.9, c=1.0):
    """Transition matrix: `phi` on the diagonal, `c` on the superdiagonal."""
    return phi * np.eye(D) + c * np.eye(D, k=1)


def stationary_cov(M, Q):
    """Stationary covariance, `C = M C M^T + Q`. Requires spectral radius of `M` < 1."""
    return sla.solve_discrete_lyapunov(M, Q)


def steady_KF_covs(M, H, Q, R):
    """Steady-state KF forecast and analysis covariances, `(Pf, Pa)`.

    Assumes that `(M, H)` is detectable and `(M, Q^{1/2})` stabilisable,
    and that obs are taken every time step.
    """
    Pf = sla.solve_discrete_are(M.T, H.T, Q, R)
    K = sla.solve(H @ Pf @ H.T + R, H @ Pf).T
    Pa = (np.eye(len(M)) - K @ H) @ Pf
    return Pf, Pa


def linear_step(M):
    """Make `step(x, t, dt)` for a linear (discrete-time) model with matrix `M`.

    Works for single states `(D,)` and ensembles `(N, D)`.
    The model is discrete; it only supports `dt=1`.
    """

    def step(x, t, dt):
        assert dt == 1, "Discrete-time model: only supports dt=1."
        return x @ M.T

    return step
