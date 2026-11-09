"""Test DA methods against analytic (linear-Gaussian) results, using `mods.AR`.

Unlike `tests/test_iEnKS.py` (which only tests equivalences between methods),
these test absolute values, component-wise, including unobserved components
(which get updated only via cross-covariances).
"""

import numpy as np
import pytest

import dapper.da_methods as da
from dapper.mods.AR import stationary_cov, steady_KF_covs
from dapper.mods.AR.tutorials import HMM as _HMM
from dapper.tools.seeding import set_seed

M = _HMM.Dyn.linear(None, 0, 1)
Q = _HMM.Dyn.noise.C.full
H = _HMM.Obs(0).linear(None)
R = _HMM.Obs(0).noise.C.full

Pf, Pa = steady_KF_covs(M, H, Q, R)
C = stationary_cov(M, Q)


def _run(xp, Ko, seed=3000):
    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = Ko
    set_seed(seed)
    xx, yy = HMM.simulate()
    xp.assimilate(HMM, xx, yy)
    return xp.stats


def test_analytic_refs():
    """Sanity-check the reference solutions themselves."""
    assert np.allclose(C, M @ C @ M.T + Q)
    inv = np.linalg.inv
    assert np.allclose(Pa, inv(inv(Pf) + H.T @ inv(R) @ H))
    assert np.allclose(Pf, M @ Pa @ M.T + Q)
    # Obs reduce variance, but don't beat the noise floor
    assert np.all(np.diag(Q) < np.diag(Pa))
    assert np.all(np.diag(Pa) < np.diag(Pf))
    assert np.all(np.diag(Pf) < np.diag(C))


def test_ExtKF_riccati():
    """ExtKF covariance converges to the DARE solution (deterministic)."""
    stats = _run(da.ExtKF(), Ko=250)
    # Convergence is to ~1e-13 by ko=200
    assert np.allclose(stats.spread.f[-1] ** 2, np.diag(Pf), rtol=1e-8, atol=0)
    assert np.allclose(stats.spread.a[-1] ** 2, np.diag(Pa), rtol=1e-8, atol=0)


# Monte Carlo tests. The errors are strongly auto-correlated (esp. for x_0),
# so convergence is slow. With Ko=4000, the max (over components) relative
# deviation of err.ms from its expectation, over 20 seeds, was at worst
# 20% (ExtKF), 29% (Climatology), 18% (OptInterp). Tolerances are set with margin;
# they still catch e.g. a missing/doubled Q or R, or wrong cross-covariances.
Ko_MC = 4000
burn = 100

# OI: prior = climatology, so (with zero-mean, independent-of-past stationary truth)
# the analysis error cov is (I - K H) C, with K the climatological gain.
K_C = np.linalg.solve(H @ C @ H.T + R, H @ C).T
P_OI = (np.eye(len(C)) - K_C @ H) @ C


@pytest.mark.parametrize(
    ("xp", "expected", "rtol"),
    [
        (da.ExtKF(), np.diag(Pa), 0.3),
        (da.Climatology(), np.diag(C), 0.4),
        (da.OptInterp(), np.diag(P_OI), 0.3),
    ],
    ids=["ExtKF", "Climatology", "OptInterp"],
)
def test_err_ms(xp, expected, rtol):
    """Time-averaged (analysis) squared errors match analytic expectations."""
    stats = _run(xp, Ko=Ko_MC)
    err_ms = np.mean(stats.err.a[burn:] ** 2, axis=0)
    assert np.allclose(err_ms, expected, rtol=rtol, atol=0)
