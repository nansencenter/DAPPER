"""Tests for dapper.tools.randvars."""

import numpy as np
import pytest

from dapper.tools.randvars import (
    RV,
    GaussRV,
    LaplaceParallelRV,
    LaplaceRV,
    StudRV,
    UniParallelRV,
    UniRV,
)

M = 5
N = 2000  # large enough for statistical tests to be reliable
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Original test (kept, bare call removed)
# ---------------------------------------------------------------------------


def test_gauss_rv_zero():
    grv = GaussRV(mu=0, C=0, M=4)
    assert (grv.sample(5) == np.zeros((5, 4))).all()


# ---------------------------------------------------------------------------
# GaussRV: shape and basic statistics
# ---------------------------------------------------------------------------


class TestGaussRV:
    def test_sample_shape(self):
        grv = GaussRV(M=M)
        E = grv.sample(N)
        assert E.shape == (N, M)

    def test_sample_mean_near_zero(self):
        grv = GaussRV(mu=0, C=1, M=M)
        E = grv.sample(N)
        assert np.allclose(E.mean(0), 0, atol=0.1)

    def test_sample_mean_nonzero(self):
        mu = np.arange(1, M + 1, dtype=float)
        grv = GaussRV(mu=mu, C=1, M=M)
        E = grv.sample(N)
        assert np.allclose(E.mean(0), mu, atol=0.2)

    def test_sample_variance(self):
        grv = GaussRV(mu=0, C=4, M=M)
        E = grv.sample(N)
        assert np.allclose(E.var(0), 4, atol=0.5)

    def test_zero_noise_exact(self):
        grv = GaussRV(mu=3, C=0, M=M)
        E = grv.sample(10)
        assert np.all(E == 3)


# ---------------------------------------------------------------------------
# Other RV_with_mean_and_cov subclasses: shape checks
# ---------------------------------------------------------------------------


class TestRVSubclassShapes:
    @pytest.mark.parametrize(
        "cls", [LaplaceRV, LaplaceParallelRV, UniRV, UniParallelRV]
    )
    def test_sample_shape(self, cls):
        rv = cls(mu=0, C=1, M=M)
        assert rv.sample(N).shape == (N, M)

    def test_stud_rv_shape(self):
        rv = StudRV(dof=10, mu=0, C=1, M=M)
        assert rv.sample(N).shape == (N, M)


# ---------------------------------------------------------------------------
# RV base class: func and is0 dispatch
# ---------------------------------------------------------------------------


class TestRVDispatch:
    def test_is0(self):
        rv = RV(M=M, is0=True)
        E = rv.sample(N)
        assert np.all(E == 0)
        assert E.shape == (N, M)

    def test_func(self):
        rv = RV(M=M, func=lambda n: np.ones((n, M)) * 7)
        E = rv.sample(N)
        assert np.all(E == 7)
        assert E.shape == (N, M)

    def test_func_shape_checked(self):
        rv = RV(M=M, func=lambda n: np.ones((n, M + 1)))  # wrong M
        with pytest.raises((AssertionError, ValueError)):
            rv.sample(3)


# ---------------------------------------------------------------------------
# RV_with_mean_and_cov constructor edge cases
# ---------------------------------------------------------------------------


class TestRVWithMeanAndCovConstructor:
    def test_scalar_mu_broadcast(self):
        rv = GaussRV(mu=2, C=1, M=M)
        assert np.all(rv.mu == 2)
        assert len(rv.mu) == M

    def test_scalar_C(self):
        rv = GaussRV(mu=0, C=3, M=M)
        E = rv.sample(N)
        assert np.allclose(E.var(0), 3, atol=0.5)

    def test_M_inferred_from_mu(self):
        mu = np.ones(M)
        rv = GaussRV(mu=mu, C=1)
        assert rv.M == M

    def test_M_inferred_from_C_scalar_and_mu(self):
        mu = np.ones(M)
        rv = GaussRV(mu=mu, C=1)
        assert rv.M == M

    def test_mismatched_mu_M_raises(self):
        # Currently AssertionError; will become TypeError after Phase 2b
        with pytest.raises((AssertionError, TypeError)):
            GaussRV(mu=np.ones(3), C=1, M=5)
