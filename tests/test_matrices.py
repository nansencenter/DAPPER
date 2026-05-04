"""Tests for dapper.tools.matrices: CovMat and helpers."""

import numpy as np
import pytest

from dapper.mods import CovMat
from dapper.tools.matrices import randcov
from dapper.tools.seeding import set_seed

set_seed(3)


# ---------------------------------------------------------------------------
# Original regression tests (issue 11)
# ---------------------------------------------------------------------------


def test_1():
    d = np.array([0.3, 1, 2, 1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full), d)


def test_2():
    """Special case: np.all(d==d[0])"""
    d = np.array([1, 1, 1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full), d)


def test_3():
    """Make sure truncation and sorting works."""
    d = np.array([1, 0, 1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full), d)


# ---------------------------------------------------------------------------
# Round-trips: full matrix recovered from each input kind
# ---------------------------------------------------------------------------

M = 5


@pytest.fixture
def full_cov():
    return randcov(M)


class TestRoundTrips:
    def test_full(self, full_cov):
        C = CovMat(full_cov, "full")
        assert np.allclose(C.full, full_cov)

    def test_diag(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        C = CovMat(d, "diag")
        assert np.allclose(np.diag(C.full), d)

    def test_Right(self, full_cov):
        R = np.linalg.cholesky(full_cov).T
        C = CovMat(R, "Right")
        assert np.allclose(C.full, full_cov, atol=1e-10)

    def test_Left(self, full_cov):
        L = np.linalg.cholesky(full_cov)
        C = CovMat(L, "Left")
        assert np.allclose(C.full, full_cov, atol=1e-10)

    def test_A(self, full_cov):
        """'A' kind: pre-centred ensemble anomalies."""
        L = np.linalg.cholesky(full_cov)
        N = 50
        rng = np.random.default_rng(7)
        A = rng.standard_normal((N, M)) @ L.T
        A -= A.mean(0)
        C = CovMat(A, "A")
        # Sample cov from A should be close to full_cov
        expected = (A.T @ A) / (N - 1)
        assert np.allclose(C.full, expected, atol=1e-10)

    def test_E(self, full_cov):
        """'E' kind: full ensemble (not pre-centred)."""
        L = np.linalg.cholesky(full_cov)
        N = 50
        rng = np.random.default_rng(7)
        E = rng.standard_normal((N, M)) @ L.T
        C = CovMat(E, "E")
        A = E - E.mean(0)
        expected = (A.T @ A) / (N - 1)
        assert np.allclose(C.full, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Lazy / cached properties
# ---------------------------------------------------------------------------


class TestLazyProperties:
    def test_sym_sqrt(self, full_cov):
        C = CovMat(full_cov, "full")
        S = C.sym_sqrt
        assert np.allclose(S @ S, full_cov, atol=1e-10)

    def test_sym_sqrt_inv(self, full_cov):
        C = CovMat(full_cov, "full")
        Si = C.sym_sqrt_inv
        assert np.allclose(Si @ Si, C.pinv, atol=1e-10)

    def test_pinv(self, full_cov):
        C = CovMat(full_cov, "full")
        P = C.pinv
        assert np.allclose(P @ full_cov, np.eye(M), atol=1e-10)

    def test_sym_sqrt_cached(self, full_cov):
        C = CovMat(full_cov, "full")
        assert C.sym_sqrt is C.sym_sqrt  # same object returned


# ---------------------------------------------------------------------------
# Rank, eigenvalues, eigenvectors
# ---------------------------------------------------------------------------


class TestEVD:
    def test_full_rank(self, full_cov):
        C = CovMat(full_cov, "full")
        assert C.rk == M

    def test_rank_deficient(self):
        d = np.array([2.0, 1.0, 0.0, 0.0])
        C = CovMat(d, "diag")
        assert C.rk == 2

    def test_ews_positive(self, full_cov):
        C = CovMat(full_cov, "full")
        assert np.all(C.ews > 0)

    def test_ews_sorted_descending(self, full_cov):
        C = CovMat(full_cov, "full")
        assert np.all(np.diff(C.ews) <= 0)

    def test_V_orthonormal(self, full_cov):
        C = CovMat(full_cov, "full")
        VTV = C.V.T @ C.V
        assert np.allclose(VTV, np.eye(M), atol=1e-10)

    def test_reconstruct_from_evd(self, full_cov):
        C = CovMat(full_cov, "full")
        reconstructed = (C.V * C.ews) @ C.V.T
        assert np.allclose(reconstructed, full_cov, atol=1e-10)


# ---------------------------------------------------------------------------
# inv raises on rank-deficient input
# ---------------------------------------------------------------------------


class TestInv:
    def test_inv_full_rank(self, full_cov):
        C = CovMat(full_cov, "full")
        Inv = C.inv
        assert np.allclose(Inv @ full_cov, np.eye(M), atol=1e-10)

    def test_inv_raises_rank_deficient(self):
        d = np.array([2.0, 1.0, 0.0])
        C = CovMat(d, "diag")
        with pytest.raises(RuntimeError):
            _ = C.inv
