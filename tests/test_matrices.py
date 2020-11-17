"""Test bug (issue 11) reported by Julien Brajard."""

import numpy as np
import dapper as dpr


def test_1():
    d = np.array([.3, 1, 2, 1])
    C = dpr.CovMat(d)
    assert np.allclose(np.diag(C.full), d)


def test_2():
    """Special case: np.all(d==d[0])"""
    d = np.array([1, 1, 1])
    C = dpr.CovMat(d)
    assert np.allclose(np.diag(C.full), d)


def test_3():
    """Make sure truncation and sorting works."""
    d = np.array([1, 0, 1])
    C = dpr.CovMat(d)
    assert np.allclose(np.diag(C.full), d)
