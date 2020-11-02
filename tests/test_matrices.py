
from dapper import *
import numpy as np

def test_1():
    """This causes the bug reported by Julien Brajard"""
    d = np.array([.3,1,2,1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full) ,d)

def test_2():
    """Special case: np.all(d==d[0])"""
    d = np.array([1,1,1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full) ,d)

def test_3():
    """Make sure truncation and sorting works."""
    d = np.array([1,0,1])
    C = CovMat(d)
    assert np.allclose(np.diag(C.full) ,d)
