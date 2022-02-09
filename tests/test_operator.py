"""Tests of the Operator class"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.LA import Fmat


def test_operator_defaults():
    M = 3
    op = modelling.Operator(**{"M": M})
    assert (op.linear() == np.identity(M)).all()
    assert op.model(3) == 3
    assert op.model(3, 2, 1) == 3
    assert (op.noise.sample(M) == np.zeros(M)).all()


def test_operator_dyn():
    """Simple test using 1D linear advection."""
    Nx = 6

    tseq = modelling.Chronology(dt=1, dko=5, T=10)
    Fm = Fmat(Nx, c=-1, dx=1, dt=tseq.dt)

    def step(x, t, dt):
        assert dt == tseq.dt
        return x @ Fm.T

    Dyn = {
        "M": Nx,
        "model": step,
        "linear": lambda x, t, dt: Fm,
        "noise": 0,
    }

    Dyn_op = modelling.Operator(**Dyn)

    # Square wave
    x = np.array([1, 1, 1, 0, 0, 0])

    x1 = Dyn_op(x, tseq.T - 1, tseq.dt)
    assert (x1 == np.array([0, 1, 1, 1, 0, 0])).all()

    x2 = Dyn_op(x1, tseq.T - 3, 1)
    assert (x2 == np.array([0, 0, 1, 1, 1, 0])).all()

    x3 = Dyn_op(x2, tseq.T - 4, 1)
    assert (x3 == np.array([0, 0, 0, 1, 1, 1])).all()

    x4 = Dyn_op(x3, tseq.T - 5, 1)
    assert (x4 == np.array([1, 0, 0, 0, 1, 1])).all()
