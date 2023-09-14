"""Check if numpy random number generation algorithms have changed."""

import numpy as np

from dapper.tools.seeding import rng, set_seed

sd = 9427845


def test_1():
    set_seed(sd)
    assert np.isclose(rng.random(), 0.2656374066419439)


def test_2():
    set_seed(sd)
    assert np.isclose(rng.standard_normal(), 0.23094045831569926)


def test_3():
    set_seed(sd)
    assert np.isclose(rng.choice(np.arange(99), 1), 12)
