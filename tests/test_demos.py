"""Test `demo.py` files in model dirs."""

from importlib import import_module

import pytest
from matplotlib import pyplot as plt

import dapper as dpr

plt.ion()


@pytest.mark.parametrize(("path"), dpr._find_demos())
def test_imp(path):
    import_module(path)
    assert True
