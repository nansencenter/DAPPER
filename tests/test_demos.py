import pytest
from matplotlib import pyplot as plt
from importlib import import_module

import dapper as dpr

plt.ion()


@pytest.mark.parametrize(("path"), dpr._find_demos())
def test_imp(path):
    import_module(path)
    assert True
