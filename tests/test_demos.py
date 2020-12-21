"""Test `demo.py` files in model dirs."""

from importlib import import_module

import matplotlib as mpl
import pytest
from matplotlib import pyplot as plt

import dapper as dpr

mpl.use("Qt5Agg")
plt.ion()


def _find_demos(as_path=False):
    "Find all model demo.py scripts."
    lst = []
    for d in (dpr.rc.dirs.dapper/"mods").iterdir():
        x = d/"demo.py"
        if x.is_file():
            x = x.relative_to(dpr.rc.dirs.DAPPER)
            if not as_path:
                x = str(x.with_suffix("")).replace("/", ".")
            lst.append(x)
    return lst


@pytest.mark.parametrize(("path"), _find_demos())
def test_imp(path):
    import_module(path)
    assert True
