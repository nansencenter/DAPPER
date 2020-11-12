import pytest
from matplotlib import pyplot as plt
from importlib import import_module

import dapper as dpr

demos = []
for d in (dpr.rc.dirs.dapper / "mods").iterdir():
    x = d/"demo.py"
    if x.is_file():
        demos.append(x.relative_to(dpr.rc.dirs.DAPPER))


plt.ion()


@pytest.mark.parametrize(("path"), demos)
def test_imp(path):
    x = str(path.with_suffix("")).replace("/", ".")
    import_module(x)
    assert True
