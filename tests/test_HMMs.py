"""Test HMM found in model directories."""

import os
from importlib import import_module

import pytest

from dapper.dpr_config import rc
from dapper.mods import HiddenMarkovModel


def _defines_HMM(path):

    # Don't run QG on CI or Tox
    if "QG" in path.parts and (
            os.environ.get("IS_CI", False) or
            os.environ.get("IS_TOX", False)):
        return False

    if (
        path.suffix == ".py"
        and path.stem != "__init__"
        and "HiddenMarkovModel" in "".join(open(path))
    ):
        return True

    return False


mods = rc.dirs.dapper / "mods"
root = rc.dirs.dapper
HMMs = [p.relative_to(root) for p in mods.glob("**/*.py") if _defines_HMM(p)]


@pytest.mark.parametrize(("path"), HMMs, ids=str)
def test_HMM(path):
    """Test that any HMM in module can be simulated."""
    p = "." + str(path.with_suffix("")).replace("/", ".")
    module = import_module(p, root.stem)

    def exclude(key, HMM):
        """Exclude certain, untestable HMMs"""
        if key == "HMM_trunc":
            return True
        return False

    for key, HMM in vars(module).items():
        if isinstance(HMM, HiddenMarkovModel) and not exclude(key, HMM):
            HMM.tseq.BurnIn = 0
            HMM.tseq.Ko = 1
            xx, yy = HMM.simulate()
            assert True
