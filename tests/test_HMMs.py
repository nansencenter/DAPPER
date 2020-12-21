"""Test HMM found in model directories."""

import os
from importlib import import_module
from pathlib import Path

import pytest

import dapper.tools.progressbar
from dapper.mods import HiddenMarkovModel

dapper.tools.progressbar.disable_progbar = True

modules_with_HMM = []

for root, dir, files in os.walk("."):
    if "mods" in root:

        if os.environ.get("TRAVIS", False) and ("QG" in root):
            continue

        for f in sorted(files):
            if f.endswith(".py"):
                filepath = Path(root) / f

                lines = "".join(open(filepath).readlines())
                if "HiddenMarkovModel" in lines:
                    modules_with_HMM.append(filepath)


@pytest.mark.parametrize(("path"), modules_with_HMM, ids=str)
def test_HMM(path):
    """Test that any HMM in module can be simulated."""
    p = str(path.with_suffix("")).replace("/", ".")
    module = import_module(p)

    def exclude(key, HMM):
        """Exclude HMMs that are not testable w/o further configuration."""
        if key == "HMM_trunc":
            return True
        return False

    for key, HMM in vars(module).items():
        if isinstance(HMM, HiddenMarkovModel) and not exclude(key, HMM):
            HMM.t.BurnIn = 0
            HMM.t.KObs = 1
            xx, yy = HMM.simulate()
            assert True
