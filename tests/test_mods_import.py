import pytest
import os
from pathlib import Path
from importlib import import_module

HMMs = []

for root, dir, files in os.walk("."):
    if "mods" in root:
        for f in files:
            if f.endswith(".py"):
                filepath = Path(root) / f
                lines = "".join(open(filepath).readlines())
                if "HiddenMarkovModel" in lines:
                    HMMs.append(filepath)

@pytest.mark.parametrize(("module_path"), HMMs)
def test_tables_L63(module_path):
    p = str(module_path.with_suffix("")).replace("/", ".")
    import_module(p)
    assert True
