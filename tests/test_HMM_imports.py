import pytest
import os
from pathlib import Path
from importlib import import_module

HMMs = []

for root, dir, files in os.walk("."):
    if "mods" in root:

        # Can uncomment if you have compiled and generated samples
        # if "QG" in root:
            # continue

        for f in sorted(files):
            if f.endswith(".py"):
                filepath = Path(root) / f

                lines = "".join(open(filepath).readlines())
                if "HiddenMarkovModel" in lines:
                    HMMs.append(filepath)


@pytest.mark.parametrize(("path"), HMMs)
def import_path(path):
    p = str(path.with_suffix("")).replace("/", ".")
    import_module(p)
    assert True
