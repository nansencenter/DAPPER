"""Contains the Fortran-90 code of the model."""

from pathlib import Path

# Exclude py_mod.*, e.g. "py_mod.cpython-38-x86_64-linux-gnu"
__here__ = Path(__file__).parent
excluded = __here__.glob("py_mod*")
excluded = [str(x.relative_to(__here__).with_suffix("")) for x in excluded]
__pdoc__ = {x: False for x in excluded}
