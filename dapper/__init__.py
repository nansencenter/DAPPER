"""Data Assimilation with Python: a Package for Experimental Research

.. include:: ./README.md
"""

__version__ = "1.3.0"

# A parsimonious list of imports used in the examples
from .dpr_config import rc
from .tools.datafiles import find_latest_run, load_xps
from .tools.rounding import round2sigfig
from .tools.seeding import set_seed
from .xp_launch import combinator, seed_and_simulate, xpList
from .xp_process import xpSpace


# Exclude "demo.py" files from documentation.
# NB: pdoc complains/warns that it cannot find these files,
# but it still works fine. https://github.com/pdoc3/pdoc/issues/206
def _find_demos(as_path=False):
    """Discover all model demo.py scripts."""
    lst = []
    for d in (rc.dirs.dapper/"mods").iterdir():
        x = d/"demo.py"
        if x.is_file():
            x = x.relative_to(rc.dirs.DAPPER)
            if not as_path:
                x = str(x.with_suffix("")).replace("/", ".")
            lst.append(x)
    return lst


__pdoc__ = {k: False for k in _find_demos()}
