"""Root package of **DAPPER**
(Data Assimilation with Python: a Package for Experimental Research)

.. include:: ./README.md
"""

__version__ = "1.5.0"

# A parsimonious list of imports used in the examples
from .dpr_config import rc
from .tools.datafiles import find_latest_run, load_xps
from .tools.rounding import round2sigfig
from .tools.seeding import set_seed
from .xp_launch import combinator, seed_and_simulate, xpList
from .xp_process import xpSpace
