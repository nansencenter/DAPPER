"""Data Assimilation with Python: a Package for Experimental Research

.. include:: ./README.md
"""

__version__ = "1.2.3"

import sys

# Imports used by examples
from .dpr_config import rc
from .tools.magic import magic_naming, spell_out
from .tools.rounding import round2, round2sigfig
from .tools.seeding import set_seed
from .xp_launch import combinator, seed_and_simulate, xpList
from .xp_process import (default_fig_adjustments, default_styles,
                         discretize_cmap, find_latest_run, load_xps,
                         make_label, rel_index, xpSpace)
