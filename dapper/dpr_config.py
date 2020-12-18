"""Load default and user configurations into the rc dict.

This includes disabling liveplotting if necessary."""

import os
import sys
from pathlib import Path

import matplotlib as mpl
import yaml
from struct_tools import DotDict

##################################
# Load configurations
##################################
dapper_dir = Path(__file__).absolute().parent
rc = DotDict()
for d in [dapper_dir, "~", sys.path[0]]:
    d = Path(d).expanduser()
    for prefix in [".", ""]:
        f = d / (prefix+"dpr_config.yaml")
        if f.is_file():
            dct = yaml.load(open(f), Loader=yaml.SafeLoader)
            if dct:
                rc.update(dct)


##################################
# Setup dir paths
##################################
rc.dirs = DotDict()
rc.dirs.dapper = dapper_dir
rc.dirs.DAPPER = rc.dirs.dapper.parent
# Data path
x = rc.pop("data_root")
if x.lower() == "$cwd":
    x = Path.cwd()
elif x.lower() == "$dapper":
    x = rc.dirs.DAPPER
else:
    x = Path(x)
rc.dirs.data = x / "dpr_data"
rc.dirs.samples = rc.dirs.data / "samples"

# Expanduser, create dir
for d in rc.dirs:
    rc.dirs[d] = rc.dirs[d].expanduser()
    os.makedirs(rc.dirs[d], exist_ok=True)


##################################
# Disable rc.liveplotting ?
##################################
LP = rc.liveplotting
if LP:
    backend = mpl.get_backend().lower()
    non_interactive = ['agg', 'ps', 'pdf', 'svg', 'cairo', 'gdk']
    LP &= not any([backend == x for x in non_interactive])
    # Also disable for inline backends, which are buggy with liveplotting
    LP &= 'inline' not in backend
    LP &= 'nbagg' not in backend
    if not LP:
        print("\nWarning: You have not disableed interactive/live plotting"
              " in your dpr_config.py,"
              " but this is not supported by current backend:"
              f" {mpl.get_backend()}."
              " To enable it, try using another backend,"
              " e.g., mpl.use('Qt5Agg').\n")
rc.liveplotting = LP
