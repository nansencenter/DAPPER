"""Load default and user configurations into the rc dict.

This includes disabling liveplotting if necessary.
"""

import os
import sys
from pathlib import Path

import matplotlib as mpl
import yaml
from mpl_tools import is_using_interactive_backend
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
# Disable rc.liveplotting in case of non-interactive backends
##################################
# Otherwise, warnings are thrown on every occurence of plt.pause
# (though not plot_pause), and (who knows) maybe errors too.
# Also, the assimilation slows down, even though nothing is shown.
LP = rc.liveplotting
if LP and not is_using_interactive_backend():
    # Note: plot_pause could maybe be adapted to also work for
    # "inline" backend (which is not striclty interactive), but I think
    # this would be buggy, and is incompatible with a "stop" button.
    print("\nWarning: You have not disableed interactive/live plotting"
          " in your dpr_config.py,"
          " but this is not supported by the current matplotlib backend:"
          f" {mpl.get_backend()}. To enable it, try using another backend.\n")
    LP = False
rc.liveplotting = LP
