"""Load default and user configurations into the `rc` dict.

The `rc` dict can be updated (after startup) as any normal dict. See the
[source](https://github.com/nansencenter/DAPPER/blob/master/dapper/dpr_config.yaml)
for the default configuration.
"""

import os
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
rc.loaded_from = []
for d in [dapper_dir, "~", "~/.config", "."]:
    d = Path(d).expanduser().absolute()
    for prefix in [".", ""]:
        f = d / (prefix+"dpr_config.yaml")
        if f.is_file():
            dct = yaml.load(open(f), Loader=yaml.SafeLoader)
            rc.loaded_from.append(str(f))
            if dct:
                if d == dapper_dir:
                    rc.update(dct)
                else:
                    for k in dct:
                        if k in rc:
                            rc[k] = dct[k]
                        else:
                            print(f"Warning: invalid key '{k}' in '{f}'")


##################################
# Setup dir paths
##################################
rc.dirs = DotDict()
rc.dirs.dapper = dapper_dir
rc.dirs.DAPPER = rc.dirs.dapper.parent
# Data path
x = rc.pop("data_root")
if x.lower() in ["$cwd", "$pwd"]:
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
    print("\nWarning: You have not disableed interactive/live plotting",
          "in your dpr_config.yaml,",
          "but this is not supported by the current matplotlib backend:",
          f"{mpl.get_backend()}. To enable it, try using another backend.\n")
    LP = False
rc.liveplotting = LP
