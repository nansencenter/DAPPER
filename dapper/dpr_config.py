"""Load default and user configurations into the `rc` dict.

The `rc` dict can be updated (after startup) as any normal dict.

The config file should reside in your `$HOME/`, or `$HOME/.config/` or `$PWD/`.
If several exist, the last one found (from the above ordering) is used.
The default configuration is given below.

```yaml
--8<-- "dapper/dpr_config.yaml"
```
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
        f = d / (prefix + "dpr_config.yaml")
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
# Otherwise, warnings are thrown on every occurence of plt.pause (not plot_pause),
# and assimilate() slows down, even though nothing is shown.
if rc.liveplotting and not is_using_interactive_backend():
    print(
        "\nWarning: You have not disableed interactive/live plotting",
        "in your dpr_config.yaml,",
        "but this is not supported by the current matplotlib backend:",
        f"{mpl.get_backend()}. To enable it, try using another backend.\n",
    )
    rc.liveplotting = False
