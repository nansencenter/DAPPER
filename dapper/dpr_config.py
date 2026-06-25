"""Runtime configuration for DAPPER.

Defaults are set below. Override at runtime:

    import dapper
    dapper.rc.liveplotting = False

Or use environment variables: `DAPPER_DATA_ROOT`, `DAPPER_LIVEPLOTTING`,
`DAPPER_PROGBAR` (the latter two are useful for remote jobs).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
from mpl_tools import is_using_interactive_backend


@dataclass
class Comps:
    """Curtail heavy computations."""

    error_only: bool = False
    max_spectral: int = 51


@dataclass
class DirPaths:
    dapper: Path = field(default_factory=Path)
    DAPPER: Path = field(default_factory=Path)
    data: Path = field(default_factory=Path)
    samples: Path = field(default_factory=Path)


@dataclass
class RC:
    # Methods used to average multivariate ("field") stats.
    # PS: If the model is computationally trivial, the stats computations take
    # up most time. Therefore, not all of these are activated by default.
    field_summaries: list[str] = field(
        default_factory=lambda: [
            "m",  # plain mean
            # "ms",   # mean-square
            "rms",  # root-mean-square
            "ma",  # mean-absolute
            # "gm",   # geometric mean
        ]
    )
    store_i: bool = False  # Store stats between analysis times?
    sigfig: int = 4  # Default significant figures
    liveplotting: bool = True  # Enable liveplotting?
    progbar: bool = True  # Enable progress bars?
    place_figs: bool = False  # Place (certain) figures automatically (experimental)?
    comps: Comps = field(default_factory=Comps)
    dirs: DirPaths = field(default_factory=DirPaths)


rc = RC()

##################################
# Apply environment variable overrides
##################################
# Where to store experimental settings and results
# (e.g. you don't want this to be in your Dropbox).
_data_root: str | Path = os.environ.get("DAPPER_DATA_ROOT", "~")

if (_liveplotting := os.environ.get("DAPPER_LIVEPLOTTING")) is not None:
    rc.liveplotting = _liveplotting.lower() not in ("no", "false", "0")

if (_progbar := os.environ.get("DAPPER_PROGBAR")) is not None:
    rc.progbar = _progbar.lower() not in ("no", "false", "0")

##################################
# Setup dir paths
##################################
_dapper_dir = Path(__file__).absolute().parent
rc.dirs.dapper = _dapper_dir
rc.dirs.DAPPER = _dapper_dir.parent
rc.dirs.data = Path(_data_root).expanduser() / "dpr_data"
rc.dirs.samples = rc.dirs.data / "samples"

for _name, _path in vars(rc.dirs).items():
    setattr(rc.dirs, _name, Path(_path).expanduser())
    os.makedirs(getattr(rc.dirs, _name), exist_ok=True)

##################################
# Disable liveplotting if non-interactive
##################################
if rc.liveplotting and not is_using_interactive_backend():
    print(
        "\nWarning: You have not disabled interactive/live plotting,",
        "but this is not supported by the current matplotlib backend:",
        f"{mpl.get_backend()}. To enable it, try using another backend.\n",
    )
    rc.liveplotting = False
