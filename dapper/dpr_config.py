"""Load DAPPER configuration settings.

View configuration using
>>> print(rc)

Override the defaults by putting a file ``dpr_config.py``
in your home or cwd, containing (for example):
``
rc = {
    "data_root": "my_preferred_data_location",
    "welcome_message": False,
}
``
"""
from dapper import *

from importlib import import_module

####################
#  Default config  #
####################
rc = DotDict(
    # Where to store the experimental settings and results.
    # For example, you don't want this to be in your Dropbox.
    # Use "$cwd" for cwd, "$dapper" for where the DAPPER dir is.
    data_root = "~",
    # Methods used to average multivariate ("field") stats:
    field_summary_methods='m,rms,ma',
    # Curtail heavy computations:
    comp_threshold_a = -1,
    comp_threshold_b = 51,
    comp_threshold_c = -1,
    # Default significant figures:
    sigfig = 4,
    # Store stats between analysis times?
    store_u = False,
    # Enable liveplotting?
    liveplotting_enabled = True,
    # Print startup message?
    welcome_message = True,
    # Place (certain) figures automatically (buggy) ?
    place_figs = False,
)

##################################
# Load user configurations
##################################
for d in ["~", sys.path[0]]:
    d = Path(d).expanduser()
    f = "dpr_config"
    if (d/f).with_suffix(".py").is_file():
        # https://stackoverflow.com/a/129374
        sys.path.insert(0,str(d))
        mod = import_module(f)
        rc.update(mod.rc)
        sys.path = sys.path[1:]


##################################
# Dir paths
##################################
rc.dirs = DotDict()
rc.dirs.dapper = Path(__file__).absolute().parent
rc.dirs.DAPPER = rc.dirs.dapper.parent
# Data path
x = rc.pop("data_root")
if   x.lower()=="$cwd"   : x = Path.cwd()
elif x.lower()=="$dapper": x = rc.dirs.DAPPER
else                     : x = Path(x)
rc.dirs.data = x / "dpr_data"
rc.dirs.samples = rc.dirs.data / "samples"

# Expanduser, create dir
for d in rc.dirs:
    rc.dirs[d] = rc.dirs[d].expanduser()
    os.makedirs(rc.dirs[d], exist_ok=True)


##################################
# Plotting settings
##################################
BE = mpl.get_backend().lower()
LP = rc.liveplotting_enabled
if LP: # Check if we should disable anyway:
    non_interactive = ['agg','ps','pdf','svg','cairo','gdk']
    LP &= not any([BE==x for x in non_interactive])
    # Also disable for inline backends, which are buggy with liveplotting
    LP &= 'inline' not in BE
    LP &= 'nbagg'  not in BE
    if not LP:
        print("\nWarning: You have not disableed interactive/live plotting"
              " in your dpr_config.py,"
              " but this is not supported by current backend:"
              f" {mpl.get_backend()}."
              " To enable it, try using another backend, e.g., mpl.use('Qt5Agg').\n")
rc.liveplotting_enabled = LP
