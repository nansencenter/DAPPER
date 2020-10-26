"""Load rc: default settings"""
from dapper import *

##################################
# Dirs
##################################

dirs = DotDict(dapper = Path(__file__).absolute().parent)

# Load rc files from [dapper, user-home, sys.path[0]]
_rc_locations = [dirs.dapper, Path("~").expanduser(), Path(sys.path[0])]
_rc = configparser.ConfigParser()
_rc.read(x/'dpr_config.ini' for x in _rc_locations)
# Convert to dict
rc = DotDict({s:DotDict(_rc.items(s)) for s in _rc.sections() if s not in ['int','bool']})
# Parse sections
for x in _rc['int' ]: rc[x] = _rc['int' ].getint(x)
for x in _rc['bool']: rc[x] = _rc['bool'].getboolean(x)

# Define paths
x = rc.dirs.data
if   x=="cwd"    : _root = Path.cwd()
elif x=="$dapper": _root = dirs.DAPPER
else             : _root = Path(x)
dirs.DAPPER  = dirs.dapper.parent
dirs.data    = _root / "dpr_data"
dirs.samples = dirs.data / "samples"
for d in dirs:
    dirs[d] = dirs[d].expanduser()
rc.dirs = dirs

# Create dirs
for d in rc.dirs:
    os.makedirs(rc.dirs[d], exist_ok=True)



##################################
# Plotting settings
##################################
_BE = mpl.get_backend().lower()
_LP = rc.liveplotting_enabled
if _LP: # Check if we should disable anyway:
    non_interactive = ['agg','ps','pdf','svg','cairo','gdk']
    _LP &= not any([_BE==x for x in non_interactive])
    # Also disable for inline backends, which are buggy with liveplotting
    _LP &= 'inline' not in _BE
    _LP &= 'nbagg'  not in _BE
    if not _LP:
        print("\nWarning: interactive/live plotting is turned on in dpr_config.ini,")
        print("but is not supported by current backend: %s."%mpl.get_backend())
        print("To enable it, try using another backend, e.g., mpl.use('Qt5Agg').\n")
rc.liveplotting_enabled = _LP
