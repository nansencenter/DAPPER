"""Load rc: default settings"""
from dapper import *

import json

class JsonDict(dict):
    """Provide json pretty-printing"""
    np_short = True

    def __repr__(self):
        lines = json.dumps(self, indent=4, sort_keys=False, default=self.fallback)
        lines = lines.split("\n")
        lines = self.de_escape_newlines(lines)
        cropr = lambda t: t[:80] + ("" if len(t)<80 else "...")
        lines = [cropr(ln) for ln in lines]
        return "\n".join(lines)

    def __str__(self):
        return repr(self)

    def fallback(self, obj):
        if JsonDict.np_short and 'numpy.ndarray' in str(type(obj)):
            return f"ndarray, shape {obj.shape}, dtype {obj.dtype}"
        else:
            return str(obj)

    def de_escape_newlines(self,lines):
        """De-escape newlines. Include current indent."""
        new = []
        for line in lines:
            if "\\n" in line:
                hang = 2
                ____ = " " * (len(line) - len(line.lstrip()) + 2)
                line = line.replace('": "', '":\n'+____) # add newline at :
                line = line.replace('\\n', '\n'+____) # de-escape newlines
                line = line.split("\n")
                line[-1] = line[-1].rstrip('"')
            else:
                line = [line]
            new += line
        return new


dirs = DotDict(dapper = Path(__file__).absolute().parent)

# Load rc files from [dapper, user-home, sys.path[0]]
_rc_locations = [dirs.dapper, Path("~").expanduser(), Path(sys.path[0])]
_rc = configparser.ConfigParser()
_rc.read(x/'dpr_config.ini' for x in _rc_locations)
# Convert to dict
rc = DotDict({s:DotDict(_rc.items(s)) for s in _rc.sections() if s not in ['int','bool']})
# Parse styles
x = rc.plot.styles
x = x.replace('$dapper',str(dirs.dapper))
x = x.replace('/',os.path.sep)
rc.plot.styles = x
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
del dirs, x, d
# Create dirs
for d in rc.dirs:
    os.makedirs(rc.dirs[d], exist_ok=True)



##################################
# Plotting settings
##################################
import matplotlib as mpl

# user_is_patrick
import getpass
user_is_patrick = getpass.getuser() == 'pataan'

if user_is_patrick:
    from sys import platform
    # Try to detect notebook
    try:
        __IPYTHON__
        from IPython import get_ipython
        is_notebook_or_qt = 'zmq' in str(type(get_ipython())).lower()
    except (NameError,ImportError):
        is_notebook_or_qt = False
    # Switch backend
    if is_notebook_or_qt:
        pass # Don't change backend
    elif platform == 'darwin':
        try:
            mpl.use('Qt5Agg') # pip install PyQt5 (and get_screen_size needs qtpy).
            import matplotlib.pyplot # Trigger (i.e. test) the actual import
        except ImportError:
            # Was prettier/stabler/faster than Qt4Agg, but Qt5Agg has caught up.
            mpl.use('MacOSX')

_BE = mpl.get_backend().lower()
_LP = rc.liveplotting_enabled
if _LP: # Check if we should disable anyway:
    _LP &= not any([_BE==x for x in ['agg','ps','pdf','svg','cairo','gdk']])
    # Also disable for inline backends, which are buggy with liveplotting
    _LP &= 'inline' not in _BE
    _LP &= 'nbagg'  not in _BE
    if not _LP:
        print("\nWarning: interactive/live plotting is turned on in dpr_config.ini,")
        print("but is not supported by current backend: %s."%mpl.get_backend())
        print("To enable it, try using another backend, e.g., mpl.use('Qt5Agg').\n")
rc.liveplotting_enabled = _LP

# Get Matlab-like interface, and enable interactive plotting
import matplotlib.pyplot as plt 
plt.ion()

# Styles
plt.style.use(rc.plot.styles.split(","))
