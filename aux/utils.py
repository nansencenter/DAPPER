# Utilities (non-math)

from common import *

#########################################
# Progressbar
#########################################
def noobar(itrble, desc):
  """Simple progress bar. To be used if tqdm not installed."""
  L  = len(itrble)
  print('{}: {: >2d}'.format(desc,0),end='')
  for k,i in enumerate(itrble):
    yield i
    p = (k+1)/L
    e = '' if k<(L-1) else '\n'
    print('\b\b\b\b {: >2d}%'.format( \
        int(100*p)),end=e)
    sys.stdout.flush()

try:
  # Get progbar description by inspecting caller function.
  import inspect
  def pdesc(desc):
    if desc is not None:
      return desc
    #try:
      #stackoverflow.com/q/15608987
      #DAM_name  = inspect.stack()[2].frame.f_locals['cfg'].name
    #except (KeyError, AttributeError):
    #stackoverflow.com/a/900404
    DAM_name  = inspect.stack()[2].function
    return DAM_name 
except ImportError:
  def pdesc(desc): 
    return desc or "Prog."

# Define progbar as tqdm or noobar
try:
  import tqdm
  def progbar(inds, desc=None, leave=1):
    return tqdm.tqdm(inds,desc=pdesc(desc),leave=leave)
except ImportError:
  def progbar(inds, desc=None, leave=1):
    return noobar(inds,desc=pdesc(desc))




#########################################
# Console input / output
#########################################

#stackoverflow.com/q/292095
import select
def poll_input():
  i,o,e = select.select([sys.stdin],[],[],0.0001)
  for s in i: # Only happens if <Enter> has been pressed
    if s == sys.stdin:
      return sys.stdin.readline()
  return None

# Can't get thread solution working (in combination with getch()):
# stackoverflow.com/a/25442391/38281

# stackoverflow.com/a/21659588/38281
# (Wait for) any key:
def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch
    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch
getch = _find_getch()


# RGB constants
cwhite = array([1,1,1])
cred   = array([1,0,0])
cgreen = array([0,1,0])
cblue  = array([0,0,1])

# Terminal color codes. Use:
# print(bcolors.WARNING + "Warning: test" + bcolors.ENDC)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_blue(*kargs):
  s = ' '.join([str(k) for k in kargs])
  print(bcolors.OKBLUE + s + bcolors.ENDC)



try:
  from tabulate import tabulate as tabulate_orig
  def _tabulate(data,headr):
    data  = list(map(list, zip(*data))) # Transpose
    inds  = ['[{}]'.format(d) for d in range(len(data))] # Gen nice inds
    return tabulate_orig(data,headr,showindex=inds)
except ImportError:
  # pandas more common than tabulate, but slower to import
  import pandas
  pandas.options.display.width = None # Auto-adjust linewidth
  pandas.options.display.colheader_justify = 'center'
  def _tabulate(data,headr):
    df = pandas.DataFrame.from_items([i for i in zip(headr,data)])
    return df.__repr__()

def tabulate(data,headr=(),formatters=()):
  """Pre-processor for tabulate().
  data:  list-of-lists, whose 'rows' will be printed as columns.
         This coincides with the output of Dict.values().
  headr: list or tuple.
  If 'data' is a dict, then the keys will be the headr.
  formatter: define formats to apply before relaying to pandas.
        Default: attr.__name__ (when applicable).
  Example:
  >>> print(tabulate(DAMs.distinct_attrs))
  """

  if hasattr(data,'keys'):
    headr = list(data)
    data  = data.values()

  if not formatters:
    formatters = ({
        'test'  : lambda x: hasattr(x,'__name__'),
        'format': lambda x: x.__name__
        },)
  for f in formatters:
    data = [[f['format'](j) for j in row] if f['test'](row[0]) else row for row in data]

  return _tabulate(data, headr)


#########################################
# Writing / Loading
#########################################

import glob, re
def find_last_v(dirpath):
  if not dirpath.endswith(os.path.sep):
    dirpath += os.path.sep
  ls = glob.glob(dirpath+'v*.npz')
  if ls == []:
    return 0
  #v = np.max([int(x.split(dirpath)[1].split('v')[1].split('_')[0]) for x in ls])
  v = np.max([int(re.search(dirpath + 'v([1-9]*).*\.npz',x).group(1)) for x in ls])
  return v

def rel_name(filepath,rel_to='./'):
  fn = os.path.relpath(filepath,rel_to)
  fn, ext = os.path.splitext(fn)
  return fn

from os import makedirs
def save_dir(filepath):
  """Make dir DAPPER/data/filepath_without_ext"""
  dirpath = 'data' + os.path.sep\
      + rel_name(filepath) + os.path.sep
  makedirs(dirpath, exist_ok=True)
  return dirpath

def parse_parrallelization_args():
  """Get experiment version (v) and indices (inds) from command-line."""
  i_v    = sys.argv.index('save_path')+1
  i_inds = sys.argv.index('inds')+1
  assert i_v < i_inds
  save_path = sys.argv[i_v]
  inds   = sys.argv[i_inds:]
  inds   = [int(x) for x in inds]
  return save_path, inds

from subprocess import call
import time
def parallelize(script,inds,max_workers=4):
  save_path  = save_dir(script)
  v          = find_last_v(save_path) + 1
  v          = 'v' + str(v)
  save_path += v
  tmp        = 'tmp_' + v + '_t' + str(time.time())[-2:]
  with open(tmp,'w') as f:
    f.write('# Auto-generated for experiment parallelization.\n\n')
    f.write('source ~/.screenrc\n\n')
    for w in range(max_workers):
      iiw = [str(x) for x in inds[np.mod(inds,max_workers)==w]]
      f.write('screen -t exp_' + '_'.join(iiw) + ' python -i ' + script
          + ' save_path ' + save_path
          + ' inds ' + ' '.join(iiw) + '\n')
  sleep(0.2)
  call(['screen', '-S', v,'-c', tmp])
  sleep(0.2)
  os.remove(tmp)


def save_data(path,inds,**kwargs):
  path += '_inds_' + '_'.join([str(x) for x in inds])
  print('Saving to',path)
  np.savez(path,**kwargs)




#import time
class Timer():
  """
  Usage:
  with Timer('<description>'):
    do_stuff()
  """
  def __init__(self, name=None):
      self.name = name

  def __enter__(self):
      self.tstart = time.time()

  def __exit__(self, type, value, traceback):
      #pass # Turn off timer messages
      if self.name:
          print('[%s]' % self.name, end='')
      print('Elapsed: %s' % (time.time() - self.tstart))
