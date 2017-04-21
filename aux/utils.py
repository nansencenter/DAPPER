# Utilities (non-math)

from common import *

#########################################
# Progressbar
#########################################
def noobar(itrble, desc):
  """Simple progress bar. To be used if tqdm not installed."""
  L  = len(itrble)
  print('{}: {: >2d}'.format(desc,0), end='')
  for k,i in enumerate(itrble):
    yield i
    p = (k+1)/L
    e = '' if k<(L-1) else '\n'
    print('\b\b\b\b {: >2d}%'.format(int(100*p)), end=e)
    sys.stdout.flush()

# Get progbar description by inspecting caller function.
import inspect
def pdesc(desc):
  if desc is not None:
    return desc
  try:
    #stackoverflow.com/q/15608987
    DAC_name  = inspect.stack()[3].frame.f_locals['name_hook']
  except (KeyError, AttributeError):
    #stackoverflow.com/a/900404
    DAC_name  = inspect.stack()[2].function
  return DAC_name 

# Define progbar as tqdm or noobar
try:
  import tqdm
  def progbar(inds, desc=None, leave=1):
    return tqdm.tqdm(inds,desc=pdesc(desc),leave=leave)
except ImportError as err:
  install_warn(err)
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



# Terminal color codes. Use:
termcolors={
    'blue'      : '\033[94m',
    'green'     : '\033[92m',
    'OKblue'    : '\033[94m',
    'OKgreen'   : '\033[92m',
    'WARNING'   : '\033[93m',
    'FAIL'      : '\033[91m',
    'ENDC'      : '\033[0m' ,
    'header'    : '\033[95m',
    'bold'      : '\033[1m' ,
    'underline' : '\033[4m' ,
}

def print_c(*kargs,color='blue',**kwargs):
  s = ' '.join([str(k) for k in kargs])
  print(termcolors[color] + s + termcolors['ENDC'],**kwargs)


# Local np.set_printoptions. stackoverflow.com/a/2891805/38281
import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


try:
  import tabulate as tabulate_orig
  tabulate_orig.MIN_PADDING = 0
  def _tabulate(data,headr):
    data  = list(map(list, zip(*data))) # Transpose
    inds  = ['[{}]'.format(d) for d in range(len(data))] # Gen nice inds
    return tabulate_orig.tabulate(data,headr,showindex=inds)
except ImportError as err:
  install_warn(err)
  # pandas more widespread than tabulate, but slower to import
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
  >>> print(tabulate(cfgs.distinct_attrs()))
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


class MLR_Print:
  """
  Multi-Line, Recursive repr (print) functionality.
  Set class variables to change look:
   - 'indent': indentation per level
   - 'ch': character to use for "spine" (e.g. '|' or ' ')
  """
  indent=3
  ch='.'

  # Recursion monitoring.
  _stack=[]
  # Don't reference using .__class__ !!!
  # Coz will use subclass, and different instances of _stack.

  def __repr__(self):
    NL = '\n' + self.ch + ' '*(self.indent-1)

    # Recursion prevention
    is_top_level = False
    if MLR_Print._stack == []:
      is_top_level = True
    if self in MLR_Print._stack:
      return "**Recursion**"
    MLR_Print._stack += [self]

    # Print self type+name
    s = "<" + type(self).__name__ + '>'
    if hasattr(self,'name'):
      s += ': ' + self.name

    # Print self's variables
    for key in sorted(vars(self),key=lambda x: x.lower()):
      if key == 'name': pass
      else:
        s += NL + key + ': '
        t  = repr(getattr(self,key)) # sub-string
        if '\n' not in t:
          # Print regularly
          s += t
        else:
          # Activate multi-line printing
          s +=                NL+' '*self.indent  # first line
          s += t.replace('\n',NL+' '*self.indent) # other lines

    # Empty _stack when top-level printing finished
    if is_top_level:
      MLR_Print._stack = []

    return s


#########################################
# Writing / Loading Independent experiments
#########################################

import glob, re
def find_last_v(dirpath):
  if not dirpath.endswith(os.path.sep):
    dirpath += os.path.sep
  ls = glob.glob(dirpath+'v*.npz')
  if ls == []:
    return 0
  #v = max([int(x.split(dirpath)[1].split('v')[1].split('_')[0]) for x in ls])
  v = max([int(re.search(dirpath + 'v([1-9]*).*\.npz',x).group(1)) for x in ls])
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



#########################################
# Multiprocessing
#########################################
NPROC = 4
import multiprocessing, signal
from functools import partial
def multiproc_map(func,xx,**kwargs):
  """
  Multiprocessing.
  Basically a wrapper for multiprocessing.pool.map(). Deals with
   - additional, fixed arguments.
   - KeyboardInterruption (python bug?)
  """
  # stackoverflow.com/a/35134329/38281
  orig = signal.signal(signal.SIGINT, signal.SIG_IGN)
  pool = multiprocessing.Pool(NPROC)
  signal.signal(signal.SIGINT, orig)
  try:
    f   = partial(func,**kwargs) # stackoverflow.com/a/5443941/38281
    res = pool.map_async(f, xx)
    res = res.get(60)
  except KeyboardInterrupt as e:
    # Not sure why, but something this hangs,
    # so we repeat the try-block to catch another interrupt.
    try:
      traceback.print_tb(e.__traceback__)
      pool.terminate()
      sys.exit(0)
    except KeyboardInterrupt as e2:
      traceback.print_tb(e2.__traceback__)
      pool.terminate()
      sys.exit(0)
  else:
    pool.close()
  pool.join()
  return res



#########################################
# Tic-toc
#########################################

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





#########################################
# Misc
#########################################
class Bunch(dict):
  def __init__(self,**kw):
    dict.__init__(self,kw)
    self.__dict__ = self


def filter_out(orig_list,*unwanted):
  """
  Returns new list from orig_list with unwanted removed.
  Also supports re.search() by inputting a re.compile('patrn') among unwanted.
  """
  new = []
  for word in orig_list:
    for x in unwanted:
      try:
        # Regex compare
        rm = x.search(word)
      except AttributeError:
        # String compare
        rm = x==word
      # Insert
      if rm:
        break
    else:
      new.append(word)
  return new

def safe_del(dct,*args):
  "Returns new dct from dct with args removed. Also supports regex."
  #Non-regex version:
  #dct = {k:v for k,v in dct.items() if k not in args}
  out = dct.copy()
  for key in dct.keys():
    for arg in args:
      try:
        # Regex compare
        rm = bool(arg.match(key))
      except AttributeError:
        # String compare
        rm = arg==key
      # Insert in new
      if rm:
        del out[key]
  return out

def select(orig,*args):
  """
  Used to forward (pass on) keyword arguments.
  Pro: returns empty dict if args not found.
  Con: enforces same name at all levels.
  Consider other approaches:
   - a 'safe_get()' which falls-back to some defaults.
   - using a DFLT class, within which the
     defaults are stored and accessible for all levels and signatures.
  """
  return {key:orig[key] for key in args if key in orig}


