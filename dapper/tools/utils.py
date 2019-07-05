# Utilities (non-math)

from dapper import *

#########################################
# Progressbar
#########################################

import inspect
def pdesc(desc):
  "Get progbar description by caller inspection."

  if desc is not None:
    return desc

  # Assume the "progress" happens 2 calling levels above
  level = 2

  try:
    # Assuming we're in a DAC, go look above (i.e. stack[3]) for a name_hook.
    name = inspect.stack()[level+1].frame.f_locals['name_hook'] #so.com/q/15608987
  except (KeyError, AttributeError):
    # Otherwise: just get name of what's calling progbar (i.e. stack[2]) 
    name = inspect.stack()[level].function #so.com/a/900404
  return name


def noobar(iterable, desc=None, leave=None):
  """Simple progress bar. Fallback in case tqdm not installed."""
  if desc is None: desc = "Prog"
  L  = len(iterable)
  print('{}: {: >2d}'.format(desc,0), end='')
  for k,i in enumerate(iterable):
    yield i
    p = (k+1)/L
    e = '' if k<(L-1) else '\n'
    print('\b\b\b\b {: >2d}%'.format(int(100*p)), end=e)
    sys.stdout.flush()


# Define progbar
import tqdm
def progbar(iterable, desc=None, leave=1):
  "Prints a nice progress bar in the terminal"
  desc = pdesc(desc)
  # if is_notebook:
    # This fails in QtConsole (which also yields is_notebook==True)
    # return tqdm.tqdm_notebook(iterable,desc=desc,leave=leave)
  # else:
  return tqdm.tqdm(iterable,desc=desc,leave=leave,smoothing=0.3,dynamic_ncols=True)
  # Printing during the progbar loop (may occur with error printing)
  # can cause tqdm to freeze the entire execution. 
  # Seemingly, this is caused by their multiprocessing-safe stuff.
  # Disable this, as per github.com/tqdm/tqdm/issues/461#issuecomment-334343230
  # pb = tqdm.tqdm(...)
  # try: pb.get_lock().locks = []
  # except AttributeError: pass
  # return pb


# Set to True before a py.test (which doesn't like reading stdin)
disable_user_interaction = False

# Non-blocking, non-echo read1 from stdin.
try:
    # Linux. See Misc/read1_trials.py
    import termios, sys
    def set_term_settings(TS):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TS)
    def new_term_settings():
      "Make stdin.read non-echo and non-block"

      TS_old = termios.tcgetattr(sys.stdin)
      TS_new = termios.tcgetattr(sys.stdin)

      # Make tty non-echo.
      TS_new[3] = TS_new[3] & ~(termios.ECHO | termios.ICANON)

      # Make tty non-blocking.
      TS_new[6][termios.VMIN] = 0
      TS_new[6][termios.VTIME] = 0

      set_term_settings(TS_new)
      return TS_old

    def _read1():
      return os.read(sys.stdin.fileno(), 1)

    try:
        # Test setting/restoring settings 
        TS_old = new_term_settings()
        set_term_settings(TS_old)
        
        # Wrap the progressbar generator with temporary term settings
        # Note: could also do it when loading/exiting DAPPER (untested),
        # but it doesn't work to repeatedly do it within the assim loop.
        orig_progbar = progbar
        def progbar(iterable, desc=None, leave=1):
          TS_old = new_term_settings()
          try:
            for i in orig_progbar(iterable, pdesc(desc), leave):
                yield i
          finally:
            # Should restore settings both after normal termination
            # and if KeyboardInterrupt or other exception happened during loop.
            set_term_settings(TS_old)

    except:
        # Fails in non-terminal environments
        disable_user_interaction = True

except ImportError:
    # Windows
    import msvcrt
    def _read1():
      if msvcrt.kbhit():
        return msvcrt.getch()
      else:
        return None

def read1():
  "Get 1 character. Non-blocking, non-echoing."
  if disable_user_interaction: return None
  return _read1()



#########################################
# Path manipulation
#########################################

import socket
def save_dir(script,host=True):
  """Make dirs['data']/script/hostname, with some refinements."""
  host   = socket.gethostname().split('.')[0] if host else ''
  script = os.path.splitext(script)[0]
  root   = os.path.commonpath([os.path.abspath(x) for x in [dirs['data'],script]])
  script = os.path.relpath(script, root)
  sdir   = os.path.join(dirs['data'],script,host,'')
  os.makedirs(sdir, exist_ok=True)
  return sdir


#########################################
# Console input / output
#########################################

import inspect
def spell_out(*args):
  """
  Print (args) including variable names.
  Example:
  >>> print(3*2)
  >>> 3*2:
  >>> 6
  """
  frame  = inspect.stack()[1].frame
  lineno = frame.f_lineno

  # This does not work coz they (both) end reading when encountering a func def
  # code  = inspect.getsourcelines(frame)
  # code  = inspect.getsource(frame)

  # Instead, read the source manually...
  f = inspect.getfile(frame)
  if "ipython-input" in f:
    # ( does not work with pure python where f == "<stdin>" )
    line = inspect.getsource(frame)
  else:
    try:
      f = os.path.relpath(f)
      with open(f) as stream:
        line = stream.readlines()[lineno-1]
    except FileNotFoundError:
      line = "(Print command on line number " + str(lineno) + " [of unknown source])"

  # Find opening/closing brackets
  left  = line. find("(")
  right = line.rfind(")")

  # Print header
  with coloring(cFG.MAGENTA):
    print(line[left+1:right] + ":")
  # Print (normal)
  print(*args)


def print_together(*args):
  "Print 1D arrays stacked together."
  print(np.vstack(args).T)


# Local np.set_printoptions. stackoverflow.com/a/2891805/38281
import contextlib
@contextlib.contextmanager
@functools.wraps(np.set_printoptions)
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


import tabulate as tabulate_orig
tabulate_orig.MIN_PADDING = 0
def tabulate(data,headr=(),formatters=(),inds='nice'):
  """Pre-processor for tabulate().

  Main task: transpose 'data' (list-of-lists).

  If 'data' is a dict, the 'headr' will be keys.

  'formatter': define formats to apply before relaying to tabulate().
  Default: attr.__name__ (when applicable).

  Example::

    >>> print(tabulate(cfgs.distinct_attrs()))
  """

  # Extract dict
  if hasattr(data,'keys'):
    headr = list(data)
    data  = data.values()

  # Default formats
  if not formatters:
    formatters = ({
        'test'  : lambda x: hasattr(x,'__name__'),
        'format': lambda x: x.__name__
        },{
        'test'  : lambda x: isinstance(x,np.ndarray),
        'format': lambda x: "..." if isinstance(x,np.ndarray) else x
        },
        )
  # Apply formatting (if not applicable, data is just forwarded)
  for f in formatters:
    match = lambda row: any(f['test'](j) for j in row)
    formt = lambda row: [f['format'](j) for j in row]
    data = [formt(row) if match(row) else row for row in data]

  # Transpose
  data = list(map(list, zip(*data)))

  # Generate nice indices
  if inds=='nice':
    inds = ['[{}]'.format(d) for d in range(len(data))]
  else:
    pass # Should be True or False

  return tabulate_orig.tabulate(data,headr,showindex=inds)


def repr_type_and_name(thing):
  """Print thing's type [and name]"""
  s = "<" + type(thing).__name__ + '>'
  if hasattr(thing,'name'):
    s += ': ' + thing.name
  return s


# Adapted from stackoverflow.com/a/3603824
class ImmutableAttributes():
  """
  Freeze (make immutable) attributes of class instance.
  Applies to 
  """
  __isfrozen = False
  __keys     = None
  def __setattr__(self, key, value):
    #if self.__isfrozen and hasattr(self, key):
    if self.__isfrozen and key in self.__keys:
      raise AttributeError(
          "The attribute %r of %r has been frozen."%(key,type(self)))
    object.__setattr__(self, key, value)
  def _freeze(self,keys):
    self.__keys     = keys
    self.__isfrozen = True



from IPython.lib.pretty import pretty as pretty_repr
class NestedPrint:
  """Multi-Line, Recursive repr (print) functionality.

  Set class variables to change look:

   - 'indent': indentation per level
   - 'ch': character to use for "spine" (e.g. '|' or ' ')
   - 'ordr_by_linenum': 0: alphabetically, 1: linenumbr, -1: reverse
  """
  indent=3
  ch='.'
  ordr_by_linenum = 0

  # numpy print options
  threshold=10
  precision=None

  # Recursion monitoring.
  _stack=[] # Reference using NestedPrint._stack, ...
  # not self._stack or self.__class__, which reference sub-class "instance".

  # Reference using self.excluded, to access sub-class "instance".
  excluded = [] # Don't include in printing
  excluded.append(re.compile('^_')) # "Private"
  excluded.append('name') # Treated separately

  included = [] # Only print these (also determines ordering).
  ordering = [] # Determine ordering (with precedence over included).
  aliases  = {} # Rename attributes (vars)

  def __repr__(self):
    with printoptions(threshold=self.threshold,precision=self.precision):
      # new line chars
      NL = '\n' + self.ch + ' '*(self.indent-1)

      # Infinite recursion prevention
      is_top_level = False
      if NestedPrint._stack == []:
        is_top_level = True
      if self in NestedPrint._stack:
        return "**Recursion**"
      NestedPrint._stack += [self]

      # Use included or filter-out excluded
      keys = self.included or filter_out(vars(self), *self.excluded)

      # Process attribute repr's
      txts = {}
      for key in keys:
        t = pretty_repr(getattr(self,key)) # sub-repr
        if '\n' in t:
          # Activate multi-line printing
          t = t.replace('\n',NL+' '*self.indent)      # other lines
          t = NL+' '*self.indent + t                  # first line
        t = NL + self.aliases.get(key,key) + ': ' + t # key-name
        txts[key] = t

      def sortr(x):
        if x in self.ordering:
          key = -1000 + self.ordering.index(x)
        else:
          if self.ordr_by_linenum:
            key = self.ordr_by_linenum*txts[x].count('\n')
          else:
            key = x.lower()
            # Convert str to int (assuming ASCII) for comparison with above cases
            key = sum( ord(x)*128**i for i,x in enumerate(x[::-1]) )
        return key

      # Assemble string
      s = repr_type_and_name(self)
      for key in sorted(txts, key=sortr):
        s += txts[key]

      # Empty _stack when top-level printing finished
      if is_top_level:
        NestedPrint._stack = []

      return s


class AlignedDict(OrderedDict):
  """Provide aligned-printing for dict."""
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
  def __str__(self):
    L = max([len(s) for s in self.keys()], default=0)
    s = " "
    for key in self.keys():
      s += key.rjust(L)+": "+repr(self[key])+"\n "
    s = s[:-2]
    return s
  def __repr__(self):
    return type(self).__name__ + "(**{\n " + str(self).replace("\n",",\n") + "\n})"
  def _repr_pretty_(self, p, cycle):
    # Is implemented by OrderedDict, so must overwrite.
    if cycle: p.text('{...}')
    else:     p.text(self.__repr__())
  
class Bunch(dict):
  def __init__(self,**kw):
    dict.__init__(self,kw)
    self.__dict__ = self


# From stackoverflow.com/q/22797580 and more
class NamedFunc():
  "Provides custom repr for functions."
  def __init__(self,_func,_repr):
    self._func = _func
    self._repr = _repr
    #functools.update_wrapper(self, _func)
  def __call__(self, *args, **kw):
    return self._func(*args, **kw)
  def __repr__(self):
    argnames = self._func.__code__.co_varnames[
      :self._func.__code__.co_argcount]
    argnames = "("+",".join(argnames)+")"
    return "<NamedFunc>"+argnames+": "+self._repr

class NameFunc():
  "Decorator version"
  def __init__(self,name):
     self.fn_name = name
  def __call__(self,fn):
      return NamedFunc(fn,self.fn_name)



#########################################
# Misc
#########################################

# # Temporarily set attribute values
# @contextlib.contextmanager
# def set_tmp(obj,attr,val):
#     tmp = getattr(obj,attr)
#     setattr(obj,attr,val)
#     yield 
#     setattr(obj,attr,tmp)

import contextlib
@contextlib.contextmanager
def set_tmp(obj, attr, val):
    """
    Temporarily set an attribute.
    code.activestate.com/recipes/577089
    """
    was_there = False
    tmp = None
    if hasattr(obj, attr):
      try:
        if attr in obj.__dict__:
          was_there = True
      except AttributeError:
        if attr in obj.__slots__:
          was_there = True
      if was_there:
        tmp = getattr(obj, attr)
    setattr(obj, attr, val)

    yield #was_there, tmp

    if not was_there: delattr(obj, attr)
    else:             setattr(obj, attr, tmp)



# Better than tic-toc !
import time
class Timer():
  """Timer.

  Example::

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

def find_1st(xx):
  try:                  return next(x for x in xx if x)
  except StopIteration: return None
def find_1st_ind(xx):
  try:                  return next(k for k in range(len(xx)) if xx[k])
  except StopIteration: return None

# stackoverflow.com/a/2669120
def sorted_human( lst ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(lst, key = alphanum_key)

def keep_order_unique(arr):
  "Undo the sorting that np.unique() does."
  _, inds = np.unique(arr,return_index=True)
  return arr[np.sort(inds)]

def de_abbreviate(abbrev_d, abbreviations):
  "Expand any abbreviations (list of 2-tuples) that occur in abbrev_d (dict)."
  for a,b in abbreviations:
    if a in abbrev_d:
      abbrev_d[b] = abbrev_d[a]
      del abbrev_d[a]



def filter_out(orig_list,*unwanted,INV=False):
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
      if (not INV)==bool(rm):
        break
    else:
      new.append(word)
  return new

def all_but_1_is_None(*args):
  "Check if only 1 of the items in list are Truthy"
  return sum(x is not None for x in args) == 1

# From stackoverflow.com/q/3012421
class lazy_property(object):
    '''
    Lazy evaluation of property.
    Should represent non-mutable data,
    as it replaces itself.
    '''
    def __init__(self,fget):
      self.fget = fget
      self.func_name = fget.__name__

    def __get__(self,obj,cls):
      value = self.fget(obj)
      setattr(obj,self.func_name,value)
      return value



class AssimFailedError(RuntimeError):
    pass

def raise_AFE(msg,time_index=None):
  if time_index is not None:
    msg += "\n(k,kObs,fau) = " + str(time_index) + ". "
  raise AssimFailedError(msg)


def vectorize0(f):
  """Vectorize f for its 1st (index 0) argument, and do so recursively.

  Compared to np.vectorize:

    - less powerful, but safer to only vectorize 1st argument
    - doesn't evaluate the 1st item twice
    - doesn't always return array

  Still, it's quite messy, and should not be used in "production code".

  Example::

    >>> @vectorize0
    >>> def add(x,y):
    >>>   return x+y
    >>> add(1,100)
    101
    >>> x = np.arange(6).reshape((3,-1))
    >>> add(x,100)
    array([[100, 101],
         [102, 103],
         [104, 105]])
    >>> add([20,x],100)
    [120, array([[100, 101],
          [102, 103],
          [104, 105]])]
  """
  @functools.wraps(f)
  def wrapped(x,*args,**kwargs):
    if hasattr(x,'__iter__'):
      out = [wrapped(xi,*args,**kwargs) for xi in x]
      if isinstance(x,np.ndarray):
        out = np.asarray(out)
    else:
      out = f(x,*args,**kwargs)
    return out
  return wrapped


