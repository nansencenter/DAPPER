# Utilities (non-math)

from dapper import *


#########################################
# Progressbar
#########################################

import inspect
def pdesc(desc):
    "Get progbar description by introspection."

    if desc is not None:
        return desc

    stack = inspect.stack()
    # Gives FULL REPR:
    # stack[4].frame.f_locals['name_hook']

    # Go look above in the stack for a name_hook.
    for level in range(2,6):
        locals = stack[level].frame.f_locals
        if 'pb_name_hook' in locals:
            name = locals['pb_name_hook']
            break
    else:
        # Otherwise: just get name of what's
        # calling progbar (i.e. stack[2]) 
        name = stack[2].function

    return name

def progbar(iterable, desc=None, leave=1, **kwargs):
    "Prints a nice progress bar in the terminal"
    if disable_progbar:
        return iterable
    else:
        desc = pdesc(desc)
        # if is_notebook:
        # This fails in QtConsole (which also yields is_notebook==True)
        # return tqdm.tqdm_notebook(iterable,desc=desc,leave=leave)
        # else:
        return tqdm.tqdm(iterable,desc=desc,leave=leave,
                smoothing=0.3,dynamic_ncols=True,**kwargs)
        # Printing during the progbar loop (may occur with error printing)
        # can cause tqdm to freeze the entire execution. 
        # Seemingly, this is caused by their multiprocessing-safe stuff.
        # Disable this, as per github.com/tqdm/tqdm/issues/461#issuecomment-334343230
        # pb = tqdm.tqdm(...)
        # try: pb.get_lock().locks = []
        # except AttributeError: pass
        # return pb

# NB: Also see disable_user_interaction
try:
    import tqdm
    disable_progbar = False
except ImportError:
    disable_progbar = True


#########################################
# Make read1()
#########################################
# Non-blocking, non-echo read1 from stdin.

# Set to True when stdin or term settings isn't supported, for example when:
#  - running via py.test
#  - multiprocessing
# Btw, multiprocessing also doesn't like tqdm itself.
disable_user_interaction = False

try:
    # Linux. See Misc/read1_trials.py
    import termios, sys

    def set_term_settings(TS):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TS)

    def new_term_settings():
        """Make stdin.read non-echo and non-block"""
        # NB: This makes ipython quit (by pressing C^D twice) hang!
        #     So it should be undone before returning to prompt.
        #     Thus, setting/restoring at import/atexit is not viable.

        TS_old = termios.tcgetattr(sys.stdin)
        TS_new = termios.tcgetattr(sys.stdin)

        # Make tty non-echo.
        TS_new[3] = TS_new[3] & ~(termios.ECHO | termios.ICANON)

        # Make tty non-blocking.
        TS_new[6][termios.VMIN] = 0
        TS_new[6][termios.VTIME] = 0

        set_term_settings(TS_new)
        return TS_old

    try:
        # Test set/restore settings 
        TS_old = new_term_settings()
        set_term_settings(TS_old)

        # Wrap the progressbar generator so as to temporarily set term settings.
        # Alternative solution: set/restore term settings in assimilate()
        # of the da_method decorator. But that gets bloated, and anyways the logic
        # of user-control of liveplots kindof belongs together with a progressbar. 
        orig_progbar = progbar
        def progbar(iterable, desc=None, leave=1, **kwargs):
            if not disable_user_interaction:
                TS_old = new_term_settings()
            try:
                for i in orig_progbar(iterable, pdesc(desc), leave, **kwargs):
                    yield i
            except GeneratorExit:
                pass # Allows code below to run even if caller raised exception
                     # NB: Fails if caller is in a list-comprehesion! Why?
            # Restore both for normal termination or exception (propagates).
            if not disable_user_interaction:
                set_term_settings(TS_old)
  
        def _read1():
            return os.read(sys.stdin.fileno(), 1)

    except:
        # Fails in non-terminal environments
        disable_user_interaction = True

except ImportError:
    # Windows
    try:
        import msvcrt
        def _read1():
            if msvcrt.kbhit():
                return msvcrt.getch()
            else:
                return None
    except ImportError:
        disable_user_interaction = True

def read1():
    "Get 1 character. Non-blocking, non-echoing."
    if disable_user_interaction: return None
    return _read1()


#########################################
# Path manipulation
#########################################

import socket
def save_dir(script,host="",relpath=False):
    """Make dirs['data']/script_path/[hostname]."""
    script = os.path.splitext(script)[0]

    if relpath:
        # Safely determine relpath: script vs. dpr_data
        _a     = os.path.abspath
        root   = os.path.commonpath([_a(dirs['data']),_a(script)])
        script = os.path.relpath(script, root)
    else:
        # Short path
        script = os.path.basename(script)

    # Hostname
    host = socket.gethostname().split('.')[0] if host is True else host

    # Makedir
    sdir = os.path.join(dirs['data'],script,host,'')
    os.makedirs(sdir, exist_ok=True)

    return sdir

import glob
def get_filenums(glb):
    ls = glob.glob(glb+'*')
    return [int(re.search(glb+'([0-9]*).*',f).group(1)) for f in ls]

def run_path(script,host="",relpath=False,timestamp=True):
    """save_dir + run_number"""
    sdir = save_dir(script,host=host,relpath=relpath)
    sdir = sdir + "run_"

    if timestamp:
        from datetime import datetime
        now = datetime.now()
        run_number = now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_number = str(1 + max(get_filenums(sdir),default=0))

    sdir = sdir + run_number
    assert not os.path.exists(sdir)
    return sdir


def rel_path(path,start=None,ext=False):
    path = os.path.relpath(path,start)
    if not ext:
        path = os.path.splitext(path)[0]
    return path


#########################################
# Console input / output
#########################################

import inspect
def get_call():
    """Get calling statement (even if it is multi-lined).

    NB: returns full lines (may include junk before/after calls)
        coz real parsing (brackets, commas, backslash, etc) is complicated.

    Also return caller namespace.
    """
    f0         = inspect.currentframe()        # this frame
    f1         = f0.f_back                     # caller1
    name       = f1.f_code.co_name             # caller1's name
    f2         = f1.f_back                     # caller2's frame
    code,shift = inspect.getsourcelines(f2)    # caller2's code
    nEnd       = f2.f_lineno                   # caller2's lineno

    if shift: nEnd -= shift
    else: nEnd -= 1 # needed when shift==0 (don't know why)

    # Loop backwards from line nEnd
    for nStart in range(nEnd,-1,-1):
        line = code[nStart]
        if re.search(r"\b"+name+r"\b\s*\(",line): break
    else:
        raise Exception("Couldn't find caller.")

    call = "".join(code[nStart:nEnd+1])
    call = call.rstrip() # rm trailing newline 

    return call, f2.f_locals


def magic_naming(*args,**kwargs):
    """Convert args (by their names in the call) to kwargs."""
    call, locvars = get_call()

    # Use a new dict, with args inserted first, to keep ordering.
    joint_kwargs = {}

    # Insert args in kwargs
    for i,arg in enumerate(args):
        # Match arg to a name by
        # - id to a variable in the local namespace, and
        # - the presence of said variable in the call.
        mm = [name for name in locvars if locvars[name] is arg]
        mm = [name for name in mm if re.search(r"\b"+name+r"\b", call)]
        if not mm:
            raise RuntimeError("Couldn't find the name for "+str(arg))
        for m in mm: # Allows saving an arg under multiple names.
            joint_kwargs[m] = arg

    joint_kwargs.update(kwargs)
    return joint_kwargs


def spell_out(*args):
    """
    Print (args) including variable names.
    Example:
    >>> print(3*2)
    >>> 3*2:
    >>> 6
    """

    call, _ = get_call()

    # Find opening/closing brackets
    left  = call. find("(")
    right = call.rfind(")")

    # Print header
    with coloring(cFG.MAGENTA):
        print(call[left+1:right] + ":")
    # Print (normal)
    print(*args)


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
def tabulate(data,headers=(),formatters=(),inds='nice',**kwargs):
    """Pre-processor for tabulate().

    - Transposes 'data' (list-of-lists).
    - ``formatter``: define formats to apply before relaying to tabulate().
                     Default: attr.__name__ (when applicable).
    - If ``data`` is a dict, the ``headers`` will be keys.
    - Add row indices with style: [i]

    Example::

      >>> print(tabulate(cfgs.split_attrs()[0]))
    """

    # Extract dict
    if hasattr(data,'keys'):
        headers = list(data)
        data = data.values()

    # Default formats
    if not formatters:
        formatters = ({
            'test'  : lambda x: hasattr(x,'__name__'),
            'format': lambda x: x.__name__
        },{
            'test'  : lambda x: isinstance(x,np.ndarray),
            'format': lambda x: "arr(...)"
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

    return tabulate_orig.tabulate(data,headers,showindex=inds,**kwargs)


def repr_type_and_name(thing):
    """Print thing's type [and name]"""
    s = "<" + type(thing).__name__ + '>'
    if hasattr(thing,'name'):
        s += ': ' + thing.name
    return s


from IPython.lib.pretty import pretty as pretty_repr
class NestedPrint:
    """Multi-Line, Recursive repr (print) functionality.

    Define a dict called 'printopts' in your subclass
    to change print the settings.

     - inden'          : indentation per level
     - ch              : character to use for "spine" (e.g. '|' or ' ')
     - ordr_by_linenum : 0: alphabetically, 1: linenumbr, -1: reverse
     - threshold       : as for numpy
     - precision       : as for numpy
     - excluded        : attributes not to be printed (allows regex and callable)
     - included        : only print these attrs.
     - ordering        : ordering of attrs
     - aliases         : dict containing the aliases for attributes
    """
    printopts = {}

    printopts['indent']          = 3
    printopts['ch']              = '.'
    printopts['ordr_by_linenum'] = 0

    printopts['threshold'] = 10
    printopts['precision'] = None

    printopts['excluded'] = ['printopts']
    printopts['excluded'].append(re.compile('^_'))

    printopts['included'] = []
    printopts['ordering'] = []
    printopts['aliases']  = {}

    # Recursion monitoring.
    _stack={}

    def __repr__(self):

        # Merge defaults with requested printopts
        opts = {}
        for k, v in NestedPrint.printopts.items():
            if k in self.printopts:
                v2 = self.printopts[k]
                if   isinstance(v, dict): opts[k] = {**v, **v2}
                elif isinstance(v, list): opts[k] = v + v2
                else:                     opts[k] = v2
            else:                         opts[k] = v

        with printoptions(**{k:opts[k] for k in ['threshold','precision']}):

            # new line chars
            NL = '\n' + opts['ch'] + ' '*(opts['indent']-1)

            # Init
            if NestedPrint._stack=={}:
                is_top_level = True
                NestedPrint._stack[id(self)] = '<root>' 
            else:
                is_top_level = False

            # Use included or filter-out excluded
            keys = vars(self)
            if opts['included']:
                keys = intersect (keys, *opts['included'])
            else:
                keys = complement(keys, *opts['excluded'])

            # Aggregate (sub-)repr's from the attributes
            txts = {}
            for key in keys:
                val = getattr(self,key)

                # Get sub-repr (t).
                # Link to already-printed items and prevent infinite recursion!
                # NB: Dont test ``val in _stack`` -- it also accepts equality.
                if id(val) in NestedPrint._stack and isinstance(val,NestedPrint):
                    # Link
                    t = "**Same (id)** as %s"%NestedPrint._stack[id(val)]
                else:
                    # Recurse
                    NestedPrint._stack[id(val)] = NestedPrint._stack[id(self)]+'.'+key
                    t = pretty_repr(val)

                # Process t: activate multi-line printing
                if '\n' in t:
                    t = t.replace('\n',NL+' '*opts['indent'])      # other lines
                    t = NL+' '*opts['indent'] + t                  # first line

                t = NL + opts['aliases'].get(key,key) + ': ' + t   # Add key (name)
                txts[key] = t # Register

            def sortr(x):
                if x in opts['ordering']:
                    key = -1000 + opts['ordering'].index(x)
                else:
                    if opts['ordr_by_linenum']:
                        key = 100*opts['ordr_by_linenum']*txts[x].count('\n')
                        if key<=1: # one '\n' is always present, due to NL
                            key = opts['ordr_by_linenum']*len(txts[x])
                    elif opts['ordr_by_linenum'] is None:
                        key = 0 # no sort
                    else:
                        key = x.lower()
                        # Convert str to int (assuming ASCII) for comp with above cases
                        key = sum( ord(x)*128**i for i,x in enumerate(x[::-1]) )
                return key

            # Assemble string
            s = repr_type_and_name(self)
            for key in sorted(txts, key=sortr):
                s += txts[key]

            # Empty _stack when top-level printing finished
            if is_top_level:
                NestedPrint._stack = {}

            return s


# from pingfive.typepad.com/blog/2010/04
def deep_getattr(obj,name,*default):
    for n in name.split('.'): obj = getattr(obj,n,*default)
    return obj

# deep_setattr() has been removed because it's unclear
# what type of object to set for intermediate hierarchy levels.

def deep_hasattr(obj,name):
    try: deep_getattr(obj,name); return True
    except AttributeError:       return False


class AlignedDict(dict):
    """Provide aligned-printing for dict."""
    def __str__(self):
        A = 2 # len(repr(key)) - len(key) 
        L = max([len(s)+A for s in self.keys()], default=0)
        s = " " # Indentation
        for k in self.keys():
            s += repr(k).rjust(L)
            s += ": "
            s += repr(self[k])
            s +="\n " # Add  newline + indentation
        s = s[:-2] # Rm last newline + indentation
        return s
    def __repr__(self):
        s = str(self)                 # Repeat str
        s = s.replace("\n",",\n")     # Commas
        if self: s = "\n" + s + "\n"  # Surrounding newlines
        return "{" + s + "}"          # Clams


class Bunch(NestedPrint,dict):
    """A dict that also has attribute (dot) access.

    Benefit compared to a dict:

     - Verbosity of ``mycontainer.a`` vs. ``mycontainer['a']``.
     - Includes NestedPrint.

    Why not just use NestedPrint itself as a container?

    - Because it's convenient to also have item access.
    - The class name hints at the "container" purpose.

    As seen from its creation in ``__init__``,
    Bunch is not very hackey.
    Bunch is also quite robust.
    Source: stackoverflow.com/a/14620633
    Similar constructs are quite common, eg IPython/utils/ipstruct.py.
    """
    def __init__(self,*args,**kwargs):
        "Init like a normal dict."
        super(Bunch, self).__init__(*args,**kwargs) # Make a (normal) dict
        self.__dict__ = self                        # Assign it to self.__dict__


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


def functools_wraps(wrapped, lineno=1, *args, **kwargs):
    """Like functools.wraps(), but keeps lines[0:lineno] of orig. docstring."""
    doc = wrapped.__doc__.splitlines()[lineno:]

    def wrapper(orig):
        orig_header = orig.__doc__.splitlines()[:lineno]

        @functools.wraps(wrapped,*args,**kwargs)
        def new(*args2,**kwargs2):
            return orig(*args2,**kwargs2)

        new.__doc__ = "\n".join(orig_header + doc)
        return new

    return wrapper


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
    """Temporarily set an attribute.
    code.activestate.com/recipes/577089"""
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

def find_1st_ind(xx):
    try:                  return next(k for k,x in enumerate(xx) if x)
    except StopIteration: return None

# stackoverflow.com/a/2669120
def sorted_human( lst ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(lst, key = alphanum_key)

def unique_but_keep_order(arr):
    """Like ``set(arr)`` but keep the ordering."""
    return list(dict.fromkeys(arr).keys()) # Python >=3.7

# kwargs = {abbrevs.get(k,k):kwargs[k] for k in kwargs}
def de_abbreviate(abbrev_d, abbreviations):
    "Expand any abbreviations (list of 2-tuples) that occur in abbrev_d (dict)."
    for a,b in abbreviations:
        if a in abbrev_d:
            abbrev_d[b] = abbrev_d[a]
            del abbrev_d[a]

def collapse_str(string,length=5):
    """Abbreviate string to ``length``"""
    if len(string)<=length:
        return string
    else:
        return string[:length-2]+'~'+string[-1]

def flexcomp(x,*criteria):
    """Compare in various ways."""

    def _compare(x,y):
        # Callable (condition) compare
        try: return y(x)
        except TypeError: pass
        # Regex compare -- should be compiled on the outside
        try: return bool(y.search(x))
        except AttributeError:
            # Value compare
            return y==x

    return any(_compare(x,y) for y in criteria)

# For some reason, _filter (with the `including` switch) is difficult
# for the brain to parse. Use intersect and complement instead.
def      intersect(iterable,   *wanted,                  dict=False):
    return _filter(iterable,   *wanted, including=True,  dict=dict)
def     complement(iterable, *unwanted,                  dict=False):
    return _filter(iterable, *unwanted, including=False, dict=dict)

def _filter(iterable, *criteria, including=True, dict=False):
    "Only keep those elements of `ìterable`` that match any criteria."""
    # Switch: including/not
    if including: condition = lambda x:     flexcomp(x,*criteria)
    else:         condition = lambda x: not flexcomp(x,*criteria)
    # Switch: dict/list
    if dict: return {k:v for k,v in iterable.items() if condition(k)}
    else:    return [x   for x   in iterable         if condition(x)]


def transpose_lists(LL, enforce_rectangle=True, as_list=False):
    """Example:
    >>> LL = [[i*10 + j for j in range(4)] for i in range(3)]
    """
    if enforce_rectangle:
        assert all(len(LL[0])==len(row) for row in LL)

    new = zip(*LL) # transpose

    if as_list:
        # new = list(map(list, new))
        new = [list(row) for row in new]

    return new

# Incredibly, it is difficult to make this less verbose
# (the one-liner is unpredicable for non-rectangular cases)
def transpose_dicts(dict_of_dicts,enforce_rectangle=True):
    d = dict_of_dicts
    new = {}
    for i in d:

        if enforce_rectangle and new:
            assert cols == list(d[i])
        cols = list(d[i])

        for j in d[i]:
            if j not in new:
                new[j] = {}
            new[j][i] = d[i][j]

    return new



def all_but_1_is_None(*args):
    "Check if only 1 of the items in list are Truthy"
    return sum(x is not None for x in args) == 1

class lazy_property:
    """Lazy evaluation of property.

    Should represent non-mutable data,
    as it replaces itself.

    From stackoverflow.com/q/3012421"""
    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self,obj,cls):
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value


def do_once(fun):
    def new(*args,**kwargs):
        if new.already_done:
            return None # do nothing
        else:
            new.already_done = True
            return fun(*args,**kwargs)
    new.already_done = False
    return new


def monitor_setitem(cls):
    """Modify cls to track of whether its ``__setitem__`` has been called.

    See sub.py for a sublcass solution (drawback: creates a new class)."""

    orig_setitem = cls.__setitem__
    def setitem(self,key,val):
        orig_setitem(self,key,val)
        self.were_changed = True
    cls.__setitem__ = setitem

    # Using class var for were_changed => don't need explicit init
    cls.were_changed = False

    if issubclass(cls,NestedPrint):
        cls.printopts['excluded'] = \
                cls.printopts.get('excluded',[]) + ['were_changed']

    return cls


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
