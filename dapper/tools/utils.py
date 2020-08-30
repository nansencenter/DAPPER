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
        try:
            locals = stack[level].frame.f_locals
        except IndexError:
            pass
        else:
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

def rel2mods(path):
    path = Path(path).relative_to(rc.dirs.dapper/'mods').with_suffix("")
    return str(path)


#########################################
# Console input / output
#########################################

def print_cropped_traceback(ERR):

    def crop_traceback(ERR,lvl):
        msg = "Traceback (most recent call last):\n"
        try:
            # If in IPython, use its coloring functionality
            __IPYTHON__
            from IPython.core.debugger import Pdb
            import traceback as tb
            pdb_instance = Pdb()
            pdb_instance.curframe = inspect.currentframe()

            for i, frame in enumerate(tb.walk_tb(ERR.__traceback__)):
                if i<lvl: continue # skip frames
                if i==lvl: msg += "   ⋮ [cropped] \n"
                msg += pdb_instance.format_stack_entry(frame,context=3)

        except (NameError,ImportError):
            msg += "".join(traceback.format_tb(ERR.__traceback__))

        return msg

    msg = crop_traceback(ERR,1) + "\nError message: " + str(ERR)
    msg += "\n\nResuming program execution. " \
        "Use `fail_gently=False` to raise exception & halt execution.\n"
    print(msg,file=sys.stderr)


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


# TODO: doesnt work anymore?
def magic_naming(*args,**kwargs):
    """Convert args (by their names in the call) to kwargs."""
    call, locvars = get_call()

    # Use a new dict, with args inserted first, to keep ordering.
    joint_kwargs = {}

    # Insert args in kwargs
    for i,arg in enumerate(args):
        # Match arg to a name by
        # - id to a variable in the local namespace, and
        mm = [name for name in locvars if locvars[name] is arg]
        # - the presence of said variable in the call.
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


# TODO: rm?
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


# From https://stackoverflow.com/q/22797580 and more
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

# TODO: use finally?
import contextlib
@contextlib.contextmanager
def set_tmp(obj, attr, val):
    """Temporarily set an attribute.

    http://code.activestate.com/recipes/577089/"""

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

    if issubclass(cls,NicePrint):
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
