# Utilities (non-math)

from dapper import *
from pathlib import Path
import time
import traceback as tb
import os
import re
import sys


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
    import termios 

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
# Console input / output
#########################################

def print_cropped_traceback(ERR):

    def crop_traceback(ERR,lvl):
        msg = "Traceback (most recent call last):\n"
        try:
            # If in IPython, use its coloring functionality
            __IPYTHON__
            from IPython.core.debugger import Pdb
            pdb_instance = Pdb()
            pdb_instance.curframe = inspect.currentframe()

            for i, frame in enumerate(tb.walk_tb(ERR.__traceback__)):
                if i<lvl: continue # skip frames
                if i==lvl: msg += "   ⋮ [cropped] \n"
                msg += pdb_instance.format_stack_entry(frame,context=3)

        except (NameError,ImportError):
            msg += "".join(tb.format_tb(ERR.__traceback__))

        return msg

    msg = crop_traceback(ERR,1) + "\nError message: " + str(ERR)
    msg += "\n\nResuming program execution. " \
        "Use `fail_gently=False` to raise exception & halt execution.\n"
    print(msg,file=sys.stderr)

import tabulate
tabulate.MIN_PADDING = 0
from tabulate import tabulate

def repr_type_and_name(thing):
    """Print thing's type [and name]"""
    s = "<" + type(thing).__name__ + '>'
    if hasattr(thing,'name'):
        s += ': ' + thing.name
    return s


# https://stackoverflow.com/q/22797580
# https://stackoverflow.com/q/10875442
import functools
class NamedFunc():
    "Provides custom repr for functions."
    def __init__(self,func,name):
        self._function = func
        self._old_name = func.__name__
        self._new_name = name
        functools.update_wrapper(self, func)
    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
    def __str__(self):
        return self._new_name + "()"
    def __repr__(self):
        return str(self) + f" <NamedFunc of {self._old_name}>" 

def name_func(name):
    """Decorator for creating NamedFunc."""
    def namer(func):
        return NamedFunc(func,name)
    return namer

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

def rel2mods(path):
    path = Path(path).relative_to(rc.dirs.dapper/'mods').with_suffix("")
    return str(path)

# # Temporarily set attribute values
# @contextlib.contextmanager
# def set_tmp(obj,attr,val):
#     tmp = getattr(obj,attr)
#     setattr(obj,attr,val)
#     yield 
#     setattr(obj,attr,tmp)

# TODO 2: use finally?
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
class Timer():
    """Timer context manager.

    Example::

    >>> with Timer('<description>'):
    >>>     time.sleep(1.23)
    [<description>] Elapsed: 1.23
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        #pass # Turn off timer messages
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))

def find_1st_ind(xx):
    try:                  return next(k for k,x in enumerate(xx) if x)
    except StopIteration: return None

# https://stackoverflow.com/a/2669120
def sorted_human( lst ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(lst, key = alphanum_key)

def unique_but_keep_order(arr):
    """Like ``set(arr)`` but keep the ordering."""
    return list(dict.fromkeys(arr).keys()) # Python >=3.7

def collapse_str(string,length=6):
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

    From https://stackoverflow.com/q/3012421
    """
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
