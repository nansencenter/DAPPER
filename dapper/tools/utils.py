"""Utilities (misc, non-math)."""

import contextlib
import inspect
import re
import sys
import time
import traceback as tb

from struct_tools import NicePrint


def print_cropped_traceback(ERR):

    def crop_traceback(ERR, lvl):
        msg = "Traceback (most recent call last):\n"
        try:
            # If in IPython, use its coloring functionality
            __IPYTHON__  # noqa
            from IPython.core.debugger import Pdb
            pdb_instance = Pdb()
            pdb_instance.curframe = inspect.currentframe()

            for i, frame in enumerate(tb.walk_tb(ERR.__traceback__)):
                if i < lvl:
                    continue  # skip frames
                if i == lvl:
                    msg += "   ⋮ [cropped] \n"
                msg += pdb_instance.format_stack_entry(frame, context=3)

        except (NameError, ImportError):
            msg += "".join(tb.format_tb(ERR.__traceback__))

        return msg

    msg = crop_traceback(ERR, 1) + "\nError message: " + str(ERR)
    msg += "\n\nResuming execution. " \
        "Use `fail_gently=False` to raise exception & halt execution.\n"
    print(msg, file=sys.stderr)


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


@contextlib.contextmanager
def set_tmp(obj, attr, val):
    """Temporarily set an attribute.

    Example:
    >>> class A:
    >>>     pass
    >>> a = A()
    >>> a.x = 1  # Try deleting this line
    >>> with set_tmp(a,"x","TEMPVAL"):
    >>>     print(a.x)
    >>> print(a.x)

    Based on
    http://code.activestate.com/recipes/577089/
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

    try:
        yield  # was_there, tmp
    except BaseException:
        raise
    finally:
        if not was_there:
            delattr(obj, attr)
        else:
            setattr(obj, attr, tmp)


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
        # pass # Turn off timer messages
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))


def find_1st_ind(xx):
    try:
        return next(k for k, x in enumerate(xx) if x)
    except StopIteration:
        return None


# https://stackoverflow.com/a/2669120
def sorted_human(lst):
    """Sort the given iterable in the way that humans expect."""
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key = alphanum_key)


def collapse_str(string, length=6):
    """Abbreviate string to ``length``"""
    if len(string) <= length:
        return string
    else:
        return string[:length-2]+'~'+string[-1]


def all_but_1_is_None(*args):
    "Check if only 1 of the items in list are Truthy"
    return sum(x is not None for x in args) == 1


def do_once(fun):
    def new(*args, **kwargs):
        if new.already_done:
            return None  # do nothing
        else:
            new.already_done = True
            return fun(*args, **kwargs)
    new.already_done = False
    return new


def monitor_setitem(cls):
    """Modify cls to track of whether its ``__setitem__`` has been called.

    See sub.py for a sublcass solution (drawback: creates a new class)."""

    orig_setitem = cls.__setitem__

    def setitem(self, key, val):
        orig_setitem(self, key, val)
        self.were_changed = True
    cls.__setitem__ = setitem

    # Using class var for were_changed => don't need explicit init
    cls.were_changed = False

    if issubclass(cls, NicePrint):
        cls.printopts['excluded'] = \
                cls.printopts.get('excluded', []) + ['were_changed']

    return cls
