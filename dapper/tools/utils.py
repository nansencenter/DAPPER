"""Utilities."""

import inspect
import sys
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

def collapse_str(string, length=6):
    """Abbreviate string to ``length``"""
    if len(string) <= length:
        return string
    else:
        return string[:length-2]+'~'+string[-1]


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
