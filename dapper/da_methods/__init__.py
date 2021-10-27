"""Contains the data assimilation methods included with DAPPER.

.. include:: ./README.md
"""
import dataclasses
import functools
import inspect
import sys
import time
import traceback
from dataclasses import dataclass

import dapper.stats


def da_method(*default_dataclasses):
    """Turn a dataclass-style class into a DA method for DAPPER (`xp`).

    This decorator applies to classes that define DA methods.
    An instances of the resulting class is referred to (in DAPPER)
    as an `xp` (short for experiment).

    The decorated classes are defined like a `dataclass`,
    but are decorated by `@da_method()` instead of `@dataclass`.

    .. note::
        The classes must define a method called `assimilate`.
        This method gets slightly enhanced by this wrapper which provides:

        - Initialisation of the `Stats` object, accessible by `self.stats`.
        - `fail_gently` functionality.
        - Duration timing
        - Progressbar naming magic.

    Example:
    >>> @da_method()
    ... class Sleeper():
    ...     "Do nothing."
    ...     seconds : int  = 10
    ...     success : bool = True
    ...     def assimilate(self, *args, **kwargs):
    ...         for k in range(self.seconds):
    ...             time.sleep(1)
    ...         if not self.success:
    ...             raise RuntimeError("Sleep over. Failing as intended.")

    Internally, `da_method` is just like `dataclass`,
    except that adds an outer layer
    (hence the empty parantheses in the above)
    which enables defining default parameters which can be inherited,
    similar to subclassing.

    Example:
    >>> class ens_defaults:
    ...     infl : float = 1.0
    ...     rot  : bool  = False

    >>> @da_method(ens_defaults)
    ... class EnKF:
    ...     N     : int
    ...     upd_a : str = "Sqrt"
    ...
    ...     def assimilate(self, HMM, xx, yy):
    ...         ...

    .. hint::
        Apart from what's listed in the above `Note`, there is nothing special to the
        resulting `xp`.  That is, just like any Python object, it can serve as a data
        container, and you can write any number of attributes to it (at creation-time,
        or later).  For example, you can set attributes that are not used by the
        `assimilate` method, but are instead used to customize other aspects of the
        experiments (see `dapper.xp_launch.run_experiment`).
    """

    def dataclass_with_defaults(cls):
        """Like `dataclass`, but add some DAPPER-specific things.

        This adds `__init__`, `__repr__`, `__eq__`, ...,
        but also includes inherited defaults,
        ref https://stackoverflow.com/a/58130805,
        and enhances the `assimilate` method.
        """

        def set_field(name, type_, val):
            """Set the inherited (i.e. default, i.e. has value) field."""
            # Ensure annotations
            cls.__annotations__ = getattr(cls, '__annotations__', {})
            # Set annotation
            cls.__annotations__[name] = type_
            # Set value
            setattr(cls, name, val)

        # APPend default fields without overwriting.
        # NB: Don't implement (by PREpending?) non-default args -- to messy!
        for default_params in default_dataclasses:
            # NB: Calling dataclass twice always makes repr=True
            for field in dataclasses.fields(dataclass(default_params)):
                if field.name not in cls.__annotations__:
                    set_field(field.name, field.type, field)

        # Create new class (NB: old/new classes have same id)
        cls = dataclass(cls)

        # Define the new assimilate method (has bells and whistles)
        def assimilate(self, HMM, xx, yy, desc=None, fail_gently=False, **stat_kwargs):
            # Progressbar name
            pb_name_hook = self.da_method if desc is None else desc # noqa

            # Init stats
            self.stats = dapper.stats.Stats(self, HMM, xx, yy, **stat_kwargs)

            # Assimilate
            time0 = time.time()
            try:
                _assimilate(self, HMM, xx, yy)
            except Exception as ERR:
                if fail_gently:
                    self.crashed = True
                    if fail_gently not in ["silent", "quiet"]:
                        _print_cropped_traceback(ERR)
                else:
                    # Don't use _print_cropped_traceback here -- It would
                    # crop out errors in the DAPPER infrastructure itself.
                    raise
            self.stat("duration", time.time()-time0)

        # Overwrite the assimilate method with the new one
        try:
            _assimilate = cls.assimilate
        except AttributeError as error:
            raise AttributeError(
                "Classes decorated by da_method()"
                " must define a method called 'assimilate'.") from error
        cls.assimilate = functools.wraps(_assimilate)(assimilate)

        # Shortcut for register_stat
        def stat(self, name, value):
            dapper.stats.register_stat(self.stats, name, value)
        cls.stat = stat

        # Make self.__class__.__name__ an attrib.
        # Used by xpList.table_prep().
        cls.da_method = cls.__name__

        return cls
    return dataclass_with_defaults


def _print_cropped_traceback(ERR):

    # A more "standard" (robust) way:
    # https://stackoverflow.com/a/32999522
    def crop_traceback(ERR):
        msg = "Traceback (most recent call last):\n"
        try:
            # If in IPython, use its coloring functionality
            __IPYTHON__  # type: ignore
        except (NameError, ImportError):
            msg += "".join(traceback.format_tb(ERR.__traceback__))
        else:
            from IPython.core.debugger import Pdb
            pdb_instance = Pdb()
            pdb_instance.curframe = inspect.currentframe()

            import dapper.da_methods
            keep = False
            for frame in traceback.walk_tb(ERR.__traceback__):
                if keep:
                    msg += pdb_instance.format_stack_entry(frame, context=3)
                elif frame[0].f_code.co_filename == dapper.da_methods.__file__:
                    keep = True
                    msg += "   ⋮ [cropped] \n"

        return msg

    msg = crop_traceback(ERR) + "\nError message: " + str(ERR)
    msg += ("\n\nResuming execution."
            "\nIf instead you wish to raise the exceptions as usual,"
            "\nwhich will halt the execution (and enable post-mortem debug),"
            "\nthen use `fail_gently=False`")
    print(msg, file=sys.stderr)


from .baseline import Climatology, OptInterp, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
