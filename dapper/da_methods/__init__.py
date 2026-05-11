"""Contains the data assimilation methods included with DAPPER.

--8<-- "dapper/da_methods/README.md"
"""

import functools
import time
from dataclasses import dataclass, field
from typing import dataclass_transform

from dapper.stats import Avrgs, Stats, register_stat


def _setup_da_method(cls: type) -> None:
    """Apply dataclass and enhance `assimilate()` so that it includes

    - timing
    - gentle failure handling
    - progress bar naming
    - stats (see `dapper.stats.Stats`) init.
    """
    dataclass(cls)

    _assimilate = cls.assimilate

    def assimilate(self, HMM, xx, yy, desc=None, fail_gently=False, **stat_kwargs):
        pb_name_hook = self.da_method if desc is None else desc  # noqa
        self.stats = Stats(self, HMM, xx, yy, **stat_kwargs)
        time0 = time.time()
        try:
            _assimilate(self, HMM, xx, yy)
        except Exception as ERR:
            if fail_gently:
                self.crashed = True
                if fail_gently not in ["silent", "quiet"]:
                    _print_cropped_traceback(ERR)
            else:
                raise
        self.stat("duration", time.time() - time0)

    cls.assimilate = functools.wraps(_assimilate)(assimilate)

    def stat(self, name, value):
        register_stat(self.stats, name, value)

    cls.stat = stat
    cls.da_method = cls.__name__


@dataclass_transform()
@dataclass
class da_method:
    """Base class for all DA methods (xp objects, short for "experiment").

    Inheriting from this class makes the subclass a dataclass
    (auto-generating `__init__`, `__repr__`, `__eq__`)
    and also endows it with the following

    - `.stats` (`Stats`): initialised when `assimilate()` is called.
    - `.avrgs` (`Avrgs`): populated by `stats.average_in_time()`.

    It also wraps subclass' (mandatory) `assimilate(self, HMM, xx, yy)
    so as to set up stats, times execution, and `fail_gently`.

    Example::

        class MyMethod(da_method):
            param: float = 1.0

            def assimilate(self, HMM, xx, yy):
                ...

    To share default parameters across methods, define a
    ``@dataclass(kw_only=True)`` base and inherit from both::

        @dataclass(kw_only=True)
        class ens_defaults:
            infl: float = 1.0
            rot:  bool  = False

        class EnKF(da_method, ens_defaults):
            N: int

            def assimilate(self, HMM, xx, yy): ...
    """

    stats: Stats = field(init=False, repr=False, default=None)
    avrgs: Avrgs = field(init=False, repr=False, default=None)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "assimilate" not in cls.__dict__:
            raise AttributeError(
                f"{cls.__name__!r} inherits from `da_method`"
                " but does not define assimilate()."
            )
        _setup_da_method(cls)


def _print_cropped_traceback(ERR):
    import inspect
    import sys
    import traceback

    # A more "standard" (robust) way:
    # https://stackoverflow.com/a/32999522
    def crop_traceback(ERR):
        msg = "Traceback (most recent call last):\n"
        try:
            # If in IPython, use its coloring functionality
            __IPYTHON__  # type: ignore # noqa: B018
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
    msg += (
        "\n\nResuming execution."
        "\nIf instead you wish to raise the exceptions as usual,"
        "\nwhich will halt the execution (and enable post-mortem debug),"
        "\nthen use `fail_gently=False`"
    )
    print(msg, file=sys.stderr)


# Import all da_methods
# for _mod in Path(__file__).parent.glob("*.py"):
#     if _mod != Path(__file__) and not _mod.stem.startswith("_"):
#         _mod = __import__(__package__ + "." + _mod.stem, fromlist=['*'])
#         del globals()[_mod.__name__.split(".")[-1]]  # rm module itself
#         globals().update({k: v for k, v in vars(_mod).items()
#                           if isinstance(v, type) and hasattr(v, "da_method")})

# The above does not allow for go-to-definition, so
from .baseline import Climatology, OptInterp, Persistence, PreProg, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
