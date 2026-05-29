"""Contains the data assimilation methods included with DAPPER.

--8<-- "dapper/da_methods/README.md"
"""

import time
from dataclasses import dataclass, field
from typing import dataclass_transform

from dapper.stats import Avrgs, Stats


@dataclass_transform()
@dataclass
class da_method:
    """Base class for all DA methods.

    Objects hereof are often called `xp`, short for "experiment".

    Inheriting from this class makes the subclass a dataclass
    (auto-generating `__init__`, `__repr__`, `__eq__`)
    and also endows it `.stats`, `.avrgs`, and an enhanced `.assimilate()`.

    Example::

        class MyMethod(da_method):
            param: float = 1.0

            def assimilate(self, HMM, xx, yy):
                ...

    To share default parameters across methods, define a
    `@dataclass(kw_only=True)` base and inherit from both::

        @dataclass(kw_only=True)
        class ens_defaults:
            infl: float = 1.0
            rot:  bool  = False

        class EnKF(da_method, ens_defaults):
            N: int

            def assimilate(self, HMM, xx, yy): ...
    """

    stats: Stats = field(init=False, repr=False, default=None)  # ty: ignore[invalid-assignment]
    """[`stats.Stats`][] object of time series recorded during `assimilate()`."""
    avrgs: Avrgs = field(init=False, repr=False, default=None)  # ty: ignore[invalid-assignment]
    """[`stats.Avrgs`][] object of time-averaged statistics.

    Populated by `stats.average_in_time()` from [`stats`][da_methods.da_method.stats].
    """

    def assimilate(
        self,
        HMM,
        xx,
        yy,
        desc: str | None = None,
        fail_gently: bool | str = False,
        **stat_kwargs,
    ) -> None:
        """Wraps subclasses `.assimilate()` method to add the extra parameters below,

        as well as initialise `self.stats` and measure wall-clock time.

        Parameters
        ----------
        HMM:
            The [`mods.HiddenMarkovModel`][] defining the twin experiment.
        xx:
            True states, shape `(K+1, Nx)`.
        yy:
            Observations, shape `(Ko+1, Ny)`.
        desc:
            Label for the progress bar. Defaults to the class name.
        fail_gently:
            If truthy, catch exceptions and print a cropped traceback instead
            of re-raising. Pass `"silent"` or `"quiet"` to suppress even that.
        **stat_kwargs:
            Forwarded to [`stats.Stats.__init__`][]
            (e.g. `liveplots=True`, `store_i=False`).
        """
        pb_name_hook = self.da_method if desc is None else desc  # noqa
        self.stats = Stats(self, HMM, xx, yy, **stat_kwargs)
        time0 = time.time()
        try:
            self._assimilate(HMM, xx, yy)
        except Exception as ERR:
            if fail_gently:
                self.crashed = True
                if fail_gently not in ["silent", "quiet"]:
                    _print_cropped_traceback(ERR)
            else:
                raise
        self.stats.register("duration", time.time() - time0)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "assimilate" not in cls.__dict__:
            raise AttributeError(
                f"{cls.__name__!r} inherits from `da_method`"
                " but does not define assimilate()."
            )
        # Apply dataclass and wire subclass `assimilate()` to the base-class wrapper.
        dataclass(cls)
        cls._assimilate = cls.__dict__["assimilate"]
        del cls.assimilate  # inherit da_method.assimilate (the wrapper) via MRO
        cls.da_method = cls.__name__


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
                    msg += pdb_instance.format_stack_entry(
                        frame,
                        context=3,  # ty: ignore[unknown-argument]
                    )
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


from .baseline import Climatology, OptInterp, Persistence, PreProg, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
