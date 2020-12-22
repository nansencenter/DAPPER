"""Contains the data assimilation methods included with DAPPER.

See the README section on
[DA Methods](https://github.com/nansencenter/DAPPER#DA-Methods)
for an overview of the methods included with DAPPER.

## Defining your own method

Follow the example of one of the methods within one of the
sub-directories/packages.
The simplest example is perhaps
`dapper.da_methods.ensemble.EnKF`.
"""

import dataclasses
import functools
import time
from dataclasses import dataclass

import dapper.stats


def da_method(*default_dataclasses):
    """Decorator (factory) for classes that define DA methods.

    These classes must be defined like a `dataclass`,
    except decorated by `@da_method()` instead of `@dataclass`.
    They must also define a method called `assimilate`
    which gets slightly enhanced by this wrapper which provides:

    - Initialisation of the `Stats` object
    - `fail_gently` functionality.
    - Duration timing
    - Progressbar naming magic.

    Instances of these classes are what is referred to as `xp`s.
    I.e. `xp`s are essentially just a `dataclass` with some particular attributes.

    Example:
    >>> @da_method()
    >>> class Sleeper():
    >>>     "Do nothing."
    >>>     seconds : int  = 10
    >>>     success : bool = True
    >>>     def assimilate(self,*args,**kwargs):
    >>>         for k in range(self.seconds):
    >>>             time.sleep(1)
    >>>         if not self.success:
    >>>             raise RuntimeError("Sleep over. Failing as intended.")

    Note that `da_method` is actually a "two-level decorator",
    which is why the empty parenthesis were used above.
    The outer level can be used to define defaults that are "inherited"
    by similar DA methods:

    Example:
    >>> @dataclass
    >>> class ens_defaults:
    >>>   infl : float = 1.0
    >>>   rot  : bool  = False
    >>>
    >>> @da_method(ens_defaults)
    >>> class EnKF:
    >>>     N     : int
    >>>     upd_a : str = "Sqrt"
    >>>
    >>>     def assimilate(self,HMM,xx,yy):
    >>>         ...
    >>>
    >>>
    >>> @da_method(ens_defaults)
    >>> class LETKF:
    >>>     ...
    """

    def dataclass_with_defaults(cls):
        """Decorator based on dataclass.

        This adds `__init__`, `__repr__`, `__eq__`, ...,
        but also includes inherited defaults,
        cf. https://stackoverflow.com/a/58130805,
        and enhances the `assimilate` method.
        """

        # Default fields invovle: (1) annotations and (2) attributes.
        def set_field(name, type, val):
            if not hasattr(cls, '__annotations__'):
                cls.__annotations__ = {}
            cls.__annotations__[name] = type
            if not isinstance(val, dataclasses.Field):
                val = dataclasses.field(default=val)
            setattr(cls, name, val)

        # APPend default fields without overwriting.
        # Don't implement (by PREpending?) non-default args -- to messy!
        for D in default_dataclasses:
            # NB: Calling dataclass twice always makes repr=True, so avoid this.
            for F in dataclasses.fields(dataclass(D)):
                if F.name not in cls.__annotations__:
                    set_field(F.name, F.type, F)

        # Create new class (NB: old/new classes have same id)
        cls = dataclass(cls)

        # Shortcut for self.__class__.__name__
        cls.da_method = cls.__name__

        def assimilate(self, HMM, xx, yy, desc=None, **stat_kwargs):
            # Progressbar name
            pb_name_hook = self.da_method if desc is None else desc # noqa
            # Init stats
            self.stats = dapper.stats.Stats(self, HMM, xx, yy, **stat_kwargs)
            # Assimilate
            time_start = time.time()
            _assimilate(self, HMM, xx, yy)
            dapper.stats.register_stat(self.stats, "duration", time.time()-time_start)

        _assimilate = cls.assimilate
        cls.assimilate = functools.wraps(_assimilate)(assimilate)

        return cls
    return dataclass_with_defaults


from .baseline import Climatology, OptInterp, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
