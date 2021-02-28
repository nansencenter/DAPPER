"""Contains the data assimilation methods included with DAPPER.

See the README section on
[DA Methods](https://github.com/nansencenter/DAPPER#DA-Methods)
for an overview of the methods included with DAPPER.

## Defining your own method

Follow the example of one of the methods within one of the
sub-directories/packages.
The simplest example is perhaps
`dapper.da_methods.ensemble.EnKF`.


## General advice for programming/debugging scientific experiments

- Start with something simple.
  This helps make sure the basics of the experiment are reasonable.
  For example, start with

      - a pre-existing example,
      - something you are able to reproduce,
      - a small/simple model.

        - Set the observation error to be small.
        - Observe everything.
        - Don't include model error and/or noise to begin with.

- Additionally, test a simple/baseline method to begin with.
  When including an ensemble method, start with using a large ensemble,
  and introduce localisation later.

- Take incremental steps towards your ultimate experiment setup.
  Validate each incremental setup with prints/plots.
  If results change, make sure you understand why.

- Use short experiment duration.
  You probably don't need statistical significance while debugging.
"""
import dataclasses
import functools
import time
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

        # The new assimilate method
        def assimilate(self, HMM, xx, yy, desc=None, **stat_kwargs):
            # Progressbar name
            pb_name_hook = self.da_method if desc is None else desc # noqa
            # Init stats
            self.stats = dapper.stats.Stats(self, HMM, xx, yy, **stat_kwargs)
            # Assimilate
            time_start = time.time()
            _assimilate(self, HMM, xx, yy)
            dapper.stats.register_stat(
                self.stats, "duration", time.time()-time_start)

        # Overwrite the assimilate method with the new one
        try:
            _assimilate = cls.assimilate
        except AttributeError as error:
            raise AttributeError(
                "Classes decorated by da_method()"
                " must define a method called 'assimilate'.") from error
        cls.assimilate = functools.wraps(_assimilate)(assimilate)

        # Make self.__class__.__name__ an attrib.
        # Used by xpList.split_attrs().
        cls.da_method = cls.__name__

        return cls
    return dataclass_with_defaults


from .baseline import Climatology, OptInterp, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
