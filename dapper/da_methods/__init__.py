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

import functools
import time
import abc
from dataclasses import MISSING

import dapper.stats


class da_method(abc.ABC):
    """Base class for da method.
    The class is motivated by dataclasses.

    Specific DA method classes must be defined like a `dataclass`.
    They must also define a method called `assimilate`
    which gets slightly enhanced by `_assimilate` wrapper which provides:

    - Initialisation of the `Stats` object
    - `fail_gently` functionality.
    - Duration timing
    - Progressbar naming magic.

    Instances of these classes are what is referred to as `xp`s.
    I.e. `xp`s are essentially just a `dataclass` with some particular attributes.

    Examples
    --------
    >>> class ens_method(da_method):
    ...     infl: float        = 1.0
    ...     rot: bool          = False
    ...     fnoise_treatm: str = 'Stoch'

    >>> class EnKF(ens_method):
    ...     upd_a: str
    ...     N: int
    ...
    ...     def assimilate(self, HMM, xx, yy):
    ...         print("Running assimilation")
    """
    no_repr = ["assimilate", "no_repr"]

    def __init_subclass__(cls):
        # name of da_method must be a class attribute
        cls.da_method = cls.__name__

    def __init__(self, *args, **kwargs):
        cls = self.__class__

        # Get class fields
        _fields = cls.Get_fields()
        i_arg = 0
        # initialize attributes
        for key in _fields:
            try:
                # account for positional arguments
                setattr(self, key, args[i_arg])
                i_arg += 1
            except IndexError:
                # then account for keywords arguments
                try:
                    setattr(self, key, kwargs[key])
                except KeyError:
                    # using default values
                    value = getattr(cls, key, MISSING)
                    if value is not MISSING:
                        setattr(self, key, value)
                    else:
                        # if attribute value is not given
                        # raise TypeError
                        raise TypeError(f"{key} must have a value")

        # Use _assimilate to wrap the abstract class assimilate
        self._assimilator = self.assimilate
        self.assimilate = self._assimilate
        # functools wrap cannot be used in class methods
        # because docstrings of methods cannot be changed.
        self.assimilate.__func__.__doc__ = self._assimilator.__doc__
        self.assimilate.__func__.__name__ = self._assimilator.__name__
        self.assimilate.__func__.__module__ = self._assimilator.__module__

    def _assimilate(self, HMM, xx, yy, desc=None, **stat_kwargs):
        """Wraps assimilate method"""
        # Progressbar name
        pb_name_hook = self.__class__.__name__ if desc is None else desc # noqa
        # Init stats
        self.stats = dapper.stats.Stats(self, HMM, xx, yy, **stat_kwargs)
        # Assimilate
        time_start = time.time()
        self._assimilator(HMM, xx, yy)
        dapper.stats.register_stat(self.stats, "duration", time.time()-time_start)

    @abc.abstractmethod
    def assimilate(HMM, xx, yy):
        """Abstract DA method. If you see this, a
        dosctring for your method is recommended.
        """
        pass

    @classmethod
    def Get_fields(cls):
        """Get all keys of annotations"""
        _fields = []
        # number of base classes of da_methods
        n = len(__class__.__mro__)
        for b in cls.__mro__[:-n]:
            # Same as dataclass, attributes are from
            # class annotations. Default values
            # are set for class attributes
            # See Also:
            # https://github.com/python/cpython/blob/b4796875d598b34f5f21cb13a8d3551574532595/Lib/dataclasses.py#L849
            # Some classes do not have any __annotations__ and should be ignored
            if not hasattr(b, "__annotations__"):
                continue
            for key in b.__annotations__.keys():
                _fields.append(key)
        return _fields

    def __repr__(self):
        txt = ', '.join([f"{field}= {getattr(self, field, MISSING)}"
                        for field in self.Get_fields()])
        return self.__class__.__qualname__ + f"({txt})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        _field = self.Get_fields()
        values_self = [getattr(self, field, MISSING) for
                       field in _field]
        values_other = [getattr(other, field, MISSING) for
                        field in _field]
        return values_other == values_self


from .baseline import Climatology, OptInterp, Var3D
from .ensemble import LETKF, SL_EAKF, EnKF, EnKF_N, EnKS, EnRTS
from .extended import ExtKF, ExtRTS
from .other import LNETF, RHF
from .particle import OptPF, PartFilt, PFa, PFxN, PFxN_EnKF
from .variational import Var4D, iEnKS
