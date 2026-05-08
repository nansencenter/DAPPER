"""Contains models included with DAPPER.

--8<-- "dapper/mods/README.md"
"""

import copy as cp
import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import struct_tools

# Imports used to set up HMMs
import dapper.tools.progressbar as pb
from dapper.dpr_config import rc
from dapper.mods.utils import Id_op
from dapper.tools.chronos import Chronology
from dapper.tools.localization import no_localization
from dapper.tools.matrices import CovMat
from dapper.tools.randvars import RV, GaussRV
from dapper.tools.seeding import set_seed

from .integration import with_recursion, with_rk4
from .utils import Id_Obs, ens_compatible, linspace_int, partial_Id_Obs


class HiddenMarkovModel(struct_tools.NicePrint):
    """Container class (with some embellishments) for a Hidden Markov Model (HMM).

    Should contain the details necessary to run synthetic DA experiments,
    also known as "twin experiment", or OSSE (observing system simulation experiment).
    The synthetic truth and observations may then be obtained by running
    [`.simulate`][mods.HiddenMarkovModel.simulate].

    !!! note
        Each model included with DAPPER comes with several examples
        of model settings from the literature.
        See, for example, [`mods.Lorenz63.sakov2012`][].

    !!! warning
        These example configs do not necessarily hold a high programming standard,
        as they may have been whipped up at short notice to replicate some experiments,
        and are not intended for re-use.
        Nevertheless, sometimes they are re-used by another configuration script,
        leading to a major gotcha/pitfall: changes made to the imported `HMM` (or
        the model's module itself) also impact the original object (since they
        are mutable and thereby referenced).  This *usually* isn't an issue, since
        one rarely imports two/more separate configurations. However, the test suite
        imports all configurations, which might then unintentionally interact.
        To avoid this, you should use the `copy` method of the `HMM`
        before making any changes to it.
    """

    def __init__(
        self,
        Dyn,
        Obs,
        tseq,
        X0,
        liveplotters=None,
        sectors=None,
        name=None,
    ):
        """Initialize.

        Parameters
        ----------
        Dyn : Operator or dict
            Operator for the dynamics.
        Obs : Operator or callable(ko) -> Operator
            Operator for the observations.
            For time-dependent observations, pass a callable ``Obs(ko) -> Operator``.
        tseq : tools.chronos.Chronology
            Time sequence of the HMM process.
        X0 : tools.randvars.RV
            Random distribution of initial condition
        liveplotters : list, optional
            A list of tuples. See example use in function `LPs` of [`mods.Lorenz63`][].
            - The first element of the tuple determines if the liveplotter
            is shown by default. If `False`, the liveplotter is only shown when
            included among the `liveplots` argument of `assimilate`
            - The second element in the tuple gives the corresponding liveplotter
            function/class.
        sectors : dict, optional
            Labelled indices referring to parts of the state vector.
            When defined, field-mean statistics are computed for each sector.
            Example use can be found in  `docs/examples/param_estim.py`
            and `dapper/mods/Lorenz96/miyoshi2011.py`
        name : str, optional
            Label for the `HMM`.
        """
        self.Dyn = Dyn if isinstance(Dyn, Operator) else Operator(**Dyn)
        self.Obs = (
            Obs
            if isinstance(Obs, TimeDependentOperator)
            else TimeDependentOperator(Obs)
        )
        self.tseq = tseq
        self.X0 = X0
        self.liveplotters = liveplotters or []
        self.sectors = sectors or {}
        self.name = name or self._default_name()

        # if not hasattr(self.Obs, "localizer"):
        #     self.Obs.localizer = no_localization(self.Nx, self.Ny)

        # if self.Obs.noise.C == 0 or self.Obs.noise.C.rk != self.Obs.noise.C.M:
        #     raise ValueError("Rank-deficient R not supported.")

    # ndim shortcuts
    @property
    def Nx(self):
        return self.Dyn.M

    printopts = {"ordering": ["Dyn", "Obs", "tseq", "X0"], "indent": 4}

    def simulate(self, desc="Truth & Obs"):
        """Generate synthetic truth and observations."""
        Dyn, Obs, tseq, X0 = self.Dyn, self.Obs, self.tseq, self.X0

        # Init
        xx = np.zeros((tseq.K + 1, Dyn.M))
        yy = np.empty(tseq.Ko + 1, dtype=object)

        x = X0.sample(1).squeeze()
        xx[0] = x

        # Loop
        for k, ko, t, dt in pb.progbar(tseq.ticker, desc):
            x = Dyn(x, t - dt, dt)
            x = x + np.sqrt(dt) * Dyn.noise.sample(1).squeeze()
            if ko is not None:
                yy[ko] = Obs(ko)(x) + Obs(ko).noise.sample(1).squeeze()
            xx[k] = x

        return xx, yy

    def copy(self):
        return cp.deepcopy(self)

    @staticmethod
    def _default_name():
        name = inspect.getfile(inspect.stack()[2][0])
        try:
            name = str(Path(name).relative_to(rc.dirs.dapper / "mods"))
        except ValueError:
            name = str(Path(name))
        return name


class TimeDependentOperator:
    """Callable wrapper for ``HMM.Obs`` enablign time-dependent obs. operators.

    The call argument ``ko`` is the observation index (not wall time).
    The return value is always an `Operator`.

    Examples: `docs/examples/time-dep-obs-operator.py`
    and `dapper/mods/QG/sakov2008.py`.
    """

    def __init__(self, op_or_func):
        """Initialize from a constant `Operator` or a callable ``ko -> Operator``.

        When given an `Operator`, it is returned unchanged for any ``ko``
        (constant-in-time case).
        When given a callable, it is called with ``ko`` on each access.
        """
        if isinstance(op_or_func, Operator):
            self._op = op_or_func
            self._func = None
        else:
            self._op = None
            self._func = op_or_func

    def __call__(self, ko: int) -> "Operator":
        return self._op if self._op is not None else self._func(ko)

    def __repr__(self):
        return repr(self(0))


class Operator(struct_tools.NicePrint):
    """Container for the dynamical and the observational maps.

    Parameters
    ----------
    M : int
        Length of output vectors.
    model : function
        The actual operator.
    noise : RV, optional
        The associated additive noise. The noise can also be a scalar or an
        array, producing `GaussRV(C=noise)`.

    Note
    ----
    Any remaining keyword arguments are written to the object as attributes.
    """

    def __init__(
        self,
        M: int,
        model: Callable | None = None,
        noise: RV | float | np.ndarray | None = None,
        linear: Callable | None = None,
        localizer: Callable | None = None,
        **kwargs,
    ):
        self.M = M

        # Default to the Identity operator
        if model is None:
            model = Id_op()
            if linear is None:

                def linear(*args):
                    return np.eye(M)

        self.model = model
        self.linear = linear
        self.localizer = localizer

        # None/0 => No noise
        if isinstance(noise, RV):
            self.noise = noise
        else:
            if noise is None:
                noise = 0
            if np.isscalar(noise):
                self.noise = GaussRV(C=noise, M=M)
            else:
                self.noise = GaussRV(C=noise)

        # Write remaining unknown attributes
        if kwargs:
            import warnings

            warnings.warn(
                f"Unknown Operator kwargs: {list(kwargs)}",
                DeprecationWarning,
                stacklevel=2,
            )
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    printopts = {"ordering": ["M", "model", "noise"], "indent": 4}
