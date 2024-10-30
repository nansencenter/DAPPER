"""Contains models included with DAPPER.

--8<-- "dapper/mods/README.md"
"""

import copy as cp
import inspect
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
        **kwargs,
    ):
        """Initialize.

        Parameters
        ----------
        Dyn : Operator or dict
            Operator for the dynamics.
        Obs : Operator or TimeDependentOperator or dict
            Operator for the observations
            Can also be time-dependent, ref `TimeDependentOperator`.
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
        # Expected args/kwargs, along with type and default.
        attrs = dict(
            Dyn=(Operator, None),
            Obs=(TimeDependentOperator, None),
            tseq=(Chronology, None),
            X0=(RV, None),
            liveplotters=(list, []),
            sectors=(dict, {}),
            name=(str, self._default_name()),
        )

        # Un-abbreviate
        abbrevs = {"LP": "liveplotters", "loc": "localizer"}
        for key in list(kwargs):
            try:
                full = abbrevs[key]
            except KeyError:
                pass
            else:
                assert full not in kwargs, "Could not sort out arguments."
                kwargs[full] = kwargs.pop(key)

        # Collect args, kwargs.
        for key, (type_, default) in attrs.items():
            val = locals()[key] or kwargs.get(key, default)
            # Convert dict to object
            if not isinstance(val, type_) and val is not None:
                val = type_(**val)
            kwargs[key] = val

        # Transfer kwargs to self
        for key in attrs:
            setattr(self, key, kwargs.pop(key))
        assert not kwargs, (
            f"Arguments {list(kwargs)} not recognized. "
            "If you want, you can still write them to the HMM, "
            "but this must be done after initialisation."
        )

        # Further defaults
        # if not hasattr(self.Obs, "localizer"):
        #     self.Obs.localizer = no_localization(self.Nx, self.Ny)

        # Validation
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
    """Wrapper for `Operator` that enables time dependence.

    The time instance should be specified by `ko`,
    i.e. the index of an observation time.

    Examples: `docs/examples/time-dep-obs-operator.py`
    and `dapper/mods/QG/sakov2008.py`.
    """

    def __init__(self, **kwargs):
        """Can be initialized like `Operator`, in which case the resulting
        object will always return the same `Operator` nomatter the input time.

        If initialized with 1 argument: `dict(time_dependent=func)`
        then `func` must return an `Operator` object.
        """
        try:
            fun = kwargs["time_dependent"]
            assert len(kwargs) == 1
            assert callable(fun)
            self.Ops = fun
        except KeyError:
            self.Op1 = Operator(**kwargs)

    def __repr__(self):
        return "<" + type(self).__name__ + "> " + str(self)

    def __str__(self):
        if hasattr(self, "Op1"):
            return "CONSTANT operator sepcified by .Op1:\n" + repr(self.Op1)
        else:
            return ".Ops: " + repr(self.Ops)

    def __call__(self, ko):
        try:
            return self.Ops(ko)
        except AttributeError:
            return self.Op1


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

    def __init__(self, M, model=None, noise=None, **kwargs):
        self.M = M

        # Default to the Identity operator
        if model is None:
            model = Id_op()
            kwargs["linear"] = lambda *args: np.eye(M)
        # Assign
        self.model = model

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

        # Write attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    printopts = {"ordering": ["M", "model", "noise"], "indent": 4}
