"""Models included with DAPPER.

.. include:: ./README.md
"""

from __future__ import annotations

import copy as cp
import inspect
from optparse import Option
from pathlib import Path
from typing import Optional, List, Dict, TYPE_CHECKING, Callable, Union, Any
from webbrowser import Opera

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


def _default_name() -> str:
    name = inspect.getfile(inspect.stack()[2][0])
    try:
        name = str(Path(name).relative_to(rc.dirs.dapper / "mods"))
    except ValueError:
        name = str(Path(name))
    return name


class Operator(struct_tools.NicePrint):
    """Container for the dynamical and the observational maps.

    Parameters
    ----------
    M: int
        Length of output vectors.
    model: function
        The actual operator.
    linear:
    noise: RV, optional
        The associated additive noise. The noise can also be a scalar or an
        array, producing `GaussRV(C=noise)`.
    localizer: TODO: Describe
    object : TODO: Describe
    loc_shift : TODO: Describe

    Any remaining keyword arguments are written to the object as attributes.
    """

    def __init__(
        self,
        M: int,
        model: Optional[Callable] = None,
        linear: Optional[Callable] = None,
        noise: Optional[Union[RV, float]] = None,
        localizer: Any = None,  # TODO: Add proper type hints
        object: Any = None,  # TODO: Add proper type hints
        loc_shift: Any = None,  # TODO: Add proper type hints
    ):

        self.M = M
        self.model = model
        self.linear = linear
        self.noise = noise
        self.localizer = localizer
        self.object = object
        self.loc_shift = loc_shift

        # Default to the Identity operator
        if model is None:
            self.model = Id_op()

        if linear is None:
            self.linear = lambda *args: np.eye(M)

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

    # TODO: Probably good to remove this and force users to call Dyn.model(...) - explicit is better
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    printopts = {"ordering": ["M", "model", "noise"], "indent": 4}


class HiddenMarkovModel(struct_tools.NicePrint):
    """Container for a Hidden Markov Model (HMM).

    This should contain the details necessary to run synthetic DA experiments,
    also known as "twin experiment", or OSSE (observing system simulation experiment).
    The synthetic truth and observations may then be obtained by running
    `HiddenMarkovModel.simulate`.

    See scripts in examples for more details.

    Parameters
    ----------
    Dyn: `Operator` or dict
        Operator for the dynamics.
    Obs: `Operator` or dict
        Operator for the observations
    tseq: `dapper.tools.chronos.Chronology`
        Time sequence of the HMM process.
    X0: `dapper.tools.randvars.RV`
        Random distribution of initial condition
    liveplotters: `list`, optional
        A list of tuples. See example use in function `LPs` of `dapper.mods.Lorenz63`.
        - The first element of the tuple determines if the liveplotter
        is shown by default. If `False`, the liveplotter is only shown when
        included among the `liveplots` argument of `assimilate`
        - The second element in the tuple gives the corresponding liveplotter
        function/class.
    sectors: `dict`, optional
        Labelled indices referring to parts of the state vector.
        When defined, field-mean statistics are computed for each sector.
        Example use can be found in  `examples/param_estim.py`
        and `dapper/mods/Lorenz96/miyoshi2011.py`
    name: str, optional
        Label for the `HMM`.
    """

    def __init__(
        self,
        Dyn: Optional[Operator] = None,
        Obs: Optional[Operator] = None,
        tseq: Optional[Chronology] = None,
        X0: Optional[RV] = None,
        liveplotters: Optional[List[Callable]] = None,
        sectors: Optional[Dict[str, np.array]] = None,
        name: Optional[str] = _default_name(),
    ):

        self.Dyn = Dyn
        self.Obs = Obs
        self.tseq = tseq
        self.X0 = X0
        self.liveplotters = [] if liveplotters is None else liveplotters
        self.sectors = {} if sectors is None else sectors
        self.name = name

    # ndim shortcuts
    @property
    def Nx(self):
        return self.Dyn.M

    @property
    def Ny(self):
        return self.Obs.M

    printopts = {"ordering": ["Dyn", "Obs", "tseq", "X0"], "indent": 4}

    def simulate(self, desc="Truth & Obs"):
        """Generate synthetic truth and observations."""
        Dyn, Obs, tseq, X0 = self.Dyn, self.Obs, self.tseq, self.X0

        # Init
        xx = np.zeros((tseq.K + 1, Dyn.M))
        yy = np.zeros((tseq.Ko + 1, Obs.M))

        x = X0.sample(1)
        xx[0] = x

        # Loop
        for k, ko, t, dt in pb.progbar(tseq.ticker, desc):
            x = Dyn(x, t - dt, dt)
            x = x + np.sqrt(dt) * Dyn.noise.sample(1)
            if ko is not None:
                yy[ko] = Obs(x, t) + Obs.noise.sample(1)
            xx[k] = x

        return xx, yy

    def copy(self):
        return cp.deepcopy(self)
