"""Models included with DAPPER.

See the README section on
[test cases (models)](https://github.com/nansencenter/DAPPER#Test-cases-models)
for an overview of the models included with DAPPER.

The models are all simple;
this facililates the reliability, reproducibility, and attributability
of the experiment results.

## Defining your own model

Follow the example of one of the models within the `dapper/mods` folder.
Essentially, you just need to define all of the attributes of a
`dapper.mods.HiddenMarkovModel`.
To make sure this is working, we suggest the following structure:

- Make a dir: `my_model`
- Make a file: `my_model/__init__.py` where you define the core
  workings of the model.
  Typically, this culminates in a `step(x, t, dt)` function.

    - The model step operator (and the obs operator) must support
      2D-array (i.e. ensemble) and 1D-array (single realization) input.
      See

          - `dapper.mods.Lorenz63`: use of `ens_compatible`.
          - `dapper.mods.Lorenz96`: use of relatively clever slice notation.
          - `dapper.mods.LorenzUV`: use of cleverer slice notation: `...` (ellipsis).
            Consider pre-defining the slices like so:

                iiX = (..., slice(None, Nx))
                iiP = (..., slice(Nx, None))

            to abbreviate the indexing elsewhere.

          - `dapper.mods.QG`: use of parallelized for loop (map).

    - Optional: To use the (extended) Kalman filter, or 4D-Var,
      you will need to define the model linearization.
      Note: this only needs to support 1D input (single realization).

- Make a file: `my_model/demo.py` to visually showcase
  a simulation of the model.
- Ideally, both `my_model/__init__.py` and `my_model/demo.py`
  do not rely on the rest of DAPPER.
- Make a file: `my_model/my_settings_1.py` that define a complete
  Hidden Markov Model ready for a synthetic experiment
  (also called "twin experiment" or OSSE).


<!--
* To begin with, test whether the model works
    * on 1 realization
    * on several realizations (simultaneously)
* Thereafter, try assimilating using
    * a big ensemble
    * a safe (e.g. 1.2) inflation value
    * small initial perturbations
      (big/sharp noises might cause model blow up)
    * small(er) integrational time step
      (assimilation might create instabilities)
    * very large observation noise (free run)
    * or very small observation noise (perfectly observed system)
-->
"""

__pdoc__ = {"explore_props": False}

import copy as cp
import inspect
from pathlib import Path

import numpy as np
import struct_tools

# Imports used to set up HMMs
import dapper.tools.progressbar as pb
from dapper.dpr_config import rc
from dapper.mods.utils import Id_mat, Id_op
from dapper.tools.chronos import Chronology
from dapper.tools.localization import no_localization
from dapper.tools.matrices import CovMat
from dapper.tools.randvars import RV, GaussRV
from dapper.tools.seeding import set_seed

from .integration import with_recursion, with_rk4
from .utils import Id_Obs, ens_compatible, linspace_int, partial_Id_Obs


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
    t: `dapper.tools.chronos.Chronology`
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

    def __init__(self, *args, **kwargs):
        # Valid args/kwargs, along with type and default.
        # Note: it's still ok to write attributes to the HMM following the init.
        attrs = dict(Dyn=(Operator, None),
                     Obs=(Operator, None),
                     t=(Chronology, None),
                     X0=(RV, None),
                     liveplotters=(list, []),
                     sectors=(dict, {}),
                     name=(str, HiddenMarkovModel._default_name))

        # Transfer args to kwargs
        for arg, kw in zip(args, attrs):
            assert (kw not in kwargs), "Could not sort out arguments."
            kwargs[kw] = arg

        # Un-abbreviate
        abbrevs = {"LP": "liveplotters", "loc": "localizer"}
        for k in list(kwargs):
            try:
                full = abbrevs[k]
            except KeyError:
                pass
            else:
                assert (full not in kwargs), "Could not sort out arguments."
                kwargs[full] = kwargs.pop(k)

        # Transfer kwargs to self
        for k, (type_, default) in attrs.items():
            # Get kwargs[k] or default
            if k in kwargs:
                v = kwargs.pop(k)
            elif callable(default):
                v = default()
            else:
                v = default
            # Convert dict to type
            if not isinstance(v, (type_, type(None))):
                v = type_(**v)
            # Write
            setattr(self, k, v)
        assert kwargs == {}, f"Arguments {list(kwargs)} is/are invalid."

        # Further defaults
        if not hasattr(self.Obs, "localizer"):
            self.Obs.localizer = no_localization(self.Nx, self.Ny)

        # Validation
        if self.Obs.noise.C == 0 or self.Obs.noise.C.rk != self.Obs.noise.C.M:
            raise ValueError("Rank-deficient R not supported.")

    # ndim shortcuts
    @property
    def Nx(self): return self.Dyn.M
    @property
    def Ny(self): return self.Obs.M

    printopts = {'ordering': ['Dyn', 'Obs', 't', 'X0'], "indent": 4}

    def simulate(self, desc='Truth & Obs'):
        """Generate synthetic truth and observations."""
        Dyn, Obs, chrono, X0 = self.Dyn, self.Obs, self.t, self.X0

        # Init
        xx    = np.zeros((chrono.K   + 1, Dyn.M))
        yy    = np.zeros((chrono.KObs+1, Obs.M))

        xx[0] = X0.sample(1)

        # Loop
        for k, kObs, t, dt in pb.progbar(chrono.ticker, desc):
            xx[k] = Dyn(xx[k-1], t-dt, dt) + np.sqrt(dt)*Dyn.noise.sample(1)
            if kObs is not None:
                yy[kObs] = Obs(xx[k], t) + Obs.noise.sample(1)

        return xx, yy

    def copy(self):
        return cp.deepcopy(self)

    @staticmethod
    def _default_name():
        name = inspect.getfile(inspect.stack()[2][0])
        try:
            name = str(Path(name).relative_to(rc.dirs.dapper/'mods'))
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
    noise: RV, optional
        The associated additive noise. The noise can also be a scalar or an
        array, producing `GaussRV(C=noise)`.

    Any remaining keyword arguments are written to the object as attributes.
    """

    def __init__(self, M, model=None, noise=None, **kwargs):
        self.M = M

        # None => Identity model
        if model is None:
            model = Id_op()
            kwargs['linear'] = lambda x, t, dt: Id_mat(M)
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

    printopts = {'ordering': ['M', 'model', 'noise'], "indent": 4}
