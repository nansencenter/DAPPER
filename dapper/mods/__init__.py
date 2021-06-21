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
    - Typically, this culminates in a `step(x, t, dt)` function,
      which defines the dynamical model/system mapping the state `x`
      from one time `t` to another `t + dt`.
      This model "operator" must support
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

     - You should also define an example initial state, `x0`.  This facililates
       the specification of initial conditions for different synthetic
       experiments, as random variables centered on `x0`.  It is also a
       convenient way just to specify the system size as `len(x0)`.  In many
       experiments, the specific value of `x0` does not matter, because most
       systems are chaotic, and the average of the stats are computed only for
       `time > BurnIn > 0`, which will not depend on `x0` if the experiment is
       long enough.  Nevertheless, it's often convenient to pre-define a point
       on the attractor, or basin, or at least ensure "physicality", for
       quicker spin-up (burn-in).

    - Optional: define a number called `Tplot` which defines
      the (sliding) time window used by the liveplotting of diagnostics.

    - Optional: To use the (extended) Kalman filter, or 4D-Var,
      you will need to define the model linearization, typically called `dstep_dx`.
      Note: this only needs to support 1D input (single realization).

- Most models are defined using a procedural and function-based style.
  However, `dapper.mods.LorenzUV` and `dapper.mods.QG` use OOP.
  This is more flexible & robust, and better suited when different
  control-variable settings are to be investigated.

    .. note::
        In parameter estimation problems, the parameters are treated as input
        variables to the "forward model". This does not necessarily require
        OOP. See `examples/param_estim.py`.

- Make a file: `my_model/demo.py` to visually showcase
  a simulation of the model, and verify it's working.

    .. hint::
        To begin with, test whether the model works on 1 realization,
        before running it with several (simultaneously).
        Also, start with a small integration time step,
        before using more efficient/adventurous time steps.
        Note that the time step might need to be shorter in assimilation,
        because it may cause instabilities.

- Ideally, both `my_model/__init__.py` and `my_model/demo.py`
  do not rely on components of DAPPER outside of `dapper.mods`.

- Make a file: `my_model/my_settings_1.py` that defines
  (or "configures", since there is usually little actual programming taking place)
  a complete Hidden Markov Model ready for a synthetic experiment
  (also called "twin experiment" or OSSE).
  See `dapper.mods.HiddenMarkovModel` for details on what this requires.
  Each existing model comes with several examples of model settings from the literature.
  See, for example, `dapper.mods.Lorenz63.sakov2012`.

    .. warning::
      These configurations do not necessarily hold a very high programming standard,
      as they may have been whipped up at short notice to replicate some experiments,
      and are not intended for re-use.

      Nevertheless, sometimes they are re-used by another configuration script,
      leading to a major gotcha/pitfall: changes made to the imported `HMM` (or
      the model's module itself) also impact the original object (since they
      are mutable and thereby referenced).  This usually isn't an issue, since
      one rarely imports two/more separate configurations. However, the test suite
      imports all configurations, which might then unintentionally interact.
      To avoid this, you should use the `copy` method of the `HMM`
      before making any changes to it.

    Once you've made some experiments you believe are noteworthy you should add a
    "suggested settings/tunings" section in comments at the bottom of
    `my_model/my_settings_1.py`, listing some of the relevant DA method
    configurations that you tested, along with the RMSE (or other stats) that
    you obtained for those methods.  You will find plenty of examples already in DAPPER,
    used for cross-referenced with literature to verify the workings of DAPPER
    (and the reproducibility of publications).
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
