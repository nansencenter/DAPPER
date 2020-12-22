"""Models included with DAPPER.

See the README section on
[test cases (models)](https://github.com/nansencenter/DAPPER#Test-cases-models)
for an overview of the models included with DAPPER.

The models are all simple;
this facililates the reliability, reproducibility, and transparency
of DA experiments.

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
      See `dapper.mods.Lorenz63` and `dapper.mods.Lorenz96`
      for typical implementations,
      and `dapper.mods.QG` for how to parallelize the ensemble simulations.
    - Optional: To use the (extended) Kalman filter, or 4D-Var,
      you will need to define the model linearization.
      Note: this only needs to support 1D input (single realization).
- Make a file: `my_model/demo.py` to visually showcase
  a simulation of the model.
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

    This container contains the specification of a "twin experiment",
    i.e. an "OSSE (observing system simulation experiment)".
    """

    def __init__(self, Dyn, Obs, t, X0, **kwargs):
        # fmt: off
        self.Dyn = Dyn if isinstance(Dyn, Operator)   else Operator  (**Dyn) # noqa
        self.Obs = Obs if isinstance(Obs, Operator)   else Operator  (**Obs) # noqa
        self.t   = t   if isinstance(t  , Chronology) else Chronology(**t)   # noqa
        self.X0  = X0  if isinstance(X0 , RV)         else RV        (**X0)  # noqa
        # fmt: on

        # Name
        self.name = kwargs.pop("name", "")
        if not self.name:
            name = inspect.getfile(inspect.stack()[1][0])
            try:
                self.name = str(Path(name).relative_to(rc.dirs.dapper/'mods'))
            except ValueError:
                self.name = str(Path(name))

        # Kwargs
        abbrevs = {'LP': 'liveplotters'}
        for key in kwargs:
            setattr(self, abbrevs.get(key, key), kwargs[key])

        # Defaults
        if not hasattr(self.Obs, "localizer"):
            self.Obs.localizer = no_localization(self.Nx, self.Ny)
        if not hasattr(self, "sectors"):
            self.sectors = {}

        # Validation
        if self.Obs.noise.C == 0 or self.Obs.noise.C.rk != self.Obs.noise.C.M:
            raise ValueError("Rank-deficient R not supported.")

    # ndim shortcuts
    @property
    def Nx(self): return self.Dyn.M
    @property
    def Ny(self): return self.Obs.M

    printopts = {'ordering': ['Dyn', 'Obs', 't', 'X0']}

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


class Operator(struct_tools.NicePrint):
    """Container for operators (models)."""

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

    printopts = {'ordering': ['M', 'model', 'noise']}
