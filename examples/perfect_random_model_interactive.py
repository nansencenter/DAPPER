# ## Illustrate usage of DAPPER to (interactively) run a synthetic ("twin") experiment.

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then replace `%matplotlib notebook` below by
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git` .
# Also note that liveplotting does not work on Colab.

# %matplotlib notebook
from mpl_tools import is_notebook_or_qt as nb

import dapper as dpr
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.Lorenz96s.grudzien2020 import (EMDyn, RKDyn, TruthDyn, X0, Obs,
                                                ttruth_low_precision,
                                                tmodel_low_precision)


# #### Load experiment setup: the hidden Markov model (HMM)


# #### Generate the same random numbers each time this script is run

seed = dpr.set_seed(3000)

# #### Simulate synthetic truth (xx) and noisy obs (yy)
HMM_truth = modelling.HiddenMarkovModel(TruthDyn, Obs, ttruth_low_precision, X0)
xx, yy = HMM_truth.simulate()

# note - the truth twin takes twice the number of the integration steps
# as the model twin in this configuration
xx = xx[::2]

# #### Specify a DA method configuration ("xp" for "experiment")

xp = da.EnKF('PertObs', N=100, infl=1.00, rot=False)
# xp = da.Var3D()
# xp = da.PartFilt(N=100, reg=2.4, NER=0.3)

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

# note - this is only for the Runge-Kutta scheme
HMM_rk_ensemble = modelling.HiddenMarkovModel(RKDyn, Obs, tmodel_low_precision, X0)
xp.assimilate(HMM_rk_ensemble, xx, yy, liveplots=not nb)

# #### Average the time series of various statistics

xp.stats.average_in_time()

# #### Print some averages

print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

# note - this repeated for the Euler-Maruyama scheme
HMM_em_ensemble = modelling.HiddenMarkovModel(EMDyn, Obs, tmodel_low_precision, X0)
xp.assimilate(HMM_em_ensemble, xx, yy, liveplots=not nb)

# #### Average the time series of various statistics

xp.stats.average_in_time()

# #### Print some averages

print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))
