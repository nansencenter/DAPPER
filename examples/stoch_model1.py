# ## Reproduce basics of `bib.grudzien2020numerical`, and do live plotting.
# This is a simple demonstration of the effect of using low-precision numerics
# for a perfect-random model configuration in which the system is not dominated
# by the noise.  In this case, we see a failure of the ensemble generated
# by the Euler-Maruyama scheme to maintain adequate filter performance, due
# to the accumulation of model errors in the discretization scheme.  Although
# the stochastic Runge-Kutta scheme is of the same order strong / weak convergence
# the Runge-Kutta scheme is widely demonstrated to be a more statistically
# robust solver out-of-the-box, and is generally recommended for use
# for the ensemble generation with the model twin.  In certain situations
# in which the model structure lends itself to reduction, higher order
# schemes can be derived, such as with the Lorenz-96s system, with
# the second order Taylor-Stratonovich expansion.

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then replace `%matplotlib notebook` below by
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git` .
# Also note that liveplotting does not work on Colab.

# %matplotlib notebook
from mpl_tools import is_notebook_or_qt as nb

import dapper as dpr
import dapper.da_methods as da

# #### Load experiment setup: the hidden Markov model (HMM)

from dapper.mods.Lorenz96s.grudzien2020 import HMMs

# #### Generate the same random numbers each time this script is run

seed = dpr.set_seed(3000)

# #### Simulate synthetic truth (xx) and noisy obs (yy)

xx, yy = HMMs().simulate()

# The model trajectories only use half the integration steps
# of the true trajectory (compare `t5`, `t10`). Subsample `xx`
# so that the time series are compatible for comparison.

xx = xx[::2]

# #### Specify a DA method configuration ("xp" for "experiment")

xp = da.EnKF("PertObs", N=100)

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

xp.assimilate(HMMs("RK4"), xx, yy, liveplots=not nb)

# #### Average the time series of various statistics; print some averages

xp.stats.average_in_time()
print(xp.avrgs.tabulate(["rmse.a", "rmv.a"]))

# #### The above used the Runge-Kutta scheme. Repeat it, but with Euler-Maruyama

xp.assimilate(HMMs("EM"), xx, yy, liveplots=not nb)
xp.stats.average_in_time()
print(xp.avrgs.tabulate(["rmse.a", "rmv.a"]))
