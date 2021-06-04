# ## Illustrate usage of DAPPER to (interactively) run a synthetic ("twin") experiment.

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then you must first execute (in a code cell) this:
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git`,
# and remove `%matplotlib notebook` from below. Also note that liveplotting
# does not work on Colab.

# %matplotlib notebook
from mpl_tools import is_notebook_or_qt as nb

import dapper as dpr
import dapper.da_methods as da

# #### Load experiment setup: the hidden Markov model (HMM)

from dapper.mods.Lorenz63.sakov2012 import HMM  # isort:skip

# #### Generate the same random numbers each time this script is run

seed = dpr.set_seed(3000)

# #### Simulate synthetic truth (xx) and noisy obs (yy)

HMM.t.T = 30  # shorten experiment
xx, yy = HMM.simulate()

# #### Specify a DA method configuration ("xp" for "experiment")

xp = da.EnKF('Sqrt', N=10, infl=1.02, rot=True)
# xp = da.Var3D()
# xp = da.PartFilt(N=100, reg=2.4, NER=0.3)

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

xp.assimilate(HMM, xx, yy, liveplots=not nb)

# #### Average the time series of various statistics

xp.stats.average_in_time()

# #### Print some averages

print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))

# #### Replay liveplotters

xp.stats.replay(
    # speed=.6
)

# #### Further diagnostic plots

if nb:
    import dapper.tools.viz as viz
    viz.plot_rank_histogram(xp.stats)
    viz.plot_err_components(xp.stats)
    viz.plot_hovmoller(xx)

# #### Explore objects

if nb:
    print(xp)

if nb:
    print(HMM)

if nb:
    # print(xp.stats) # quite long printout
    print(xp.avrgs)

# #### Excercise: Why does the replay look jagged?
# Hint: provide the keyword `store_u=True` to `assimilate()` to avoid this.

# #### Excercise: Why does the replay only contain the blue lines?

# #### Excercise: Try using
# - Optimal interpolation
# - The (extended) Kalman filter
# - The iterative EnKS
#
# Hint: suggested DA method settings are listed in the HMM files,
# like `dapper.mods.Lorenz63.sakov2012`.

# #### Excercise: Run an experiment for each of these models
# - LotkaVolterra
# - Lorenz96
# - LA
# - QG

# #### Excercise: Printing other diagnostics.
# - Create a new code cell, and copy-paste the above `print(...tabulate)`
#   command into it. Then, replace `rmse` by `err.rms`. This should yield
#   the same printout, as is merely an abbreviation of the latter.
# - Next, figure out how to print the time average *forecast (i.e. prior)* error
#   (and `rmv`) instead. Explain why the values are larger than
#   for the *analysis* values.
# - Finally, instead of the `rms` spatial/field averages,
#   print the regular mean (`.m`) averages. Explain why `err.m` is nearly zero,
#   in contrast to `err.rms`.
