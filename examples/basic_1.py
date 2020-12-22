"""Illustrate usage of DAPPER to (interactively) run a 'twin experiment'."""

import dapper as dpr
import dapper.da_methods as da

# Load experiment setup: the hidden Markov model (HMM)
from dapper.mods.Lorenz63.sakov2012 import HMM  # isort:skip

# Generate the same random numbers every time
dpr.set_seed(3000)

HMM.t.T = 30  # shorten experiment

# Simulate synthetic truth (xx) and noisy obs (yy)
xx, yy = HMM.simulate()

# Specify a DA method configuration ("xp" for "experiment")
xp = da.EnKF('Sqrt', N=10, infl=1.02, rot=True)
# xp = da.Var3D()
# xp = da.PartFilt(N=100,reg=2.4,NER=0.3)

# Assimilate yy, knowing the HMM; xx is used to assess the performance
xp.assimilate(HMM, xx, yy, liveplots=True)

# Average the time series of various statistics
xp.stats.average_in_time()

# Print some averages
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))

# Replay liveplotters
xp.stats.replay(speed=100)

# Further diagnostic plots:
# import dapper.tools.viz as viz
# viz.plot_rank_histogram(xp.stats)
# viz.plot_err_components(xp.stats)
# viz.plot_hovmoller(xx)

# Explore objects:
# print(HMM)
# print(xp)
# print(xp.stats)
# print(xp.avrgs)

# Excercise: Why does the replay look jagged?
# Hint: provide the keyword store_u=True to assimilate() to avoid this.

# Excercise: Why does the replay only contain the blue lines?

# Excercise: Try using
# - Optimal interpolation
# - The (extended) Kalman filter
# - The iterative EnKS
# Hint: suggested DA xp's are listed in the HMM file

# Excercise: Run an experiment for each of the models:
# - LotkaVolterra
# - Lorenz96
# - LA
# - QG
