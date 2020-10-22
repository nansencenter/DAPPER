"""Illustrate usage of DAPPER to (interactively) run a 'twin experiment'."""

# Load DAPPER
from dapper import *

# Generate the same random numbers every time
set_seed(3000)

# Load experiment setup: the hidden Markov model (HMM)
from dapper.mods.Lorenz63.sakov2012 import HMM
HMM.t.T = 30 # shorten experiment

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = HMM.simulate()

# Specify a DA method configuration
config = EnKF('Sqrt', N=10, infl=1.02, rot=True)
# config = Var3D()
# config = PartFilt(N=100,reg=2.4,NER=0.3)

# Assimilate yy, knowing the HMM; xx is used to assess the performance
config.assimilate(HMM, xx, yy, liveplots=True)

# Average the time series of various statistics
config.stats.average_in_time()

# Print some averages
print(config.avrgs.tabulate(['rmse.a','rmv.a']))

# Replay liveplotters
replay(config.stats, speed=np.inf)

# Further diagnostic plots:
# plot_rank_histogram(config.stats)
# plot_err_components(config.stats)
# plot_hovmoller(xx)

# Explore objects:
# print(HMM)
# print(config)
# print(config.stats)
# print(config.avrgs)

# Excercise: Why does the replay look jagged?
# Hint: provide the keyword store_u=True to assimilate() to avoid this.

# Excercise: Why does the replay only contain the blue lines?

# Excercise: Try using
# - Optimal interpolation
# - The (extended) Kalman filter
# - The iterative EnKS
# Hint: suggested DA configs are listed in the HMM file

# Excercise: Run an experiment for each of the models:
# - LotkaVolterra
# - Lorenz96
# - LA
# - QG
