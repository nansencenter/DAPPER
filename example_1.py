"""Illustrate how to use DAPPER to run a 'twin experiment'."""

# Load DAPPER
from dapper import *
seed(3)

# Load experiment setup: the hidden Markov Model (HMM)
from dapper.mods.Lorenz63.sak12 import HMM
HMM.t.T = 30 # shorten experiment
# config = EnKF('Sqrt', N=10, infl=1.02, rot=True)
# config = Var3D()
config = PartFilt(N=100,reg=2.4,NER=0.3)

HMM.sectors = dict(land=[0,1], ocean=[2])

# from dapper.mods.Lorenz95.boc10 import HMM
# HMM.t.T = 30 # shorten experiment
# config = PartFilt(N=100,NER=0.2 ,reg=1.3)           # 0.36

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = HMM.simulate()


# Turn on liveplotting
config.liveplots = False

# Assimilate yy, knowing the HMM; xx is used for assessment.
config.assimilate(HMM,xx,yy)

# Average stats time series
config.average_stats()

# Print averages
config.print_avrgs(['rmse.a','rmv.a'])

# Replay liveplotters -- can adjust speed, time-window, etc.
config.replay()

# Further diagnostic plots:
# plot_rank_histogram(config.stats)
# plot_err_components(config.stats)
# plot_hovmoller(xx)

# Explore objects:
# print(HMM)
# print(config)
# print(config.stats)
# print(config.avrgs)

# Excercise: Try using
# - Optimal interpolation
# - The (extended) Kalman filter
# - The iterative EnKS
# Hint: suggested DA configs are listed in the HMM file.

# Excercise: Run an experiment for each of the models:
# - LotkaVolterra
# - Lorenz95
# - LA
# - QG


