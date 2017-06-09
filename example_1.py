# Basic illustration of benchmarking DA methods with DAPPER

# Load DAPPER. Assumes pwd is <path-to-dapper>
from common import *

# Load "twin experiment" setup
from mods.Lorenz63.sak12 import setup
setup.t.T = 30

# Specify a DA method configuration
config = EnKF('Sqrt', N=10, infl=1.02, rot=True, liveplotting=True)

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = simulate(setup)

# Assimilate yy (knowing the twin setup). Assess vis-a-vis xx.
stats = config.assimilate(setup,xx,yy)

# Average stats time series
avrgs = stats.average_in_time()

# Print averages
print_averages(config,avrgs,[],['rmse_a','rmv_a'])

# Plot some diagnostics 
plot_time_series(stats)

# "Explore" objects individually
#print(setup)
#print(config)
#print(stats)
#print(avrgs)



