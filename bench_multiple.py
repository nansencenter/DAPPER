# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(9)

#from mods.Lorenz95.boc10 import setup
from mods.Lorenz63.sak12 import setup
setup.t.T = 4**5.5

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs  = List_of_Configs()

#cfgs += PFD(N=30,Qs=1.8,xN=1000,reg=0.7,NER=0.2)
#cfgs += PFD(N=30,Qs=2.0,xN=1000,reg=0.7,NER=0.2)
cfgs += PFD(N=30,Qs=2.2,xN=1000,reg=0.7,NER=0.2)
cfgs += PFD(N=30,Qs=2.4,xN=1000,reg=0.7,NER=0.2)

############################
# Assimilate
############################
stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  #config.liveplotting = True
  seed(sd0+2)

  stats += [ config.assimilate(setup,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)

# Single experiment
# config = PartFilt( N=250, NER=0.25)
# stats = assimilate(setup,config,xx,yy).average_in_time()
# print_averages(config, stats)


############################
# Plot
############################
# plot_time_series   (stats[-1])
# plot_3D_trajectory (stats[-1])
# plot_err_components(stats[-1])
# plot_rank_histogram(stats[-1])

