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
cfgs += PartFilt(N=100,reg=2.4,NER=0.3)
cfgs += PartFilt(N=100,reg=2.4,NER=0.9)
cfgs += PartFilt(N=100,reg=2.4,NER=1.0)

# TODO: infl, rot now explicit params

#N = 100
#for reg in [0.7,0.8,0.9]:
  #for Qs in [0.8,0.9,1.0,1.1]:
    #for NER in [0.3,0.9,1.0]:
      #for xN in [1000]:
        #cfgs.add(PFD,N=N,NER=NER,reg=reg,xN=xN,Qs=Qs)

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

