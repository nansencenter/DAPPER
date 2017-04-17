# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(9)

from mods.Lorenz95.boc10 import setup
setup.t.T = 4**4.0

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs = DAC_list()

cfgs.add(PartFilt,N=100,NER=0.2,reg=1.3)        # rmse_a = 0.35
cfgs.add(OptPF,   N=100,NER=0.2,reg=1.0,Qs=0.3) # rmse_a = 0.37
cfgs.add(PartFilt,N=800,NER=0.2,reg=0.8)        # rmse_a = 0.25


#N = 100
#for reg in [0.1]:
  #for Qs in sqrt([0.6, 0.8, 1.2, 1.8]):
    #for NER in [0.2]:
      #for xN in [100]:
        #cfgs.add(PFD,N=N,NER=NER,reg=reg,xN=xN,Qs=Qs,nuj=True)

############################
# Assimilate
############################
stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  #config.liveplotting = True
  seed(sd0+2)

  stats += [ assimilate(setup,config,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)

# Single experiment
# config = DAC(PartFilt, N=250, NER=0.25)
# stats = assimilate(setup,config,xx,yy).average_in_time()
# print_averages(config, stats)


############################
# Plot
############################
# plot_time_series   (stats[-1])
# plot_3D_trajectory (stats[-1])
# plot_err_components(stats[-1])
# plot_rank_histogram(stats[-1])

