# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(3)

from mods.Lorenz95.boc10 import setup
setup.t.T = 4**4

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs = DAC_list()
cfgs.add(Climatology)
cfgs.add(EnKF_N,N=24,rot=True,infl=1.01)
cfgs.add(PartFilt,N=800,NER=0.05)
#cfgs.add(iEnKF,'Sqrt', N=12, infl=1.02,rot=True,iMax=10)

#N = 50
#for p in [1.0, 1.1, 1.3, 1.5]:
  #for r in [1.0, 1.5, 2, 2.5, 3, 4, 5, 7, 9]:
    #for NER in [0.05, 0.10, 0.4, 0.99]:
      #cfgs.add(PartFilt, N=N, NER=NER, rsmpl_root=r, prior_root=p, adhoc_noise=1)

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

