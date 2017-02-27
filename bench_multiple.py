# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(3)

from mods.Lorenz84.harder import setup
setup.t.T = 800

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs = DAC_list()
cfgs.add(Climatology)
#cfgs.add(D3Var)
#cfgs.add(ExtKF,infl=8)
#cfgs.add(EnKF ,'Sqrt',N=100,rot=True,infl=1.01)
cfgs.add(EnKF_N,N=4)
#cfgs.add(PartFilt, N=100, NER=0.4)
#cfgs.add(PartFilt, N=1000, NER=0.1)

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
plot_time_series   (stats[-1])
plot_3D_trajectory (stats[-1])
plot_err_components(stats[-1])
plot_rank_histogram(stats[-1])


