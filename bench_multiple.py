# Illustrate how to benchmark multiple methods

from common import *

sd0 = seed()

from mods.Lorenz63.sak12 import setup
setup.t.T = 200

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs = DAC_list()
cfgs.add(Climatology)
cfgs.add(D3Var)
cfgs.add(EnKF,'Sqrt', N=3 ,infl=1.30,liveplotting=True)
cfgs.add(EnKF ,'Sqrt',N=10,infl=1.02,rot=True)
cfgs.add(iEnKF,'Sqrt',N=10,infl=1.02,rot=True,iMax=10)
cfgs.add(PartFilt,    N=800,NER=0.1)
cfgs.add(EnKF ,'PertObs',N=500,infl=0.95,rot=False)
cfgs.add(PartFilt, N=1000, NER=0.1)

############################
# Assimilate
############################
avrgs = []

for ic,config in enumerate(cfgs):
  #config.liveplotting = True
  seed(sd0+2)

  stats  = assimilate(setup,config,xx,yy)
  avrgs += [stats.average_in_time()]
print_averages(cfgs,avrgs)

# Single experiment
# config = DAC(PartFilt, N=250, NER=0.25)
# stats = assimilate(setup,config,xx,yy).average_in_time()
# print_averages(config, stats)


############################
# Plot
############################
# NB: Last config only!
plot_time_series   (stats,xx,dim=2)
plot_3D_trajectory (stats,xx)
plot_err_components(stats)
plot_rank_histogram(stats)


