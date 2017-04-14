# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(9)

from mods.Lorenz95.boc10 import setup
setup.t.T = 4**3.5

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs = DAC_list()
#cfgs.add(Climatology)
#cfgs.add(EnKF,'Sqrt',N=24,rot=True,infl=1.05)
#cfgs.add(EnKF_N,N=24,rot=True,infl=1.00)


#N = 300
#for reg in [0.2]:
  #for Qs in [0.3]:
    #for NER in [0.2]:
      #for Nm in [1000]:
        #cfgs.add(PFD,N=N,NER=NER,reg=reg,Nm=Nm,Qs=Qs,nuj=False)

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

