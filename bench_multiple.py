# Illustrate how to benchmark multiple methods
#
# Note: if model is very large, you may want to
# discard the stats object after each run, keeping
# only the avrgs.

from common import *

sd0 = seed(9)

from mods.Lorenz95.sak08 import setup
setup.t.T = 4**4

xx,yy = simulate(setup)

############################
# DA Configurations
############################
cfgs  = List_of_Configs()

cfgs += Climatology()
cfgs += D3Var()
cfgs += EnKF('PertObs',N=25,infl=1.10)
cfgs += EnKF('Sqrt',N=25,infl=1.05)
cfgs += EnKF('Sqrt',N=50,infl=1.03)
cfgs += EnKF_N(N=25)
cfgs += EnKF_N(N=50)

#cfgs += LETKF(         N=6,rot=True,infl=1.04,loc_rad=4)
#cfgs += LETKF(approx=1,N=8,rot=True,infl=1.25,loc_rad=4)
#cfgs += SL_EAKF(       N=6,rot=True,infl=1.07,loc_rad=6)

#cfgs.add(PFD,     N=100, xN=1000,NER=0.9,reg=0.7,Qs=0.9,nuj=1) # 1.05
#cfgs.add(PFD,     N=1000,xN=100, NER=0.9,reg=0.4,Qs=0.6,nuj=1) # 0.52

############################
# Assimilate
############################
stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  #config.store_u = True
  #config.liveplotting = True
  seed(sd0+2)

  stats += [ config.assimilate(setup,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)

# Single experiment
# config = PartFilt( N=250, NER=0.25)
# stats = config.assimilate(setup,xx,yy).average_in_time()
# print_averages(config, stats)


############################
# Plot
############################
#plot_time_series   (stats[-1])
#plot_3D_trajectory (stats[-1])
#plot_err_components(stats[-1])
#plot_rank_histogram(stats[-1])

# TODO: Verify speed changes before/after CovMat
# TODO: Rename assimilator and da_driver --> same
# TODO: cfgs.add --> cfgs +=
