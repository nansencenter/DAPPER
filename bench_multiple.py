# Illustrate how to use DAPPER to benchmark multiple DA methods

from common import *

sd0 = seed(9)

##############################
# DA Configurations
##############################
cfgs  = List_of_Configs()

from mods.Lorenz63.sak12 import setup ##################### Expected RMSE_a:
cfgs += Climatology()  # no tuning!                       # 8.5
cfgs += D3Var()        # tuning not stirctly required     # 1.26
cfgs += ExtKF(infl=90) # some inflation tuning needed     # 0.87
cfgs += EnKF('Sqrt',   N=3 ,  infl=1.30)                  # Very variable
cfgs += EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.63 (sak: 0.65)
cfgs += EnKF('PertObs',N=500, infl=0.95,rot=False)        # 0.56
cfgs += EnKF_N(        N=10,            rot=True)         # 0.54
cfgs += iEnKF('Sqrt',  N=10,  infl=1.02,rot=True,iMax=10) # 0.31
cfgs += PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
cfgs += PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
#cfgs += PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
#cfgs += PFD(xN=1000,   N=30  ,reg=0.7,NER=0.2,Qs=2)       # 0.56

#from mods.Lorenz95.sak08 import setup ##################### Expected RMSE_a:
#cfgs += ExtKF(infl=6)
#cfgs += EnKF(PertObs',N=40,infl=1.06)                     # 0.22
#cfgs += EnKF(DEnKF  ',N=40,infl=1.01)                     # 0.18
#cfgs += EnKF(PertObs',N=28,infl=1.08)                     # 0.24
#cfgs += EnKF(Sqrt   ',N=24,infl=1.02,rot=True)            # 0.18
#
#cfgs += EnKF_N(N=24,rot=True)                             # 0.18
#cfgs += iEnKF(Sqrt',N=40,iMax=10,infl=1.01,rot=True)      # 0.17
#
#cfgs += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4)  # 0.22
#cfgs += LETKF(approx=1,N=8,rot=True,infl=1.25,loc_rad=4)  # 0.36
#cfgs += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)  # 0.23


#from mods.LA.raanes2015 import setup
#from mods.Lorenz95.spectral_obs import setup
#from mods.Lorenz95.raanes2016 import setup
#from mods.LorenzXY.defaults import setup
#from mods.Lorenz84.harder import setup
# -- Get suggested tuning from setup files --


##############################
# Generate synthetic truth/obs
##############################
# Adjust experiment duration
setup.t.T = 100

xx,yy = simulate(setup)


##############################
# Assimilate
##############################
stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  #config.store_u = True
  #config.liveplotting = True
  seed(sd0+2)

  stats += [ config.assimilate(setup,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  #print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)

# Note: if model is very large, you may want to
# discard the stats objects, keeping only the avrgs.

##############################
# Plot
##############################
#plot_time_series   (stats[-1])
#plot_3D_trajectory (stats[-1])
#plot_err_components(stats[-1])
#plot_rank_histogram(stats[-1])
#plot_hovmoller     (xx,setup.t)



