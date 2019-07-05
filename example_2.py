# Illustrate how to use DAPPER
# to benchmark multiple DA methods

from dapper import *

sd0 = seed(9)

##############################
# DA Configurations
##############################
cfgs  = List_of_Configs()

from dapper.mods.Lorenz63.sak12 import HMM       # Expected RMSE_a:
cfgs += Climatology()   # no tuning!                      # 7.6
cfgs += OptInterp()     # no tuning!                      # 1.25
cfgs += Var3D(infl=0.9) # tuning not strictly required    # 1.03 
cfgs += ExtKF(infl=90)  # some inflation tuning needed    # 0.87
cfgs += EnKF('Sqrt',    N=3 ,  infl=1.30)                 # 0.82
cfgs += EnKF('Sqrt',    N=10,  infl=1.02,rot=True)        # 0.63
cfgs += EnKF('PertObs', N=500, infl=0.95,rot=False)       # 0.56
cfgs += EnKF_N(         N=10,            rot=True)        # 0.54
cfgs += iEnKS('Sqrt',   N=10,  infl=1.02,rot=True)        # 0.31
cfgs += PartFilt(       N=100 ,reg=2.4  ,NER=0.3)         # 0.38
cfgs += PartFilt(       N=800 ,reg=0.9  ,NER=0.2)         # 0.28
# cfgs += PartFilt(      N=4000,reg=0.7  ,NER=0.05)       # 0.27
# cfgs += PFxN(xN=1000,  N=30  ,Qs=2     ,NER=0.2)        # 0.56

# from dapper.mods.Lorenz95.sak08 import HMM       # Expected RMSE_a:
# cfgs += Climatology()                                     # 3.6
# cfgs += OptInterp()                                       # 0.95
# cfgs += Var3D(infl=1.05)                                  # 0.41 
# cfgs += ExtKF(infl=6)                                     # 0.24
# cfgs += EnKF('PertObs'        ,N=40,infl=1.06)            # 0.22
# cfgs += EnKF('Sqrt'           ,N=28,infl=1.02,rot=True)   # 0.18
# 
# cfgs += EnKF_N(N=24,rot=True)                             # 0.21
# cfgs += EnKF_N(N=24,rot=True,xN=2)                        # 0.18
# cfgs += iEnKS('Sqrt',N=40,infl=1.01,rot=True)             # 0.17
# 
# cfgs += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4)  # 0.22
# cfgs += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)  # 0.23

# Other models (suitable cfgs listed in HMM files):
# from dapper.mods.Lorenz84.harder       import HMM
# from dapper.mods.Lorenz95.sak08        import HMM
# from dapper.mods.LA      .raanes2015   import HMM
# from dapper.mods.Lorenz95.raanes2016   import HMM
# from dapper.mods.LorenzUV.wilks05_full import HMM


##############################
# Generate synthetic truth/obs
##############################
# Adjust experiment duration
HMM.t.T = 100

xx,yy = simulate(HMM)


##############################
# Assimilate
##############################
stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  config.liveplotting = False
  seed(sd0+2) # use common random nums

  stats += [ config.assimilate(HMM,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  # print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)


##############################
# Plot
##############################
# plot_time_series   (stats[-1])
# plot_3D_trajectory (stats[-1])
# plot_err_components(stats[-1])
# plot_rank_histogram(stats[-1])
# plot_hovmoller     (xx,HMM.t)



