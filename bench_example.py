# Example benchmarking script
# containing several setups and DA methods.

############################
# Preamble
############################
from common import *

sd0 = seed(5) # or ()

############################
# DA Configurations
############################

from mods.Lorenz63.sak12 import setup                      # Expected RMSE_a:
#config = Climatology()  # note no tuning required          # 8.5
#config = D3Var()        # tuning not stirctly required     # 1.26
#config = ExtKF(infl=90) # some inflation tuning needed     # 0.87
#config = EnKF('Sqrt',   N=3 ,  infl=1.30)                  # Very variable
#config = EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.63 (sak: 0.65)
#config = EnKF('PertObs',N=500, infl=0.95,rot=False)        # 0.56
#config = EnKF_N(        N=10,            rot=True)         # 0.54
#config = iEnKF('Sqrt',  N=10,  infl=1.02,rot=True,iMax=10) # 0.31
#config = PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
#config = PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
#config = PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
#config = PFD(xN=1000,   N=30  ,reg=0.7,NER=0.2,Qs=2)       # 0.56

#from mods.Lorenz95.sak08 import setup                      # Expected RMSE_a:
#config = ExtKF(infl=6)
#config = EnKF(PertObs',N=40,infl=1.06)                     # 0.22
#config = EnKF(DEnKF  ',N=40,infl=1.01)                     # 0.18
#config = EnKF(PertObs',N=28,infl=1.08)                     # 0.24
#config = EnKF(Sqrt   ',N=24,infl=1.02,rot=True)            # 0.18
#
#config = EnKF_N(N=24,rot=True)                             # 0.18
#config = iEnKF(Sqrt',N=40,iMax=10,infl=1.01,rot=True)      # 0.17
#
#config = LETKF(         N=6,rot=True,infl=1.04,loc_rad=4)  # 0.25
#config = LETKF(approx=1,N=8,rot=True,infl=1.25,loc_rad=4)  # 0.35
#config = SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)  # 0.22

#config = EnKS('Sqrt',N=25,infl=1.08,rot=False,tLag=2.0)
#config = EnRTS('Sqrt',N=25,infl=1.08,rot=False,cntr=0.99)


#from mods.Lorenz95.spectral_obs import setup
#from mods.Lorenz95.raanes2016 import setup
#from mods.LorenzXY.defaults import setup
#from mods.LA.raanes2015 import setup
#from mods.Lorenz84.harder import setup
# -- Get suggested tuning from setup files --


# # TODO
# from mods.QG.sak08 import setup
# #config = EnKF(PertObs',N=25,infl=1.10)
# #config = LETKF(=25,infl=1.06,loc_rad=10.0)
# #config = LETKF(approx',N=25,infl=1.06,locf=setup.locf(10,'x2y'))
# config = SL_EAKF(=25,infl=1.03,loc_rad=10)

############################
# Common
############################
setup.t.T           = 4**4.5

#config.liveplotting = True
#config.store_u      = True


############################
# Generate synthetic truth/obs
############################
seed(sd0)
xx,yy = simulate(setup)


############################
# Assimilate
############################
stats = config.assimilate(setup,xx,yy)
avrgs = stats.average_in_time()
print_averages(config,avrgs)


############################
# Plot
############################
# plot_time_series   (stats)
# plot_3D_trajectory (stats)
# plot_hovmoller(xx,setup.t)
# plot_err_components(stats)
# plot_rank_histogram(stats)

