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

from mods.Lorenz63.sak12 import setup                          # Expected RMSE_a:
#config = DAC(Climatology)                                      # 8.5
#config = DAC(D3Var)                                            # 1.26
#config = DAC(ExtKF, infl=90);                                  # 0.87
#config = DAC(EnKF,'Sqrt',    N=3 , infl=1.30)                  # 
#config = DAC(EnKF ,'Sqrt',   N=10, infl=1.02,rot=True)         # 0.63 (sak: 0.65)
#config = DAC(EnKF ,'PertObs',N=500,infl=0.95,rot=False)        # 0.56
#config = DAC(iEnKF,'Sqrt',   N=10, infl=1.02,rot=True,iMax=10) # 0.31
#config = DAC(PartFilt,       N=800,NER=0.05)                   # 0.275 (with N=4000)

config = DAC(PartFilt,'Multinomial',N=200,NER=1)

#config = DAC(EnKF_N, N=10, rot=True)

#from mods.Lorenz95.sak08 import setup                   # Expected RMSE_a:
#config = DAC(Climatology)
#config = DAC(D3Var)
#config = DAC(ExtKF, infl = 6)
#config = DAC(EnCheat,'Sqrt',N=24,infl=1.02,rot=True)
#
#config = DAC(EnKF,'PertObs',N=40,infl=1.06)            # 0.22
#config = DAC(EnKF,'DEnKF  ',N=40,infl=1.01)            # 0.18
#config = DAC(EnKF,'PertObs',N=28,infl=1.08)            # 0.24
#config = DAC(EnKF,'Sqrt   ',N=24,infl=1.02,rot=True)   # 0.18
#
#config = DAC(EnKF_N,N=24,rot=True)
#
#config = DAC(iEnKF,'Sqrt',N=40,iMax=10,infl=1.01,rot=True) # 0.17
#
#config = DAC(LETKF,         N=6,rot=True,infl=1.04,loc_rad=4)
#config = DAC(LETKF,'approx',N=8,rot=True,infl=1.25,loc_rad=4)
#config = DAC(SL_EAKF,       N=6,rot=True,infl=1.07,loc_rad=6)


#from mods.Lorenz95.spectral_obs import setup
#from mods.Lorenz95.raanes2016 import setup
#from mods.LorenzXY.defaults import setup
#from mods.LA.raanes2015 import setup
#from mods.Lorenz84.harder import setup
# -- Get suggested tuning from setup files --


# # TODO: Expect 26 it/s for truth, and 9 it/s for EnKF. Undo?
# from mods.QG.sak08 import setup
# #config = DAC(EnKF,'PertObs',N=25,infl=1.10)
# #config = DAC(LETKF,N=25,infl=1.06,loc_rad=10.0)
# #config = DAC(LETKF,'approx',N=25,infl=1.06,locf=setup.locf(10,'x2y'))
# config = DAC(SL_EAKF,N=25,infl=1.03,loc_rad=10)


############################
# Common
############################
setup.t.T           = 4**3.5

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
stats = assimilate(setup,config,xx,yy)
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

