# Example script including several setups and DA methods.

############################
# Preamble
############################
from common import *

sd0 = seed(5)

############################
# Setup
############################

#from mods.Lorenz63.sak12 import setup                        # Expected RMSE_a:
#config = DAC(EnKF,'Sqrt', N=3 ,infl=1.30)
#config = DAC(EnKF ,'Sqrt',N=10,infl=1.02,rot=True)          # 0.63 (sak: 0.65)
#config = DAC(iEnKF,'Sqrt',N=10,infl=1.02,rot=True,iMax=10)  # 0.31
#config = DAC(PartFilt,    N=800,NER=0.1)                    # 0.275 (with N=4000)
#config = DAC(ExtKF, infl=1.05); setup.t.dkObs = 10 # reduce non-linearity
#
#config = DAC(EnKF ,'PertObs',N=500,infl=0.95,rot=False)
#config = DAC(PartFilt, N=1000, NER=0.1)


from mods.Lorenz95.sak08 import setup                   # Expected RMSE_a:
#config = DAC(Climatology)
#config = DAC(D3Var)
#config = DAC(ExtKF, infl = 1.05)
#config = DAC(EnCheat,'Sqrt',N=24,infl=1.02,rot=True)
#
#config = DAC(EnKF,'PertObs',N=40,infl=1.06)            # 0.22
#config = DAC(EnKF,'DEnKF  ',N=40,infl=1.01)            # 0.18
#config = DAC(EnKF,'PertObs',N=28,infl=1.08)            # 0.24
#config = DAC(EnKF,'Sqrt   ',N=24,infl=1.02,rot=True)   # 0.18
#
config = DAC(EnKF_N,N=24,rot=True)
#
#config = DAC(iEnKF,'Sqrt',N=40,iMax=10,infl=1.01,rot=True) # 0.17
#
#config = DAC(LETKF,         N=6,rot=True,infl=1.04,locf=setup.locf(4,'x2y'))
#config = DAC(LETKF,'approx',N=8,rot=True,infl=1.25,locf=setup.locf(4,'x2y'))
#config = DAC(SL_EAKF,       N=6,rot=True,infl=1.07,locf=setup.locf(6,'y2x'))


#from mods.Lorenz95.spectral_obs import setup
# -- Get suggested tuning from setup files --
#from mods.Lorenz95.m33 import setup
# -- Get suggested tuning from setup files --
#from mods.LorenzXY.defaults import setup
# -- Get suggested tuning from setup files --
#from mods.LA.raanes2015 import setup
# -- Get suggested tuning from setup files --


############################
# Common
############################
config.liveplotting = False
setup.t.T           = 4**3.5


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
chrono = setup.t
plot_time_series  (xx      ,stats,chrono, dim=2)
plot_err_compons  (xx      ,stats,chrono, config)
plot_3D_trajectory(xx[:,:3],stats,chrono)

