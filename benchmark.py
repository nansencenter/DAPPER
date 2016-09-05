############################
# Preamble
############################
from common import *

np.random.seed(5)

############################
# Setup
############################

#from mods.Lorenz63.sak12 import setup
#cfg = DAM(EnKF,'Sqrt',N=3 ,infl=1.30)
#cfg = DAM(EnKF ,'Sqrt',N=10,infl=1.02,rot=True)          # 0.63 (sak: 0.65)
#cfg = DAM(iEnKF,'Sqrt',N=10,infl=1.02,rot=True,iMax=10)  # 0.31
#cfg = DAM(PartFilt, N=800, NER=0.1)                      # 0.275 (with N=4000)
#cfg = DAM(ExtKF, infl = 1.05); setup.t.dkObs = 10 # reduce non-linearity


from mods.Lorenz95.sak08 import setup                   # Expected RMSE_a:
#cfg = DAM(EnKF,'PertObs',N=40, infl=1.06)               # 0.22
cfg = DAM(EnKF,'DEnKF',N=40, infl=1.01)                 # 0.18
#cfg = DAM(EnKF,'PertObs',N=28,infl=1.08)                # 0.24
#cfg = DAM(EnKF,'Sqrt'   ,N=24,infl=1.013,rot=True)      # 0.18

#cfg = DAM(iEnKF,'Sqrt',N=40,iMax=10,infl=1.01,rot=True) # 0.17

#cfg = DAM(LETKF,N=6,rot=True,infl=1.04,locf=setup.locf(4,'x2y'))
#cfg = DAM(LETKF,'approx',N=8,rot=True,infl=1.25,locf=setup.locf(4,'x2y'))
#cfg = DAM(SL_EAKF,N=6,rot=True,infl=1.07,locf=setup.locf(6,'y2x'))
#
#cfg = DAM(Climatology)
#cfg = DAM(D3Var)
#cfg = DAM(ExtKF, infl = 1.05)
#cfg = DAM(EnCheat,'Sqrt',N=24,infl=1.02,rot=True)


#from mods.Lorenz95.spectral_obs import setup
# -- Get suggested tuning from setup files --
#from mods.Lorenz95.m33 import setup
# -- Get suggested tuning from setup files --
#from mods.LA.raanes2014 import setup
# -- Get suggested tuning from setup files --


############################
# Common
############################
cfg.liveplotting = True
setup.t.T        = 4**4


############################
# Generate synthetic truth/obs
############################
xx,yy = simulate(setup)


############################
# Assimilate
############################
s = assimilate(setup,cfg,xx,yy)


############################
# Report averages
############################
chrono = setup.t
kk_a = chrono.kkObsBI
kk_f = chrono.kkObsBI-1
print('Mean analysis RMSE: {: 8.5f} ± {:<5g},    RMV: {:8.5f}'
    .format(*series_mean_with_conf(s.rmse[kk_a]),mean(s.rmv[kk_a])))
print('Mean forecast RMSE: {: 8.5f} ± {:<5g},    RMV: {:8.5f}'
    .format(*series_mean_with_conf(s.rmse[kk_f]),mean(s.rmv[kk_f])))
print('Mean analysis MGSL: {: 8.5f} ± {:<5g}'
    .format(*series_mean_with_conf(s.logp_m[kk_a])))


############################
# Plot
############################
plot_time_series(xx,s,chrono,dim=2)
plot_ens_stats(xx,s,chrono,cfg)
plot_3D_trajectory(xx[:,:3],s,chrono)

