############################
# Preamble
############################
from common import *

np.random.seed(5)
#LCG(5)


############################
# Setup
############################

#from mods.Lorenz63.sak12 import setup
# Expected rmse_a = 0.63 (sak 0.65)
#cfg           = DAM(EnKF)
#cfg.N         = 10
#cfg.infl      = 1.02
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = True
#
#cfg.da_method = iEnKF # rmse_a = 0.31
#cfg.iMax      = 10
#
#cfg           = DAM(PartFilt) # rmse_a = 0.275 (N=4000)
#cfg.N         = 800
#cfg.NER       = 0.1
#
#setup.t.dkObs = 10
#cfg = DAM(ExtKF, infl = 1.05)


from mods.Lorenz95.sak08 import setup
#
cfg           = DAM(EnKF)
cfg.N         = 24
cfg.infl      = 1.018
cfg.AMethod   = 'Sqrt'
cfg.rot       = True
#
#cfg = DAM(Climatology)
#cfg = DAM(D3Var)
#cfg = DAM(ExtKF, infl = 1.05)
#cfg = DAM(EnsCheat)

#from mods.Lorenz95.spectral_obs import setup
#from mods.Lorenz95.m33 import setup

#from mods.LA.raanes2014 import setup


############################
# Common
############################
setup.t.T = 4**3
cfg.liveplotting = True


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
