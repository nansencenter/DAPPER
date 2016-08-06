############################
# Preamble
############################
from common import *

np.random.seed(5)
#LCG(5)

############################
# Setup
############################
cfg = Settings()

#from mods.Lorenz63.sak12 import params
# Expected rmse_a = 0.63 (sak 0.65)
#cfg.da_method = EnKF
#cfg.N         = 10
#cfg.infl      = 1.02
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = True
#
#cfg.da_method = iEnKF # rmse_a = 0.31
#cfg.iMax      = 10
#
#cfg.da_method = PartFilt # rmse_a = 0.275 (N=4000)
#cfg.N         = 800
#cfg.NER       = 0.1
#
#params.t.dkObs = 10
#cfg.da_method = ExtKF
#cfg.infl = 1.05


#from mods.Lorenz95.sak08 import params
##
#cfg.N         = 38
#cfg.infl      = 1.02
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = True
#cfg.da_method = EnKF
#
#cfg.da_method = Climatology
#cfg.da_method = D3Var
#cfg.da_method = ExtKF; cfg.infl = 1.05
#cfg.da_method = EnsCheat

#from mods.Lorenz95.spectral_obs import params
#from mods.Lorenz95.m33 import params

from mods.LA.raanes2014 import params
cfg.N         = 30
cfg.infl      = 1.01
#cfg.infl      = 3.4 # Why is rmse performance so insensitive to inflation
cfg.AMethod   = 'PertObs'
cfg.rot       = False
cfg.da_method = EnKF

#params.t.T = 4**3.5


############################
# Generate synthetic truth/obs
############################
f,h,chrono,X0 = params.f, params.h, params.t, params.X0

# truth
xx = zeros((chrono.K+1,f.m))
xx[0,:] = X0.sample(1)
for k,_,t,dt in chrono.forecast_range:
  xx[k,:] = f.model(xx[k-1,:],t-dt,dt) + sqrt(dt)*f.noise.sample(1)

# obs
yy = zeros((chrono.KObs+1,h.m))
for k,t in enumerate(chrono.ttObs):
  yy[k,:] = h.model(xx[chrono.kkObs[k],:],t) + h.noise.sample(1)

############################
# Assimilate
############################
s = Assimilate(params,cfg,xx,yy)


############################
# Report averages
############################
print('Mean analysis RMSE: {: 8.5f} +/- {:<5g},    RMSV: {:8.5f}'\
    .format(*series_mean_with_conf(s.rmse[chrono.kkObsBI]),mean(s.rmsv[chrono.kkObsBI])))
print('Mean forecast RMSE: {: 8.5f} +/- {:<5g},    RMSV: {:8.5f}'\
    .format(*series_mean_with_conf(s.rmse[chrono.kkObsBI-1]),mean(s.rmsv[chrono.kkObsBI-1])))
#print('Mean analysis misf: {: 8.5f} +/- {:<5g}' \
    #.format(*series_mean_with_conf(s.smisf[chrono.kkObsBI])))
#print('Mean analysis ldet: {: 8.5f} +/- {:<5g}' \
    #.format(*series_mean_with_conf(s.ldet[chrono.kkObsBI])))
print('Mean analysis logp: {: 8.5f} +/- {:<5g}' \
    .format(*series_mean_with_conf(s.logp[chrono.kkObsBI])))
print('Mean analys logp_r: {: 8.5f} +/- {:<5g}' \
    .format(*series_mean_with_conf(s.logp_r[chrono.kkObsBI])))

############################
# Plot
############################
plot_time_series(xx,s,chrono,dim=2)
plot_ens_stats(xx,s,chrono,cfg)
plot_3D_trajectory(xx[:,:3],s,chrono)

#plt.waitforbuttonpress()
