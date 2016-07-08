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

from mods.L3.sak12 import params
## Expected rmse_a = 0.63 (sak 0.65)
#cfg.N         = 10
#cfg.infl      = 1.02
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = True
##cfg.da_method = EnKF
#
#cfg.da_method = iEnKF # rmse_a = 0.31
#cfg.iMax      = 10
cfg.da_method = PartFilt # rmse_a = 0.275 (N=4000)
cfg.N         = 800
cfg.NER       = 0.1

#from mods.L40.sak08 import params
## Expected rmse_a = 0.175
#cfg.N         = 40
#cfg.infl      = 1.01
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = True
#cfg.da_method = EnKF

##from mods.L40.spectral_obs import params
#cfg.N         = 40
#cfg.infl      = 1.005
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = False
#cfg.da_method = EnKF

#from mods.LA.raanes2014 import params
## Expected rmse_a = 0.3
#cfg.N         = 30
#cfg.infl      = 3.4
#cfg.AMethod   = 'PertObs'
#cfg.rot       = False
#cfg.da_method = EnKF

############################
# Generate synthetic truth/obs
############################
f,h,chrono,X0 = params.f, params.h, params.t, params.X0

# truth
xx = zeros((chrono.K+1,f.m))
xx[0,:] = X0.sample(1)
for k,kObs,t,dt in chrono.forecast_range:
  D = sqrt(dt)*f.noise.sample(1)
  xx[k,:] = f.model(xx[k-1,:],t-dt,dt) + D

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

############################
# Plot
############################
plot_diagnostics_dashboard(xx,s,chrono,cfg.N,dim=2)
plot_3D_trajectory(xx[:,:3],s,chrono)

#plt.waitforbuttonpress()
