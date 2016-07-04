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
# Expected rmse_a = 0.63 (sak 0.65)
cfg.N       = 10
cfg.infl    = 1.02
cfg.AMethod = 'Sqrt'
cfg.rot     = True
method      = EnKF
#cfg.iMax    = 10
#method      = iEnKF # rmse_a = 0.31

#from mods.L40.sak08 import params
## Expected rmse_a = 0.175
#cfg.N = 40
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#method      = EnKF

#from mods.LA.even2009 import params
## Expected rmse_a = 0.175
#cfg.N       = 100
#cfg.infl    = 1.01
#cfg.AMethod = 'PertObs'
#cfg.rot     = False
#method      = EnKF


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
s = method(params,cfg,xx,yy)


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


