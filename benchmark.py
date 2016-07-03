############################
# Preamble
############################
from common import *

# TODO: LA model
# TODO: PartFilt
# TODO: ExtKF
# TODO: Climatology
# TODO: 1D model from workshop that preserves some quantity
# TODO: 2D model
# TODO: average obs and truth rank hist
# TODO: iEnKS-N
# TODO: Models come with their own viz specification
#
# TODO: Add before/after analysis plots
#
# TODO: Truncate SVD at 95 or 99% (evensen)
#
# TODO: unify matrix vs array (e.g. randn)
#       vs 1d array (e.g. xx[:,0] in L3.dxdt)
#       avoid y  = yy[:,kObs].reshape((p,1))
# TODO: Take advantage of pass-by-ref
# TODO: Decide on conflicts np vs math vs sp
#
# TODO: prevent CovMat from being updated

np.random.seed(5)
#LCG(5)

cfg = Settings()

############################
# L3_sak12
############################
from mods.L3.sak12 import params

# rmse_a = ? (sak: 0.82)
#cfg.N       = 3
#cfg.infl    = 1.30 # Not well-tuned coz resuls too variable
#cfg.AMethod = 'Sqrt'
#cfg.rot     = False
#method      = EnKF

# rmse_a = 0.63 (sak 0.65)
cfg.N       = 10
cfg.infl    = 1.02
cfg.AMethod = 'Sqrt'
cfg.rot     = True
method      = EnKF
#cfg.iMax    = 10
#method      = iEnKF # rmse_a = 0.31

############################
# L3_v23
############################
#from mods.L3.m23 import params

## rmse_a = 0.205
#cfg.N       = 10
#cfg.infl    = 1.00001
#cfg.AMethod = 'Sqrt'
#cfg.rot     = False
#method      = EnKF

# rmse_a = 0.185
#cfg.N       = 10
#cfg.infl    = 1.00001
#cfg.AMethod = 'Sqrt'
#cfg.rot     = False
#cfg.iMax    = 10
#method      = iEnKF



############################
# L40_sak08
############################
#from mods.L40.sak08 import params

#cfg.N = 40

# rmse_a = 0.22
#cfg.infl    = 1.045 # Requires BurnIn inflation too
#cfg.infl    = 1.06
#cfg.AMethod = 'PertObs non-transposed'
#method      = EnKF

# rmse_a = 0.175
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#method      = EnKF

# rmse_a = 0.18
#cfg.infl    = 1.01
#cfg.AMethod = 'DEnKF'
#method      = EnKF

# rmse_a = 0.17
#cfg.infl    = 1.01
#cfg.AMethod = 'Sqrt'
#cfg.rot     = True
#cfg.iMax    = 10
#method      = iEnKF

############################
# L40_v23
############################
#from mods.L40.m23 import params

#cfg.N = 40
#cfg.infl    = 1.10
#cfg.rot     = False
#cfg.AMethod = 'Sqrt'
#method      = EnKF


############################
# Generate synthetic truth and obs
############################
f,h,chrono,X0 = params.f, params.h, params.t, params.X0

xx = zeros((chrono.K+1,f.m))
xx[0,:] = X0.sample(1)
for k,kObs,t,dt in chrono.forecast_range:
  D = sqrt(dt)*f.noise.sample(1)
  xx[k,:] = f.model(xx[k-1,:],t-dt,dt) + D

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
