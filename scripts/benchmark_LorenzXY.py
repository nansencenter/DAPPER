############################
# Preamble
############################
from common import *

sd0 = seed(5)

############################
# Setup
############################
from mods.LorenzXY.defaults  import setup as setup_full
from mods.LorenzXY.truncated import setup as setup_trunc
ratio_dt = validate_int(setup_trunc.t.dt / setup_full.t.dt)
#

cfgs = DAC_list()
cfgs += Climatology()
cfgs += Var3D()
cfgs += EnKF(infl=1.10,liveplotting=True)
keep = len(cfgs)-1
cfgs += EnKF(infl=1.15)
cfgs += EnKF(infl=1.20)

############################
# Common settings
############################
for method in cfgs:
  if method.da_driver is EnKF:
    method.N       = 7
    method.upd_a = 'Sqrt'
    method.rot     = True

############################
# Assimilate
############################
ss = np.empty(len(cfgs),dict)

xx,yy = simulate(setup_full)
xx    = xx[::ratio_dt,:setup_trunc.f.m]
for k,method in enumerate(cfgs):
  seed(sd0)
  stats = assimilate(setup_trunc,method,xx,yy)
  ss[k] = stats.average_in_time()
  if k == keep: kept = stats
print_averages(cfgs,ss)


############################
# Plot
############################
config = cfgs[keep]
chrono = setup_trunc.t
plot_time_series(kept,dim=2)
plot_err_components(kept)
plot_3D_trajectory(kept)


