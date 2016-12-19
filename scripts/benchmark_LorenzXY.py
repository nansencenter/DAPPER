############################
# Preamble
############################
from common import *

sd0 = 5
np.random.seed(sd0)

############################
# Setup
############################
from mods.LorenzXY.defaults  import setup as setup_full
from mods.LorenzXY.truncated import setup as setup_trunc
ratio_dt = validate_int(setup_trunc.t.dt / setup_full.t.dt)
#

BAMs = BAM_list()
BAMs.add(Climatology)
BAMs.add(D3Var)
BAMs.add(EnKF,infl=1.10,liveplotting=True)
keep = len(BAMs)-1
BAMs.add(EnKF,infl=1.15)
BAMs.add(EnKF,infl=1.20)

############################
# Common settings
############################
for method in BAMs:
  if method.base_methd is EnKF:
    method.N       = 7
    method.upd_a = 'Sqrt'
    method.rot     = True

############################
# Assimilate
############################
ss = np.empty(len(BAMs),dict)

xx,yy = simulate(setup_full)
xx    = xx[::ratio_dt,:setup_trunc.f.m]
for k,method in enumerate(BAMs):
  seed(sd0)
  stats = assimilate(setup_trunc,method,xx,yy)
  ss[k] = stats.average_in_time()
  if k == keep: kept = stats
print_averages(BAMs,ss)


############################
# Plot
############################
cfg    = BAMs[keep]
chrono = setup_trunc.t
plot_time_series(xx,kept,chrono,dim=2)
plot_ens_stats(xx,kept,chrono,cfg)
plot_3D_trajectory(xx[:,:3],kept,chrono)


