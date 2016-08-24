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

DAMs = DAM_list()
DAMs.add(Climatology)
DAMs.add(D3Var)
DAMs.add(EnKF,infl=1.10,liveplotting=True)
keep = len(DAMs)-1
DAMs.add(EnKF,infl=1.15)
DAMs.add(EnKF,infl=1.20)

def print_table(x):
  print_averages(DAMs,x, 'rmse_a','rmv_a','logp_m_a')

############################
# Common settings
############################
for method in DAMs:
  if method.da_method is EnKF:
    method.N       = 7
    method.AMethod = 'Sqrt'
    method.rot     = True

############################
# Assimilate
############################
ss = np.empty(len(DAMs),dict)

xx,yy = simulate(setup_full)
xx    = xx[::ratio_dt,:setup_trunc.f.m]
for k,method in enumerate(DAMs):
  seed(sd0)
  stats = assimilate(setup_trunc,method,xx,yy)
  ss[k] = stats.average_after_burn()
  if k == keep: kept = stats
print_table(ss)


############################
# Plot
############################
cfg    = DAMs[keep]
chrono = setup_trunc.t
plot_time_series(xx,kept,chrono,dim=2)
plot_ens_stats(xx,kept,chrono,cfg)
plot_3D_trajectory(xx[:,:3],kept,chrono)


