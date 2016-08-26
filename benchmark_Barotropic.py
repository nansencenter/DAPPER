############################
# Preamble
############################
from common import *

sd0 = 5
np.random.seed(sd0)

############################
# Setup
############################
from mods.Barotropic.defaults  import setup
#

DAMs = DAM_list()
#DAMs.add(Climatology)
#DAMs.add(D3Var)
DAMs.add(EnKF,infl=1.10)

def print_table(x):
  print_averages(DAMs,x, 'rmse_a','rmv_a','logp_m_a')

############################
# Common settings
############################
for method in DAMs:
  if method.da_method is EnKF:
    method.N       = 20
    method.AMethod = 'Sqrt'
    method.rot     = True

############################
# Assimilate
############################
ss = np.empty(len(DAMs),dict)

xx,yy = simulate(setup)
for k,method in enumerate(DAMs):
  seed(sd0)
  stats = assimilate(setup,method,xx,yy)
  ss[k] = stats.average_after_burn()
print_table(ss)


############################
# Plot
############################
cfg    = DAMs[keep]
chrono = setup.t
plot_time_series(xx,kept,chrono,dim=2)
plot_ens_stats(xx,kept,chrono,cfg)
plot_3D_trajectory(xx[:,:3],kept,chrono)


