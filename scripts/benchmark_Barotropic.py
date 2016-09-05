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

setup.t.T = 20

DAMs = DAM_list()
DAMs.add(Climatology)
DAMs.add(EnKF,infl=1.15,keep=True,AMethod='Sqrt')
DAMs.add(LETKF  ,infl=1.10,locf=setup.locf(100,'x2y'),keep=True)
#DAMs.add(LETKF,infl=1.10,locf=setup.locf(10,'x2y'),AMethod='approx',keep=True)
#DAMs.add(SL_EAKF,infl=1.0,locf=setup.locf(10,'y2x'),keep=True)

############################
# Common settings
############################
for method in DAMs:
  method.N       = 20
  method.rot     = False

############################
# Assimilate
############################
ss   = np.empty(len(DAMs),dict)
kept = []

xx,yy = simulate(setup)
for k,method in enumerate(DAMs):
  seed(sd0)
  stats = assimilate(setup,method,xx,yy)
  ss[k] = stats.average_after_burn()
  if getattr(method,'keep',False): kept.append(stats)
print_averages(DAMs,ss)


############################
# Plot
############################
k=0
for method in DAMs:
  if getattr(method,'keep',False):
    stats = kept[k]
    k += 1
    plot_time_series(xx,stats,setup.t,dim=2)
    plot_ens_stats(xx,stats,setup.t,method)
    plot_3D_trajectory(xx[:,:3],stats,setup.t)


