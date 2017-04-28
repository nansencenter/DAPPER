############################
# Preamble
############################
from common import *

sd0 = seed(5)

############################
# Setup
############################
from mods.Barotropic.defaults  import setup
#

setup.t.T = 20

cfgs = DAC_list()
cfgs += Climatology()
cfgs += EnKF(infl=1.15,keep=True,upd_a='Sqrt')
cfgs += LETKF  (infl=1.10,locf=setup.locf(100,'x2y'),keep=True)
#cfgs += LETKF(infl=1.10,locf=setup.locf(10,'x2y'),upd_a='approx',keep=True)
#cfgs += SL_EAKF(infl=1.0,locf=setup.locf(10,'y2x'),keep=True)

############################
# Common settings
############################
for method in cfgs:
  method.N       = 20
  method.rot     = False

############################
# Assimilate
############################
ss   = np.empty(len(cfgs),dict)
kept = []

xx,yy = simulate(setup)
for k,method in enumerate(cfgs):
  seed(sd0)
  stats = assimilate(setup,method,xx,yy)
  ss[k] = stats.average_in_time()
  if getattr(method,'keep',False): kept.append(stats)
print_averages(cfgs,ss)


############################
# Plot
############################
k=0
for method in cfgs:
  if getattr(method,'keep',False):
    stats = kept[k]
    k += 1
    plot_time_series(xx,stats,setup.t,dim=2)
    plot_err_components(xx,stats,setup.t,method)
    plot_3D_trajectory(xx[:,:3],stats,setup.t)


