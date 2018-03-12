# Investigate the impact of ignoring all
# but the diagonal of the R matrix.

############################
# Preamble
############################
from common import *

from diag_R.da_methods_diagR import *

sd0 = seed(5)

############################
# Set-up
############################
from mods.Lorenz95.sak08 import *
setup.t.T = 100

# Non-block
R = inv(sla.circulant([-2,1] + [0]*37 + [1]) + 5*eye(40))
R = R * 0.6/R[0,0]

# Compose R of blocks
#block_R = inv(sla.circulant([-2,1] + [0]*7 + [1]) + 5*eye(10)) * 275/200
## Compose 4 blocks
#R = sla.block_diag(block_R, eye(10), block_R, eye(10))

# Define true R.
setup.h.noise.C = CovMat(R)

# Range of different experiments
dkObs_range = range(1,10)
dtObs_range = setup.t.dt*array(dkObs_range)


############################
# DA methods
############################
cfgs  = List_of_Configs()

cfgs += Climatology()
cfgs += OptInterp()
cfgs += Var3D()
cfgs += ExtKF()
cfgs += EnKF('PertObs',N=24)
cfgs += EnKF('DEnKF'  ,N=24)
cfgs += EnKF('Sqrt'   ,N=24,rot=True)
cfgs += EnKF_N        (N=24,rot=True)

cfgs += EnKF_N_diagR  (N=24,rot=True)
cfgs += DEnKF_diagR   (N=24)
#cfgs += DEnKF_sampleR (N=24, NR=400)
#cfgs += DEnKF_dsR(N=24, NR=10)

# (not well-) tuned inflation values.
# NB: If you change N, or rot, or any other setting, the tabulated values must be re-calibrated.
inflation_values = {
    'setting'      : [   1,    2,    3,    4,    5,    6,    7,    8,    9], # 'dkObs'
    'ExtKF'        : [  4.,   7.,   8.,  11.,  20.,  60.,  90.,  90.,  90.],
    'EnKF'         : [1.12, 1.30, 1.45, 1.55, 1.60, 1.65, 1.68, 1.72, 1.75],
    'Sqrt'         : [1.02, 1.06, 1.10, 1.12, 1.14, 1.18, 1.22, 1.25, 1.25],
    'DEnKF'        : [1.01, 1.04, 1.06, 1.08, 1.10, 1.14, 1.18, 1.22, 1.15],
    'DEnKF_diagR'  : [1.01, 1.04, 1.06, 1.08, 1.10, 1.14, 1.18, 1.22, 1.15],
    'DEnKF_sampleR': [1.01, 1.04, 1.06, 1.08, 1.10, 1.14, 1.18, 1.22, 1.15],
    'DEnKF_dsR'    : [1.01, 1.04, 1.06, 1.08, 1.10, 1.14, 1.18, 1.22, 1.15],
    }

def set_infl(dtObs):
  """Makeshift inflation setter.
  Interpolates the values in the table, from the correct config (row)."""
  table = inflation_values
  def interp(config,trait):
    return config.update_settings(infl=sp.interp(dtObs,table['setting'],table[trait]))
  for ic,config in enumerate(cfgs):
    trait1 = config.da_method.__name__
    trait2 = getattr(config,'upd_a','N/A')
    if   trait2 in table.keys(): cfgs[ic] = interp(config,trait2)
    elif trait1 in table.keys(): cfgs[ic] = interp(config,trait1)


############################
# Assimilate
############################
nRepeat = 5
avrgs = np.empty((len(dkObs_range),nRepeat,len(cfgs)),dict)
avrg2 = np.empty((len(dkObs_range),        len(cfgs)),dict)

for i,dkObs in enumerate(dkObs_range):
  setup.t.dkObs = dkObs
  set_infl(dkObs)
  print('\ndkObs: ', dkObs)
  for j in range(nRepeat):
    seed(sd0+j)
    xx,yy = simulate(setup)
    for ic,config in enumerate(cfgs):
      seed(sd0+j)
      stats         = config.assimilate(setup,xx,yy)
      avrgs[i,j,ic] = stats.average_in_time()
    print_averages(cfgs,avrgs[i,j])
  print_c('Average over',nRepeat,'repetitions:')
  avrg2[i] = average_each_field(avrgs[i],axis=0)
  print_averages(cfgs,avrg2[i])


############################
# Plot
############################

cfgs.assign_names()

plt.figure(1)
for ic,config in enumerate(cfgs):
  plt.plot(dtObs_range, [s['rmse_a'].val for s in avrg2[:,ic]], label=config.name)
plt.legend()
plt.ylabel('rmse_a')
plt.xlabel('dtObs')
    
