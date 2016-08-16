# Experiment

############################
# Preamble
############################
from common import *

sd0 = 5
seed(sd0)

############################
# Set-up
############################
from mods.Lorenz95.sak08 import setup
from mods.Lorenz95.fundamentals import dxdt

def model(F):
  def model_inner(x0,t,dt):
    return rk4(lambda t,x: dxdt(x,F),x0,t,dt)
  return model_inner

setup.t.T = 4**3

F_DA    = 8.0
F_range = arange(5,12+1)
F_range = arange(8,9)

############################
# DA methods
############################

DAMs = DAM_list()
DAMs.add(Climatology)
DAMs.add(D3Var)
DAMs.add(ExtKF,infl=1.05)
DAMs.add(EnKF_N,N=24,rot=True)

print_table = lambda x: print_averages(DAMs,x,
    'rmse_a','rmv_a','logp_m_a')


############################
# Assimilate
############################
nRepeat = 3
ss = np.empty((len(F_range),nRepeat,len(DAMs)),dict)

for i,F_true in enumerate(F_range):
  print('\nF_true: ', F_true)
  for j in range(nRepeat):
    seed(sd0 + j)
    setup.f.model = model(F=F_true)
    xx,yy          = simulate(setup)
    setup.f.model = model(F=F_DA)
    for k,cfg in enumerate(DAMs):
      seed(sd0 + j)
      stats     = assimilate(setup,cfg,xx,yy)
      ss[i,j,k] = stats.average_after_burn()
    print_table(ss[i,j])
  avrg = average_each_field(ss[i],axis=0)
  print('')
  print_table(avrg)

