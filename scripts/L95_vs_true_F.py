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
from mods.Lorenz95.core import dxdt

def model(F):
  def wrapped(x0,t,dt):
    return rk4(lambda t,x: dxdt(x,F),x0,t,dt)
  return wrapped

F_DA    = 8.0
F_range = arange(5,12+1)

############################
# DA methods
############################

DAMs = DAM_list()
DAMs.add(Climatology)
DAMs.add(D3Var)
DAMs.add(ExtKF,infl=1.05)
DAMs.add(EnKF_N,N=24,rot=True)


############################
# Assimilate
############################
nRepeat = 2
ss = np.empty((len(F_range),nRepeat,len(DAMs)),dict)

for i,F_true in enumerate(F_range):
  print('\nF_true: ', F_true)
  for j in range(nRepeat):
    seed(sd0 + j)
    setup.f.model = model(F=F_true)
    xx,yy         = simulate(setup)
    setup.f.model = model(F=F_DA)
    for k,method in enumerate(DAMs):
      seed(sd0 + j)
      stats     = assimilate(setup,method,xx,yy)
      ss[i,j,k] = stats.average_after_burn()
    print_averages(DAMs,ss[i,j])
  avrg = average_each_field(ss[i],axis=0)
  print('\nAverage over',nRepeat,'repetitions:')
  print_averages(DAMs,avrg)

save_data(save_path,inds,F_range=F_range,ss=ss,xx=xx,yy=yy)
