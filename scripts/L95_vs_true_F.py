# Experiment

############################
# Preamble
############################
from common import *

sd0 = seed(5)

############################
# Set-up
############################
from mods.Lorenz95.sak08 import setup
import mods.Lorenz95.core as L95

#def model(F):
  #def wrapped(x0,t,dt):
    #return rk4(lambda t,x: L95.dxdt(x,F),x0,t,dt)
  #return wrapped

F_DA    = 8.0
F_range = arange(8,10)

setup.t.T = 4**2.5
nRepeat   = 2

############################
# DA methods
############################

cfgs = DAC_list()
cfgs.add(Climatology)
cfgs.add(D3Var)
cfgs.add(ExtKF,infl=6)
cfgs.add(EnKF_N,N=24,rot=True)

# Not tuned
#cfgs.add(EnKF,'PertObs',N=24,infl=1.25)
#cfgs.add(EnKF,'Sqrt',N=24,infl=1.25)
#cfgs.add(EnKF_tp,N=24,infl=1.25)

#cfgs.assign_names()

############################
# Assimilate
############################
ss = np.empty((len(F_range),nRepeat,len(cfgs)),dict)

for i,F_true in enumerate(F_range):
  print_c('\nF_true: ', F_true)
  for j in range(nRepeat):
    seed(sd0 + j)
    L95.Force = F_true
    xx,yy     = simulate(setup)
    L95.Force = F_DA
    for k,method in enumerate(cfgs):
      seed(sd0 + j)
      stats     = assimilate(setup,method,xx,yy)
      ss[i,j,k] = stats.average_in_time()
    print_averages(cfgs,ss[i,j])
  avrg = average_each_field(ss[i],axis=0)
  print_c('Average over',nRepeat,'repetitions:')
  print_averages(cfgs,avrg)

#save_data(save_path,inds,F_range=F_range,ss=ss,xx=xx,yy=yy)
