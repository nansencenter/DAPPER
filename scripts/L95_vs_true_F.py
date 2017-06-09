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

F_DA    = 8.0
F_range = arange(6,12)

setup.t.T = 4**3.5
nRepeat   = 1

############################
# DA methods
############################
cfgs = List_of_Configs()

cfgs += Climatology()
cfgs += Var3D()
#cfgs += ExtKF(infl=6)
cfgs += EnKF_N    (N=24,rot=True)
cfgs += EnKF_AdInf(N=24,rot=True,Nb=1000,Fb=(1-1/40/6),br=0.01)
cfgs += EnKF_AdInf(N=24,rot=True,Nb=1000,Fb=(1-1/40/20),br=0.01)
cfgs += EnKF_AdInf(N=24,rot=True,Nb=100 ,Fb=(1-1/40/6),br=0.01)
cfgs += EnKF_AdInf(N=24,rot=True,Nb=1000,Fb=(1-1/40/6),br=0.05)
cfgs += EnKF_AdInf(N=24,rot=True,Nb=1000,Fb=(1-1/40/6),br=0.005)

# Not tuned
#cfgs.assign_names()

############################
# Assimilate
############################
avrgs = np.empty((len(F_range),nRepeat,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for i,F_true in enumerate(F_range):
  print_c('\nF_true: ', F_true)
  for j in range(nRepeat):
    seed(sd0+j)
    L95.Force = F_true
    xx,yy     = simulate(setup)
    L95.Force = F_DA
    for k,config in enumerate(cfgs):
      seed(sd0+j)
      stats[i,j,k] = config.assimilate(setup,xx,yy)
      avrgs[i,j,k] = stats[i,j,k].average_in_time()
    print_averages(cfgs,avrgs[i,j])
  avrg = average_each_field(avrgs[i],axis=0)
  print_c('Average over',nRepeat,'repetitions:')
  print_averages(cfgs,avrg)

#save_data(save_path,inds,F_range=F_range,avrgs=avrgs,xx=xx,yy=yy)
