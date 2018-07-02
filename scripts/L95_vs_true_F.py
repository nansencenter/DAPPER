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

F_DA   = 8.0
xticks = arange(6,12) # Forcing F

setup.t.T = 4**4.5
nRepeat   = 8

############################
# DA methods
############################
cfgs = List_of_Configs()

cfgs += Climatology()
cfgs += Var3D()
cfgs += ExtKF(infl=6)
cfgs += EnKF_N(N=25,rot=False)

#cfgs.assign_names()

############################
# Assimilate
############################
avrgs = np.empty((len(xticks),nRepeat,len(cfgs)),dict)
stats = np.empty_like(avrgs)

for iX,X in enumerate(xticks):
  print_c('\nF_true: ', X)
  for iR in range(nRepeat):
    seed(sd0+iR)
    L95.Force = X
    xx,yy     = simulate(setup)
    L95.Force = F_DA
    for iC,Config in enumerate(cfgs):
      seed(sd0+iR)
      stats[iX,iR,iC] = Config.assimilate(setup,xx,yy)
      avrgs[iX,iR,iC] = stats[iX,iR,iC].average_in_time()
    print_averages(cfgs,avrgs[iX,iR])
  avrg = average_each_field(avrgs[iX],axis=0)
  print_c('Average over',nRepeat,'repetitions:')
  print_averages(cfgs,avrg)

#save_data(save_path,inds,xticks=xticks,stng_var=stng_var,avrgs=avrgs,xx=xx,yy=yy)
