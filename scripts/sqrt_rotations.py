# Test how frequently/much to rotate after sqrt update.
# It seems indeed that "some" rotation is useful,
# but that there's rarely much to gain by tuning it.

from common import *

sd0 = seed(9)

############################
# DA Configurations
############################
cfgs  = List_of_Configs()

#from mods.Lorenz63.sak12 import setup
#cfgs += EnKF('Sqrt',N=6,infl=1.08,rot=False)
#cfgs += EnKF('Sqrt',N=6,infl=1.08,rot=True)
#cfgs += EnKF('Sqrt',N=20,infl=1.01,rot=True)
#cfgs += EnKF('Sqrt',N=20,infl=1.01,rot=False)
#cfgs += EnKF('Sqrt',N=20,infl=1.01,rot=0.7)
#cfgs += EnKF('Sqrt',N=20,infl=1.01,rot=(2,0.6))
#cfgs += EnKF_N(N=20,rot=False)
#cfgs += EnKF_N(N=20,rot=0.7)
#cfgs += EnKF_N(N=20,rot=True)


from mods.Lorenz95.sak08 import setup
setup.t.dkObs = 3
cfgs += EnKF('Sqrt',N=30 ,infl=1.08,rot=False)
cfgs += EnKF('Sqrt',N=30 ,infl=1.10,rot=True)
cfgs += EnKF('Sqrt',N=200,infl=1.03,rot=False)
cfgs += EnKF('Sqrt',N=200,infl=1.03,rot=True)
cfgs += EnKF('Sqrt',N=200,infl=1.03,rot=0.7)
cfgs += EnKF('Sqrt',N=200,infl=1.03,rot=(2,0.7))


############################
# Assimilate
############################
setup.t.T = 4**5
xx,yy = simulate(setup)

stats = []
avrgs = []

for ic,config in enumerate(cfgs):
  #config.liveplotting = True
  seed(sd0+2)

  stats += [ config.assimilate(setup,xx,yy) ]
  avrgs += [ stats[ic].average_in_time() ]
  print_averages(config, avrgs[-1])
print_averages(cfgs,avrgs)

