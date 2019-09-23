# Test graphics/plotting.
# This won't automatically verify if the plots are correct,
# only whether they cause errors or not.

from dapper import *
sd0 = seed_init(3)

import dapper.tools.utils
tools.utils.disable_user_interaction = True # NB remember to set to True


def test_L63():
  from dapper.mods.Lorenz63.sak12 import HMM

  cfgs  = List_of_Configs()
  cfgs += EnKF('Sqrt',   N=10 ,infl=1.02 ,rot=True)
  cfgs += PartFilt(      N=20 ,reg=2.4   ,NER=0.3)
  cfgs += OptInterp()
  # cfgs += iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)

  for iC,C in enumerate(cfgs):
    C.fail_gently=False
    C.store_u=False
    C.liveplotting="all"

  HMM.t.BurnIn = HMM.t.dtObs
  HMM.t.KObs = 2
  xx,yy = simulate(HMM)

  stats = []
  avrgs = []

  for ic,config in enumerate(cfgs):
    seed(sd0+2)

    stats += [ config.assimilate(HMM,xx,yy) ]
    avrgs += [ stats[ic].average_in_time() ]
    print_averages(config, avrgs[-1])

  print_averages(cfgs,avrgs,statkeys=['rmse_a'])

  for s in stats:
    replay(s,"all")
  replay(stats[-1], t2=1)
  replay(stats[-1], t2=0.0)
  replay(stats[-1], t2=0.3)
  replay(stats[-1], t2=0.8)
  replay(stats[-1], t2=0.8, t1=0.2)
  replay(stats[-1], t2=np.inf)
  replay(stats[-1], t2=np.inf, speed=1)
  replay(stats[-1], t2=np.inf, pause_a=0, pause_f=0)

  print(HMM); print(config); print(stats); print(avrgs)
  assert True
  return HMM, xx, yy, cfgs, stats, avrgs



def test_L95():
  cfgs  = List_of_Configs()

  from dapper.mods.Lorenz95.sak08 import HMM
  cfgs += EnKF('PertObs'        ,N=40, infl=1.06)               # 0.22
  cfgs += EnKF('Serial'         ,N=28, infl=1.02,rot=True)      # 0.18
  cfgs += OptInterp()
  cfgs += Var3D(xB=0.02)
  cfgs += ExtKF(infl=10)                                        # 0.24 
  cfgs += LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')  # 

  # from dapper.mods.Lorenz95.boc15loc import HMM
  # cfgs += EnKF_N(  N=24, rot=True ,infl=1.01)              # 0.38
  # cfgs += PartFilt(N=3000,NER=0.20,reg=1.2)                # 0.77
  # cfgs += PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)          # 0.51

  for iC,C in enumerate(cfgs):
    C.fail_gently=False
    C.store_u=True
    C.liveplotting="all"

  HMM.t.BurnIn = 10*HMM.t.dtObs
  HMM.t.KObs = 30
  # HMM.t.BurnIn = HMM.t.dtObs
  # HMM.t.KObs = 2
  xx,yy = simulate(HMM)

  stats = []
  avrgs = []

  for ic,config in enumerate(cfgs):
    seed(sd0+2)

    stats += [ config.assimilate(HMM,xx,yy) ]
    avrgs += [ stats[ic].average_in_time() ]
    print_averages(config, avrgs[-1])

  print_averages(cfgs,avrgs,statkeys=['rmse_a'])

  for s in stats:
    replay(s,"all")
  replay(stats[-1], t2=1)
  replay(stats[-1], t2=0.0)
  replay(stats[-1], t2=0.3)
  replay(stats[-1], t2=0.8)
  replay(stats[-1], t2=0.8, t1=0.2)
  replay(stats[-1], t2=np.inf)
  replay(stats[-1], t2=np.inf, speed=1)
  replay(stats[-1], t2=np.inf, pause_a=0, pause_f=0)

  print(HMM); print(config); print(stats); print(avrgs)
  assert True
  return HMM, xx, yy, cfgs, stats, avrgs



# Non py.test runs:
# HMM, xx, yy, cfgs, stats, avrgs = test_L63()
# HMM, xx, yy, cfgs, stats, avrgs = test_L95()



