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

  HMM.t.BurnIn = HMM.t.dtObs
  HMM.t.KObs = 1
  xx,yy = HMM.simulate()

  cfgs.assimilate(HMM,xx,yy,sd=sd0,free=False,print=True,liveplots="all",store_u=False,fail_gently=False)
  cfgs.print_avrgs(['rmse_a'])

  for config in cfgs:
    replay(config,"all")

  replay(config, t2=1)
  replay(config, t2=0.0)
  replay(config, t2=0.3)
  replay(config, t2=0.8)
  replay(config, t2=0.8, t1=0.2)
  replay(config, t2=np.inf)
  replay(config, t2=np.inf, speed=1)
  replay(config, t2=np.inf, pause_a=0, pause_f=0)

  spell_out(HMM);
  spell_out(config);
  spell_out(config.stats);
  spell_out(config.avrgs)

  assert True # An assertion for pytest to count
  return HMM, xx, yy, cfgs # Return useful stuff



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

  # HMM.t.BurnIn = 10*HMM.t.dtObs
  # HMM.t.KObs = 30
  HMM.t.BurnIn = HMM.t.dtObs
  HMM.t.KObs = 2
  xx,yy = HMM.simulate()

  cfgs.assimilate(HMM,xx,yy,sd=sd0,free=False,print=True,liveplots="all",store_u=False,fail_gently=False)
  cfgs.print_avrgs(['rmse_a'])

  for config in cfgs:
    replay(config,"all")

  replay(config, t2=1)
  replay(config, t2=0.0)
  replay(config, t2=0.3)
  replay(config, t2=0.8)
  replay(config, t2=0.8, t1=0.2)
  replay(config, t2=np.inf)
  replay(config, t2=np.inf, speed=1)
  replay(config, t2=np.inf, pause_a=0, pause_f=0)

  print(HMM); print(config); print(config.stats); print(config.avrgs)

  assert True # An assertion for pytest to count
  return HMM, xx, yy, cfgs # Return useful stuff



# Non py.test runs:
# HMM, xx, yy, cfgs = test_L63()
# HMM, xx, yy, cfgs = test_L95()



