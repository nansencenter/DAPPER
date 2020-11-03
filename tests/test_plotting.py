# Test graphics/plotting.
# This won't automatically verify if the plots are correct,
# only whether they cause errors or not.

from dapper import *
import dapper as dpr
from dapper.tools.liveplotting import replay

import numpy as np
import dapper.tools.utils
tools.utils.disable_user_interaction = True # NB remember to set to True


def test_L63():
  from dapper.mods.Lorenz63.sakov2012 import HMM

  xps  = dpr.xpList()
  xps += EnKF('Sqrt',   N=10 ,infl=1.02 ,rot=True)
  xps += PartFilt(      N=20 ,reg=2.4   ,NER=0.3)
  xps += OptInterp()
  # xps += iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)

  HMM.t.BurnIn = HMM.t.dtObs
  HMM.t.KObs = 1

  xps.launch(HMM,free=False,liveplots="all",store_u=False,fail_gently=False)

  for xp in xps:
      replay(xp.stats,"all")

  replay(xp.stats, t2=1)
  replay(xp.stats, t2=0.0)
  replay(xp.stats, t2=0.3)
  replay(xp.stats, t2=0.8)
  replay(xp.stats, t2=0.8, t1=0.2)
  replay(xp.stats, t2=np.inf)
  replay(xp.stats, t2=np.inf, speed=1)
  replay(xp.stats, t2=np.inf, pause_a=0, pause_f=0)

  assert True # An assertion for pytest to count
  return HMM, xps # Return useful stuff



def test_L96():
  xps  = xpList()

  from dapper.mods.Lorenz96.sakov2008 import HMM
  xps += EnKF('PertObs'        ,N=40, infl=1.06)
  xps += EnKF('Serial'         ,N=28, infl=1.02,rot=True)
  xps += OptInterp()
  xps += Var3D(xB=0.02)
  xps += ExtKF(infl=10)
  xps += LETKF(N=6,rot=True,infl=1.05,loc_rad=4,taper='Step')

  # from dapper.mods.Lorenz96.bocquet2015loc import HMM
  # xps += EnKF_N(  N=24, rot=True ,infl=1.01)
  # xps += PartFilt(N=3000,NER=0.20,reg=1.2)
  # xps += PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)

  # HMM.t.BurnIn = 10*HMM.t.dtObs
  # HMM.t.KObs = 30
  HMM.t.BurnIn = HMM.t.dtObs
  HMM.t.KObs = 2

  xps.launch(HMM,free=False,liveplots="all",store_u=False,fail_gently=False, save_as=False)

  for xp in xps:
      replay(xp.stats,"all")

  replay(xp.stats, t2=1)
  replay(xp.stats, t2=0.0)
  replay(xp.stats, t2=0.3)
  replay(xp.stats, t2=0.8)
  replay(xp.stats, t2=0.8, t1=0.2)
  replay(xp.stats, t2=np.inf)
  replay(xp.stats, t2=np.inf, speed=1)
  replay(xp.stats, t2=np.inf, pause_a=0, pause_f=0)

  assert True # An assertion for pytest to count
  return HMM, xps # Return useful stuff



# Non py.test runs:
# HMM, xps = test_L63()
# HMM, xps = test_L96()



