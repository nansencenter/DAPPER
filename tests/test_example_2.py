# Just stupidly compare the full results table.
# => seed-dependent test ==> cannot be run with different seeds or computers.

from dapper import *

# Table string creator
from dapper.admin import _print_averages
def pa(cfgs,avrgs):
  return _print_averages(cfgs,avrgs,statkeys=['rmse_a','rmse_f','rmse_u'])


##############################
# L63
##############################
from dapper.mods.Lorenz63.sak12 import HMM
HMM.t.BurnIn=0
HMM.t.KObs=10
sd0 = seed(9)

# Cfgs
cfgs  = List_of_Configs()
cfgs += Climatology()
cfgs += OptInterp()
cfgs += Var3D(xB=0.1)
cfgs += ExtKF(infl=90)
cfgs += EnKF('Sqrt',    N=3 ,  infl=1.30)
cfgs += EnKF('Sqrt',    N=10,  infl=1.02,rot=True)
cfgs += EnKF('PertObs', N=500, infl=0.95,rot=False)
cfgs += EnKF_N(         N=10,            rot=True)
cfgs += iEnKS('Sqrt',   N=10,  infl=1.02,rot=True)
cfgs += PartFilt(       N=100 ,reg=2.4  ,NER=0.3)
cfgs += PartFilt(       N=800 ,reg=0.9  ,NER=0.2)
cfgs += PartFilt(       N=4000,reg=0.7  ,NER=0.05)
cfgs += PFxN(xN=1000,   N=30  ,Qs=2     ,NER=0.2)

# Run
xx,yy = simulate(HMM)
avrgs = []
for ic,config in enumerate(cfgs):
  config.liveplotting = False
  config.store_u = True
  seed(sd0+2)
  stats = config.assimilate(HMM,xx,yy)
  avrgs += [ stats.average_in_time() ]

table = pa(cfgs,avrgs)
old = """
      da_method       N  upd_a     infl  rot     NER  Qs  reg   xB    xN  |  rmse_a ±    rmse_f ±    rmse_u ±
----  -----------  ----  -------  -----  -----  ----  --  ---  ---  ----  -  ----------  ----------  ----------
[0]   Climatology                                                         |   8.265 1     8.265 1      7.93 3
[1]   OptInterp                                                           |   1.265 0.2   8.265 1     1.325 0.1
[2]   Var3D                                                    0.1        |   1.168 0.2   2.536 1     1.546 0.4
[3]   ExtKF                       90                                      |  0.9001 0.2    1.86 0.7   1.152 0.2
[4]   EnKF            3  Sqrt      1.3                                    |  0.8368 0.2   1.514 0.4  0.9236 0.1
[5]   EnKF           10  Sqrt      1.02  True                             |  0.7257 0.2     1.3 0.3  0.8452 0.2
[6]   EnKF          500  PertObs   0.95  False                            |   0.713 0.3   1.289 0.3  0.8216 0.2
[7]   EnKF_N         10                  True                             |  0.7471 0.2   1.317 0.2  0.8661 0.2
[8]   iEnKS          10  Sqrt      1.02  True                             |  0.4521 0.1  0.9145 0.3  0.4997 0.3
[9]   PartFilt      100                         0.3       2.4             |  0.6182 0.1   1.401 0.7  0.8489 0.1
[10]  PartFilt      800                         0.2       0.9             |  0.4684 0.2  0.9491 0.3   0.587 0.1
[11]  PartFilt     4000                         0.05      0.7             |  0.4317 0.1   0.815 0.3  0.5209 0.1
[12]  PFxN           30                         0.2    2            1000  |   0.862 0.2   1.809 0.5   1.155 0.2
"""[1:-1]

def test_len():
  assert len(old)==len(table)

table = table.split('\n')
old   = old  .split('\n')

L63 = dict(table=table,old=old)


##############################
# L95
##############################
from dapper.mods.Lorenz95.sak08 import HMM
HMM.t.BurnIn=0
HMM.t.KObs=10
sd0 = seed(9)

# Cfgs
cfgs  = List_of_Configs()
cfgs += Climatology()
cfgs += OptInterp()
cfgs += Var3D(xB=0.02)
cfgs += ExtKF(infl=6)
cfgs += EnKF('PertObs'        ,N=40,infl=1.06)
cfgs += EnKF('Sqrt'           ,N=28,infl=1.02,rot=True)

cfgs += EnKF_N(N=24,rot=True)
cfgs += EnKF_N(N=24,rot=True,xN=2)
cfgs += iEnKS('Sqrt',N=40,infl=1.01,rot=True)

cfgs += LETKF(         N=7,rot=True,infl=1.04,loc_rad=4)
cfgs += SL_EAKF(       N=7,rot=True,infl=1.07,loc_rad=6)


# Run
xx,yy = simulate(HMM)
avrgs = []
for ic,config in enumerate(cfgs):
  config.store_u = True
  config.liveplotting = False
  seed(sd0+2)
  stats = config.assimilate(HMM,xx,yy)
  avrgs += [ stats.average_in_time() ]

table = pa(cfgs,avrgs)
old = """
      da_method     N  upd_a    infl  rot   loc_rad    xB  xN  |   rmse_a ±       rmse_f ±       rmse_u ±
----  -----------  --  -------  ----  ----  -------  ----  --  -  -------------  -------------  -------------
[0]   Climatology                                              |   0.5954 0.2     0.5954 0.2     0.5954 0.2
[1]   OptInterp                                                |   0.1276 0.03    0.5954 0.2     0.1276 0.03
[2]   Var3D                                          0.02      |  0.06879 0.006  0.06584 0.007  0.06879 0.006
[3]   ExtKF                     6                              |  0.04828 0.01    0.0474 0.008  0.04828 0.01
[4]   EnKF         40  PertObs  1.06                           |  0.04745 0.009  0.04719 0.008  0.04745 0.009
[5]   EnKF         28  Sqrt     1.02  True                     |  0.04771 0.009  0.04783 0.009  0.04771 0.009
[6]   EnKF_N       24                 True                     |  0.04815 0.009  0.04835 0.009  0.04815 0.009
[7]   EnKF_N       24                 True                  2  |  0.04815 0.009  0.04835 0.009  0.04815 0.009
[8]   iEnKS        40  Sqrt     1.01  True                     |  0.04707 0.008  0.04728 0.008  0.04686 0.008
[9]   LETKF         7           1.04  True        4            |  0.04857 0.01   0.04836 0.009  0.04857 0.01
[10]  SL_EAKF       7           1.07  True        6            |   0.0499 0.01   0.04894 0.009   0.0499 0.01
"""[1:-1]

table = table.split('\n')
old   = old  .split('\n')

L95 = dict(table=table,old=old)



##############################
# Test definitions
##############################

import pytest
@pytest.mark.parametrize(('lineno'),arange(len(L63['table'])))
def test_tables_L63(lineno):
    assert L63['table'][lineno] == L63['old'][lineno]

@pytest.mark.parametrize(('lineno'),arange(len(L95['table'])))
def test_tables_L95(lineno):
    assert L95['table'][lineno] == L95['old'][lineno]


