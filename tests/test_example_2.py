# Just stupidly compare the full results table.
# => seed-dependent test ==> cannot be run with different seeds or computers.

from dapper import *

statkeys = ['err.rms.a','err.rms.f','err.rms.u']

##############################
# L63
##############################
from dapper.mods.Lorenz63.sakov2012 import HMM
HMM.t.BurnIn=0
HMM.t.KObs=10
sd0 = set_seed(9)

# Cfgs
cfgs  = xpList()
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
cfgs.launch(HMM,sd0,store_u=True)

table = cfgs._repr_avrgs(statkeys,decimals=4)
old = """
      da_method     infl  upd_a       N  rot      xN  reg   NER  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
----  -----------  -----  -------  ----  -----  ----  ---  ----  -  -----------------  -----------------  -----------------
[0]   Climatology                                                |     8.2648 ±1.2554     8.2648 ±1.2554     7.9296 ±2.6645
[1]   OptInterp                                                  |     1.2649 ±0.183      8.2648 ±1.2554     1.325  ±0.1199
[2]   Var3D                                                      |     1.1683 ±0.2147     2.5356 ±1.3849     1.5462 ±0.3792
[3]   ExtKF        90                                            |     0.9001 ±0.1926     1.8596 ±0.7186     1.1517 ±0.224
[4]   EnKF          1.3   Sqrt        3  False                   |     0.7501 ±0.2121     1.6799 ±0.4566     0.9677 ±0.1783
[5]   EnKF          1.02  Sqrt       10  True                    |     0.6613 ±0.2295     1.3384 ±0.3265     0.8094 ±0.1654
[6]   EnKF          0.95  PertObs   500  False                   |     0.7036 ±0.2589     1.2653 ±0.3502     0.7958 ±0.1831
[7]   EnKF_N        1                10  True      1             |     0.6864 ±0.2409     1.3623 ±0.3144     0.8375 ±0.1723
[8]   iEnKS         1.02  Sqrt       10  True                    |     0.4439 ±0.1393     1.117  ±0.3303     0.4752 ±0.3346
[9]   PartFilt                      100               2.4  0.3   |     0.6794 ±0.1911     1.8088 ±0.7605     1.0475 ±0.2447
[10]  PartFilt                      800               0.9  0.2   |     0.4011 ±0.1163     0.8615 ±0.2759     0.5307 ±0.1264
[11]  PartFilt                     4000               0.7  0.05  |     0.4251 ±0.1316     0.9114 ±0.3036     0.5487 ±0.0995
[12]  PFxN                           30         1000       0.2   |     0.5521 ±0.1617     1.2982 ±0.2372     0.8613 ±0.1565
"""[1:-1]

def test_len():
  assert len(old)==len(table)

table = [row.rstrip() for row in table.splitlines()]
old   = [row.rstrip() for row in old  .splitlines()]

L63 = dict(table=table,old=old)

##############################
# L96
##############################
from dapper.mods.Lorenz96.sakov2008 import HMM
HMM.t.BurnIn=0
HMM.t.KObs=10
sd0 = set_seed(9)

# Cfgs
cfgs  = xpList()
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


cfgs.launch(HMM,sd0,store_u=True)

table = cfgs._repr_avrgs(statkeys,decimals=4)
old = """
      da_method    infl  upd_a     N  rot    xN  loc_rad  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
----  -----------  ----  -------  --  -----  --  -------  -  -----------------  -----------------  -----------------
[0]   Climatology                                         |     0.8335 ±0.2327     0.8335 ±0.2327     0.8335 ±0.2327
[1]   OptInterp                                           |     0.1306 ±0.0288     0.8335 ±0.2327     0.1306 ±0.0288
[2]   Var3D                                               |     0.0702 ±0.0103     0.0655 ±0.0101     0.0702 ±0.0103
[3]   ExtKF        6                                      |     0.0305 ±0.0009     0.0306 ±0.001      0.0305 ±0.0009
[4]   EnKF         1.06  PertObs  40  False               |     0.0306 ±0.0009     0.0307 ±0.0009     0.0306 ±0.0009
[5]   EnKF         1.02  Sqrt     28  True                |     0.0309 ±0.0009     0.0309 ±0.0009     0.0309 ±0.0009
[6]   EnKF_N       1              24  True    1           |     0.0313 ±0.001      0.0313 ±0.0009     0.0313 ±0.001
[7]   EnKF_N       1              24  True    2           |     0.0313 ±0.001      0.0313 ±0.0009     0.0313 ±0.001
[8]   iEnKS        1.01  Sqrt     40  True                |     0.0307 ±0.0009     0.0307 ±0.0009     0.0306 ±0.0009
[9]   LETKF        1.04            7  True    1        4  |     0.0334 ±0.0014     0.0332 ±0.0012     0.0334 ±0.0014
[10]  SL_EAKF      1.07            7  True             6  |     0.0337 ±0.0017     0.0333 ±0.0013     0.0337 ±0.0017
"""[1:-1]

table = [row.rstrip() for row in table.splitlines()]
old   = [row.rstrip() for row in old  .splitlines()]

L96 = dict(table=table,old=old)



##############################
# Test definitions
##############################

import pytest
@pytest.mark.parametrize(('lineno'),arange(len(L63['table'])))
def test_tables_L63(lineno):
    assert L63['table'][lineno] == L63['old'][lineno]


@pytest.mark.parametrize(('lineno'),arange(len(L96['table'])))
def test_tables_L96(lineno):
    assert L96['table'][lineno] == L96['old'][lineno]


