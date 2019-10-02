# Just stupidly compare the full results table.
# => seed-dependent test ==> cannot be run with different seeds or computers.

from dapper import *

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
for c in cfgs: c.store_u=True
xx,yy = simulate(HMM)
cfgs.assimilate(HMM,xx,yy,sd0+2)

table = cfgs._repr_avrgs(['rmse_a','rmse_f','rmse_u'],decimals=4)
old = """
      da_method     infl  upd_a       N  rot      xN   NER  reg  |  rmse_a  1σ   rmse_f  1σ   rmse_u  1σ
----  -----------  -----  -------  ----  -----  ----  ----  ---  -  -----------  -----------  -----------
[0]   Climatology                                                |  8.2648 ±1    8.2648 ±1    7.9296 ±3
[1]   OptInterp                                                  |  1.2649 ±0.2  8.2648 ±1    1.325  ±0.1
[2]   Var3D                                                      |  1.1683 ±0.2  2.5356 ±1    1.5462 ±0.4
[3]   ExtKF        90                                            |  0.9001 ±0.2  1.8596 ±0.7  1.1517 ±0.2
[4]   EnKF          1.3   Sqrt        3  False                   |  0.8368 ±0.2  1.5137 ±0.4  0.9236 ±0.1
[5]   EnKF          1.02  Sqrt       10  True                    |  0.7257 ±0.2  1.3002 ±0.3  0.8452 ±0.2
[6]   EnKF          0.95  PertObs   500  False                   |  0.713  ±0.3  1.2887 ±0.3  0.8216 ±0.2
[7]   EnKF_N        1                10  True      1             |  0.7471 ±0.2  1.3167 ±0.2  0.8661 ±0.2
[8]   iEnKS         1.02  Sqrt       10  True                    |  0.4521 ±0.1  0.9145 ±0.3  0.4997 ±0.3
[9]   PartFilt                      100               0.3   2.4  |  0.6182 ±0.1  1.401  ±0.7  0.8489 ±0.1
[10]  PartFilt                      800               0.2   0.9  |  0.4684 ±0.2  0.9491 ±0.3  0.587  ±0.1
[11]  PartFilt                     4000               0.05  0.7  |  0.4317 ±0.1  0.815  ±0.3  0.5209 ±0.1
[12]  PFxN                           30         1000  0.2        |  0.862  ±0.2  1.8095 ±0.5  1.1553 ±0.2
"""[1:-1]

def test_len():
  assert len(old)==len(table)

table = [row.rstrip() for row in table.splitlines()]
old   = [row.rstrip() for row in old  .splitlines()]

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


for c in cfgs: c.store_u=True
xx,yy = simulate(HMM)
cfgs.assimilate(HMM,xx,yy,sd0+2)

table = cfgs._repr_avrgs(['rmse_a','rmse_f','rmse_u'],decimals=4)
old = """
      da_method    infl  upd_a     N  rot    xN  loc_rad  |  rmse_a  1σ      rmse_f  1σ     rmse_u  1σ
----  -----------  ----  -------  --  -----  --  -------  -  --------------  -------------  --------------
[0]   Climatology                                         |  0.8335 ±0.2     0.8335 ±0.2    0.8335 ±0.2
[1]   OptInterp                                           |  0.1306 ±0.03    0.8335 ±0.2    0.1306 ±0.03
[2]   Var3D                                               |  0.0702 ±0.01    0.0655 ±0.01   0.0702 ±0.01
[3]   ExtKF        6                                      |  0.0305 ±0.0009  0.0306 ±0.001  0.0305 ±0.0009
[4]   EnKF         1.06  PertObs  40  False               |  0.0311 ±0.001   0.031  ±0.001  0.0311 ±0.001
[5]   EnKF         1.02  Sqrt     28  True                |  0.0308 ±0.001   0.0307 ±0.001  0.0308 ±0.001
[6]   EnKF_N       1              24  True    1           |  0.0311 ±0.001   0.0311 ±0.001  0.0311 ±0.001
[7]   EnKF_N       1              24  True    2           |  0.0311 ±0.001   0.0311 ±0.001  0.0311 ±0.001
[8]   iEnKS        1.01  Sqrt     40  True                |  0.0309 ±0.001   0.0309 ±0.001  0.0309 ±0.001
[9]   LETKF        1.04            7  True    1        4  |  0.0313 ±0.001   0.0312 ±0.001  0.0313 ±0.001
[10]  SL_EAKF      1.07            7  True             6  |  0.0317 ±0.001   0.0316 ±0.001  0.0317 ±0.001
"""[1:-1]

table = [row.rstrip() for row in table.splitlines()]
old   = [row.rstrip() for row in old  .splitlines()]

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


