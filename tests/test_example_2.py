"""Stupidly compare the full results table.

Use `pytest -vv tests/test_example_2.py` for a better diff when tests fail.

Possible reasons for failing:
- Random number generation might change on different versions/platforms.
- pytest imports some other Lorenz63/module, which modifies the Forcing param,
  or the HMM.tseq params, or something else.
"""

import pytest

import dapper as dpr
import dapper.da_methods as da

statkeys = ["err.rms.a", "err.rms.f", "err.rms.u"]


##############################
# L63
##############################
@pytest.fixture(scope="module")
def L63_table():
    xps = L63_gen()
    table = xps.tabulate_avrgs(statkeys, decimals=4, colorize=False)
    return table.splitlines(True)


def L63_gen():
    from dapper.mods.Lorenz63.sakov2012 import HMM as _HMM

    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 10

    dpr.set_seed(3000)

    # xps
    xps = dpr.xpList()
    xps += da.Climatology()
    xps += da.OptInterp()
    xps += da.Var3D(xB=0.1)
    xps += da.ExtKF(infl=90)
    xps += da.EnKF("Sqrt", N=3, infl=1.30)
    xps += da.EnKF("Sqrt", N=10, infl=1.02, rot=True)
    xps += da.EnKF("PertObs", N=500, infl=0.95, rot=False)
    xps += da.EnKF_N(N=10, rot=True)
    xps += da.iEnKS("Sqrt", N=10, infl=1.02, rot=True)
    xps += da.PartFilt(N=100, reg=2.4, NER=0.3)
    xps += da.PartFilt(N=800, reg=0.9, NER=0.2)
    xps += da.PartFilt(N=4000, reg=0.7, NER=0.05)
    xps += da.PFxN(xN=1000, N=30, Qs=2, NER=0.2)

    xps += da.OptPF(N=100, Qs=2, reg=0.7, NER=0.3)
    xps += da.EnKS("Serial", N=30, Lag=1)
    xps += da.EnRTS("Serial", N=30, DeCorr=0.99)

    # Run
    xps.launch(HMM, False, store_u=True)
    return xps


L63_old = """
    da_method     infl  upd_a       N  rot      xN  reg   NER  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  -----  -------  ----  -----  ----  ---  ----  -  -----------------  -----------------  -----------------
 0  Climatology                                                |     7.7676 ±1.2464     7.7676 ±1.2464     7.2044 ±2.4251
 1  OptInterp                                                  |     1.1648 ±0.1744     7.1198 ±1.1388     1.8578 ±0.4848
 2  Var3D                                                      |     1.0719 ±0.1192     1.7856 ±0.3686     1.2522 ±0.1616
 3  ExtKF        90                                            |     1.1932 ±0.4338     3.0113 ±1.1553     2.0016 ±0.8629
 4  EnKF          1.3   Sqrt        3  False                   |     0.5003 ±0.1105     1.1807 ±0.2613     0.8284 ±0.2526
 5  EnKF          1.02  Sqrt       10  True                    |     0.5773 ±0.0715     1.6134 ±0.4584     0.8839 ±0.1746
 6  EnKF          0.95  PertObs   500  False                   |     0.7422 ±0.3080     2.0616 ±1.0183     1.3171 ±0.4809
 7  EnKF_N        1                10  True      1             |     1.6050 ±0.5066     3.6838 ±0.7965     2.3756 ±0.4367
 8  iEnKS         1.02  Sqrt       10  True                    |     0.3927 ±0.2562     1.9267 ±0.7922     0.3172 ±0.1362
 9  PartFilt                      100               2.4  0.3   |     0.3574 ±0.1387     2.2799 ±1.5794     1.0327 ±0.7116
10  PartFilt                      800               0.9  0.2   |     0.5229 ±0.0832     1.3370 ±0.4291     0.8152 ±0.2085
11  PartFilt                     4000               0.7  0.05  |     0.2481 ±0.0474     0.6470 ±0.2298     0.3855 ±0.1051
12  PFxN                           30         1000       0.2   |     0.5848 ±0.0926     0.9573 ±0.2248     0.7203 ±0.1870
13  OptPF                         100               0.7  0.3   |     0.6577 ±0.1388     1.4330 ±0.4286     0.8705 ±0.2341
14  EnKS          1     Serial     30  False                   |     0.6586 ±0.1577     1.1681 ±0.3682     0.5304 ±0.1671
15  EnRTS         1     Serial     30  False                   |     0.9215 ±0.3187     2.3817 ±0.9076     0.7596 ±0.4891
"""[1:-1].splitlines(True)

# Example use of pytest-benchmark
# def test_duration(benchmark):
#     benchmark(L63_gen)


def test_len63(L63_table):
    assert len(L63_old) == len(L63_table)


@pytest.mark.parametrize(("lineno"), range(len(L63_old)))
def test_tables_L63(L63_table, lineno):
    expected = L63_old[lineno].rstrip()
    new      = L63_table[lineno].rstrip()
    assert new == expected


##############################
# L96
##############################
@pytest.fixture(scope="module")
def L96_table():
    import dapper.mods.Lorenz96 as model
    from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM

    model.Force = 8.0  # undo pinheiro2019
    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 10

    dpr.set_seed(3000)

    # xps
    xps = dpr.xpList()
    xps += da.Climatology()
    xps += da.OptInterp()
    xps += da.Var3D(xB=0.02)
    xps += da.ExtKF(infl=6)
    xps += da.EnKF("PertObs", N=40, infl=1.06)
    xps += da.EnKF("Sqrt", N=28, infl=1.02, rot=True)

    xps += da.EnKF_N(N=24, rot=True)
    xps += da.EnKF_N(N=24, rot=True, xN=2)
    xps += da.iEnKS("Sqrt", N=40, infl=1.01, rot=True)

    xps += da.LETKF(N=7, rot=True, infl=1.04, loc_rad=4)
    xps += da.SL_EAKF(N=7, rot=True, infl=1.07, loc_rad=6)

    xps.launch(HMM, store_u=True)

    table = xps.tabulate_avrgs(statkeys, decimals=4, colorize=False)
    return table.splitlines(True)


L96_old = """
    da_method    infl  upd_a     N  rot    xN  loc_rad  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  ----  -------  --  -----  --  -------  -  -----------------  -----------------  -----------------
 0  Climatology                                         |     0.8334 ±0.2326     0.8334 ±0.2326     0.8334 ±0.2326
 1  OptInterp                                           |     0.1328 ±0.0271     0.8345 ±0.2330     0.1328 ±0.0271
 2  Var3D                                               |     0.1009 ±0.0080     0.0874 ±0.0085     0.1009 ±0.0080
 3  ExtKF        6                                      |     0.0269 ±0.0010     0.0269 ±0.0012     0.0269 ±0.0010
 4  EnKF         1.06  PertObs  40  False               |     0.0318 ±0.0018     0.0317 ±0.0016     0.0318 ±0.0018
 5  EnKF         1.02  Sqrt     28  True                |     0.0375 ±0.0018     0.0375 ±0.0019     0.0375 ±0.0018
 6  EnKF_N       1              24  True    1           |     0.0311 ±0.0009     0.0310 ±0.0010     0.0311 ±0.0009
 7  EnKF_N       1              24  True    2           |     0.0304 ±0.0012     0.0304 ±0.0013     0.0304 ±0.0012
 8  iEnKS        1.01  Sqrt     40  True                |     0.0254 ±0.0009     0.0255 ±0.0009     0.0254 ±0.0008
 9  LETKF        1.04            7  True    1        4  |     0.0319 ±0.0013     0.0317 ±0.0013     0.0319 ±0.0013
10  SL_EAKF      1.07            7  True             6  |     0.0260 ±0.0017     0.0256 ±0.0014     0.0260 ±0.0017
"""[1:-1].splitlines(True)


def test_len96(L96_table):
    assert len(L96_old) == len(L96_table)


@pytest.mark.parametrize(("lineno"), range(len(L96_old)))
def test_tables_L96(L96_table, lineno):
    expected = L96_old[lineno].rstrip()
    new      = L96_table[lineno].rstrip()
    assert new == expected
