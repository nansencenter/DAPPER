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

    # xps
    xps = dpr.xpList()
    xps += da.Climatology()
    xps += da.OptInterp()
    xps += da.Persistence()
    xps += da.PreProg(lambda k, xx, yy: xx[k])
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

    for xp in xps:
        xp.seed = 3000

    # Run
    xps.launch(HMM, False, store_u=True)
    return xps


L63_old = """
    da_method     infl  upd_a       N  rot      xN  reg   NER  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  -----  -------  ----  -----  ----  ---  ----  -  -----------------  -----------------  -----------------
 0  Climatology                                                |     7.7676 ±1.2464     7.7676 ±1.2464     7.2044 ±2.4251
 1  OptInterp                                                  |     1.5874 ±0.2144     7.7676 ±1.2464     1.8993 ±0.3675
 2  Persistence                                                |    11.1180 ±0.8537     0.7996 ±0.0050     0.5859 ±0.2071
 3  PreProg                                                    |     0.0000 ±0.0000     0.0000 ±0.0000     0.0000 ±0.0000
 4  Var3D                                                      |     1.0662 ±0.2455     2.3099 ±0.7535     1.4907 ±0.3301
 5  ExtKF        90                                            |     1.3417 ±0.2388     2.8749 ±0.6220     1.9105 ±0.3725
 6  EnKF          1.3   Sqrt        3  False                   |     0.8437 ±0.2573     1.8524 ±0.7625     1.2694 ±0.3529
 7  EnKF          1.02  Sqrt       10  True                    |     0.7360 ±0.3135     1.5493 ±0.7115     1.0552 ±0.3468
 8  EnKF          0.95  PertObs   500  False                   |     0.7228 ±0.3170     1.6368 ±0.8696     1.1128 ±0.3766
 9  EnKF_N        1                10  True      1             |     0.8022 ±0.3193     1.6628 ±0.7943     1.1370 ±0.3693
10  iEnKS         1.02  Sqrt       10  True                    |     0.6630 ±0.2186     1.6572 ±0.4497     0.5732 ±0.2182
11  PartFilt                      100               2.4  0.3   |     0.8288 ±0.2360     2.2418 ±0.7123     1.4374 ±0.4028
12  PartFilt                      800               0.9  0.2   |     0.6818 ±0.2207     1.8165 ±0.5496     1.1754 ±0.3406
13  PartFilt                     4000               0.7  0.05  |     0.6613 ±0.2559     1.7120 ±0.5482     1.1145 ±0.3160
14  PFxN                           30         1000       0.2   |     1.1707 ±0.1514     2.6352 ±0.5854     1.7737 ±0.3561
15  OptPF                         100               0.7  0.3   |     0.9487 ±0.2281     2.1878 ±1.0231     1.4356 ±0.3715
16  EnKS          1     Serial     30  False                   |     0.8004 ±0.2781     1.7502 ±0.7593     0.6460 ±0.1874
17  EnRTS         1     Serial     30  False                   |     0.8004 ±0.2781     1.7502 ±0.7593     0.5406 ±0.1810
"""[
    1:-1
].splitlines(
    True
)

# Example use of pytest-benchmark
# def test_duration(benchmark):
#     benchmark(L63_gen)


def test_len63(L63_table):
    assert len(L63_old) == len(L63_table)


@pytest.mark.parametrize(("lineno"), range(len(L63_old)))
def test_tables_L63(L63_table, lineno):
    expected = L63_old[lineno].rstrip()
    new = L63_table[lineno].rstrip()
    assert new == expected


##############################
# L96
##############################
@pytest.fixture(scope="module")
def L96_table():
    xps = L96_gen()
    table = xps.tabulate_avrgs(statkeys, decimals=4, colorize=False)
    return table.splitlines(True)


def L96_gen():
    import dapper.mods.Lorenz96 as model
    from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM

    model.Force = 8.0  # undo pinheiro2019
    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 10

    # xps
    xps = dpr.xpList()
    xps += da.Climatology()
    xps += da.OptInterp()
    xps += da.Persistence()
    xps += da.PreProg(lambda k, xx, yy: xx[k])
    xps += da.Var3D(xB=0.02)
    xps += da.ExtKF(infl=6)
    xps += da.EnKF("PertObs", N=40, infl=1.06)
    xps += da.EnKF("Sqrt", N=28, infl=1.02, rot=True)

    xps += da.EnKF_N(N=24, rot=True)
    xps += da.EnKF_N(N=24, rot=True, xN=2)
    xps += da.iEnKS("Sqrt", N=40, infl=1.01, rot=True)

    xps += da.LETKF(N=7, rot=True, infl=1.04, loc_rad=4)
    xps += da.SL_EAKF(N=7, rot=True, infl=1.07, loc_rad=6)

    for xp in xps:
        xp.seed = 3000

    xps.launch(HMM, store_u=True)
    return xps


L96_old = """
    da_method    infl  upd_a     N  rot    xN  loc_rad  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  ----  -------  --  -----  --  -------  -  -----------------  -----------------  -----------------
 0  Climatology                                         |     0.8334 ±0.2326     0.8334 ±0.2326     0.8334 ±0.2326
 1  OptInterp                                           |     0.0949 ±0.0292     0.8334 ±0.2326     0.0949 ±0.0292
 2  Persistence                                         |     0.3071 ±0.0284     0.3071 ±0.0284     0.3071 ±0.0284
 3  PreProg                                             |     0.0000 ±0.0000     0.0000 ±0.0000     0.0000 ±0.0000
 4  Var3D                                               |     0.0593 ±0.0057     0.0575 ±0.0055     0.0593 ±0.0057
 5  ExtKF        6                                      |     0.0350 ±0.0010     0.0352 ±0.0011     0.0350 ±0.0010
 6  EnKF         1.06  PertObs  40  False               |     0.0355 ±0.0011     0.0356 ±0.0011     0.0355 ±0.0011
 7  EnKF         1.02  Sqrt     28  True                |     0.0352 ±0.0010     0.0352 ±0.0011     0.0352 ±0.0010
 8  EnKF_N       1              24  True    1           |     0.0360 ±0.0011     0.0360 ±0.0012     0.0360 ±0.0011
 9  EnKF_N       1              24  True    2           |     0.0360 ±0.0011     0.0360 ±0.0012     0.0360 ±0.0011
10  iEnKS        1.01  Sqrt     40  True                |     0.0356 ±0.0011     0.0357 ±0.0012     0.0356 ±0.0011
11  LETKF        1.04            7  True             4  |     0.0356 ±0.0010     0.0358 ±0.0012     0.0356 ±0.0010
12  SL_EAKF      1.07            7  True             6  |     0.0354 ±0.0010     0.0357 ±0.0012     0.0354 ±0.0010
"""[
    1:-1
].splitlines(
    True
)


def test_len96(L96_table):
    assert len(L96_old) == len(L96_table)


@pytest.mark.parametrize(("lineno"), range(len(L96_old)))
def test_tables_L96(L96_table, lineno):
    expected = L96_old[lineno].rstrip()
    new = L96_table[lineno].rstrip()
    assert new == expected
