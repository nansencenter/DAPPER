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
    return L63_gen().splitlines(True)


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
    table = xps.tabulate_avrgs(statkeys, decimals=4, colorize=False)
    print(table)
    return table


L63_old = """
    da_method     infl  upd_a       N  rot      xN  reg   NER  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  -----  -------  ----  -----  ----  ---  ----  -  -----------------  -----------------  -----------------
 0  Climatology                                                |     7.7221 ±1.2096     7.7221 ±1.2096     7.1855 ±2.4362
 1  OptInterp                                                  |     1.0847 ±0.1002     7.7221 ±1.2096     1.4922 ±0.2957
 2  Persistence                                                |    11.0250 ±0.8284     0.7712 ±0.0049     0.5838 ±0.2090
 3  PreProg                                                    |     0.0000 ±0.0000     0.0000 ±0.0000     0.0000 ±0.0000
 4  Var3D                                                      |     0.9335 ±0.1031     1.8473 ±0.4604     1.1666 ±0.2480
 5  ExtKF        90                                            |     0.7243 ±0.1429     1.6843 ±0.4642     1.0992 ±0.3168
 6  EnKF          1.3   Sqrt        3  False                   |     0.5779 ±0.0736     1.3258 ±0.2536     0.8373 ±0.1530
 7  EnKF          1.02  Sqrt       10  True                    |     0.6423 ±0.0660     1.7242 ±0.3675     1.0183 ±0.1843
 8  EnKF          0.95  PertObs   500  False                   |     0.5719 ±0.0854     1.5111 ±0.3688     0.8907 ±0.1971
 9  EnKF_N        1                10  True      1             |     0.6230 ±0.0676     1.6784 ±0.3686     0.9912 ±0.1970
10  iEnKS         1.02  Sqrt       10  True                    |     0.3581 ±0.1094     0.9190 ±0.2488     0.2589 ±0.0792
11  PartFilt                      100               2.4  0.3   |     0.4665 ±0.0932     1.1912 ±0.3142     0.7288 ±0.1990
12  PartFilt                      800               0.9  0.2   |     0.3630 ±0.0979     0.8823 ±0.2273     0.5529 ±0.1766
13  PartFilt                     4000               0.7  0.05  |     0.3468 ±0.1096     0.8412 ±0.2295     0.5231 ±0.1808
14  PFxN                           30         1000       0.2   |     0.5391 ±0.1022     1.4199 ±0.3777     0.8734 ±0.2618
15  OptPF                         100               0.7  0.3   |     0.6512 ±0.1232     1.6482 ±0.4774     1.0154 ±0.3177
16  EnKS          1     Serial     30  False                   |     0.5997 ±0.0829     1.5564 ±0.3523     0.4004 ±0.0640
17  EnRTS         1     Serial     30  False                   |     0.5997 ±0.0829     1.5564 ±0.3523     0.2485 ±0.0637
"""[1:-1].splitlines(True)

# Example use of pytest-benchmark
# def test_duration(benchmark):
#     benchmark(L63_gen)


# Ignore stats.py:warn_zero_variance() due to PreProg having var 0
@pytest.mark.filterwarnings("ignore::UserWarning")
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
    return L96_gen().splitlines(True)


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
    table = xps.tabulate_avrgs(statkeys, decimals=4, colorize=False)
    print(table)
    return table


L96_old = """
    da_method    infl  upd_a     N  rot    xN  loc_rad  |  err.rms.a  1σ      err.rms.f  1σ      err.rms.u  1σ
--  -----------  ----  -------  --  -----  --  -------  -  -----------------  -----------------  -----------------
 0  Climatology                                         |     0.8343 ±0.2329     0.8343 ±0.2329     0.8343 ±0.2329
 1  OptInterp                                           |     0.1563 ±0.0355     0.8343 ±0.2329     0.1563 ±0.0355
 2  Persistence                                         |     0.3075 ±0.0285     0.3075 ±0.0285     0.3075 ±0.0285
 3  PreProg                                             |     0.0000 ±0.0000     0.0000 ±0.0000     0.0000 ±0.0000
 4  Var3D                                               |     0.0790 ±0.0254     0.0753 ±0.0238     0.0790 ±0.0254
 5  ExtKF        6                                      |     0.0225 ±0.0012     0.0224 ±0.0011     0.0225 ±0.0012
 6  EnKF         1.06  PertObs  40  False               |     0.0233 ±0.0014     0.0232 ±0.0013     0.0233 ±0.0014
 7  EnKF         1.02  Sqrt     28  True                |     0.0228 ±0.0012     0.0228 ±0.0011     0.0228 ±0.0012
 8  EnKF_N       1              24  True    1           |     0.0225 ±0.0010     0.0225 ±0.0009     0.0225 ±0.0010
 9  EnKF_N       1              24  True    2           |     0.0225 ±0.0010     0.0224 ±0.0009     0.0225 ±0.0010
10  iEnKS        1.01  Sqrt     40  True                |     0.0231 ±0.0012     0.0230 ±0.0012     0.0231 ±0.0012
11  LETKF        1.04            7  True             4  |     0.0245 ±0.0009     0.0245 ±0.0009     0.0245 ±0.0009
12  SL_EAKF      1.07            7  True             6  |     0.0246 ±0.0010     0.0246 ±0.0010     0.0246 ±0.0010
"""[1:-1].splitlines(True)


def test_len96(L96_table):
    assert len(L96_old) == len(L96_table)


@pytest.mark.parametrize(("lineno"), range(len(L96_old)))
def test_tables_L96(L96_table, lineno):
    expected = L96_old[lineno].rstrip()
    new = L96_table[lineno].rstrip()
    assert new == expected


if __name__ == "__main__":
    L63_gen()
    L96_gen()
