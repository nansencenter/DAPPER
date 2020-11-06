"""Tests with the LA model, which are very useful for testing
boundary cases of the iEnKS (e.g. nIter=1, Lag=0)."""

##############################
# Preamble
##############################
import numpy as np

import dapper as dpr
from dapper.dict_tools import deep_getattr
from dapper.mods.LA.small import HMM

HMM.t.BurnIn = 0
HMM.t.KObs = 10

xps = dpr.xpList(unique=True)

# yapf: disable
xps += dpr.EnKF('Sqrt'   , N=20)
xps += dpr.EnKF('PertObs', N=20)
xps += dpr.EnKF('DEnKF'  , N=20)
for Lag in [0,1,3]:
    xps += dpr.EnKS('Sqrt'   , N=20, Lag=Lag)
    xps += dpr.EnKS('PertObs', N=20, Lag=Lag)
    xps += dpr.EnKS('DEnKF'  , N=20, Lag=Lag)
    for nIter in [1,4]:
        for MDA in [False,True]:
            xps += dpr.iEnKS('Sqrt'   , N=20, Lag=Lag, nIter=nIter, MDA=MDA)
            xps += dpr.iEnKS('PertObs', N=20, Lag=Lag, nIter=nIter, MDA=MDA)
            xps += dpr.iEnKS('Order1' , N=20, Lag=Lag, nIter=nIter, MDA=MDA)
# yapf: enable

for xp in xps:
    xp.seed = 3000

xps.launch(HMM, store_u=True, save_as=False)
print(xps.tabulate_avrgs(["rmse.a", "rmse.f", "rmse.u", "rmse.s"]))

##############################
# Test aux functions
##############################
find = xps.inds  # lookup inds


def _allsame(xx):
    # rtol=0 => only atol matters.
    return np.allclose(xx, xx[0], atol=1e-10, rtol=0)


def _rmse(sub, indices):
    def get_val(i):
        try:
            return deep_getattr(xps[i].avrgs, 'err.rms.' + sub).val
        except AttributeError:
            return np.nan

    return [get_val(i) for i in indices]


##############################
# Tests
##############################
# These don't evaluate the absolute numbers,
# just that they match with the results from other
# methods that are supposed to be equivalent.
# * Understanding why the matches should (and shouldn't) arise is very instructive.
# * Tests can be run with any initial seed.


# For Sqrt flavour:
# All filter/analysis (f/a) stats should be equal
# for all EnKF/EnKS/iEnKS(nIter/Lag/MDA).
def test_Sqrt():
    ii = find(strict=1, upd_a='Sqrt')
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# However, the u stats depend on the Lag, of course.
# Test together with non-iter filters:
def test_Sqrt_u():
    ii = find(strict=0, upd_a='Sqrt', Lag=0)
    assert _allsame(_rmse('u', ii))


# Idem for s stats (not def'd for filters)
def test_Sqrt_Lag1_u():
    ii = find(strict=1, upd_a='Sqrt', Lag=1)
    assert _allsame(_rmse('u', ii))
    assert _allsame(_rmse('s', ii))


def test_Sqrt_Lag3_u():
    ii = find(strict=1, upd_a='Sqrt', Lag=3)
    assert _allsame(_rmse('u', ii))
    assert _allsame(_rmse('s', ii))


# For PertObs flavour::
# - f/a stats all equal except for MDA with nIter>1.
def test_PertObs():
    ii = (find(strict=0, upd_a='PertObs', MDA=0) +
          find(strict=1, upd_a='PertObs', MDA=1, nIter=1))
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# - Still with nIter=4, f/a stats of MDA does not depend on Lag.
def test_PertObs_nIter4():
    ii = find(strict=1, upd_a='PertObs', nIter=4, MDA=1)
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# - u stats equal for filter and (iter/non-iter) smoothers with Lag=0, except MDA with nIter>1:
def test_PertObs_u():
    ii = (find(strict=0, upd_a='PertObs', MDA=0, Lag=0) +
          find(strict=1, upd_a='PertObs', MDA=1, Lag=0, nIter=1))
    assert _allsame(_rmse('u', ii))


# - u stats equal for            (iter/non-iter) smoothers with Lag=1, except MDA with nIter>1:
def test_PertObs_Lag1_u():
    ii = find(upd_a='PertObs', Lag=1)
    ii.remove(find(upd_a='PertObs', Lag=1, MDA=1, nIter=4)[0])
    assert _allsame(_rmse('u', ii))
    assert _allsame(_rmse('s', ii))


# - u stats equal for            (iter/non-iter) smoothers with Lag=3, except MDA with nIter>1:
def test_PertObs_Lag3_u():
    ii = find(upd_a='PertObs', Lag=3)
    ii.remove(find(upd_a='PertObs', Lag=3, MDA=1, nIter=4)[0])
    assert _allsame(_rmse('u', ii))
    assert _allsame(_rmse('s', ii))


# For Order1 (DEnKF) flavour:
# f/a stats all equal except for nIter>1:
def test_Order1():
    ii = find(upd_a='DEnKF') +\
         find(upd_a='Order1',nIter=1)
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# f/a stats independent of Lag for non-MDA and a given nIter:
def test_Order1_nIter4_MDA0():
    ii = find(upd_a='Order1', nIter=4, MDA=0)
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# f/a stats independent of Lag for     MDA and a given nIter:
def test_Order1_nIter4_MDA1():
    ii = find(upd_a='Order1', nIter=4, MDA=1)
    assert _allsame(_rmse('a', ii))
    assert _allsame(_rmse('f', ii))


# u   stats equal for EnKS/iEnKS(nIter=1) for a given Lag:
def test_Order1_nIter1_Lag0_u():
    ii = find(strict=0,upd_a='DEnKF' ,Lag=0) +\
         find(upd_a='Order1',Lag=0,nIter=1)
    assert _allsame(_rmse('u', ii))


def test_Order1_nIter1_Lag1_u():
    ii = find(da=dpr.EnKS, upd_a='DEnKF' ,Lag=1) +\
         find(upd_a='Order1',Lag=1,nIter=1)
    assert _allsame(_rmse('u', ii))


def test_Order1_nIter1_Lag3_u():
    ii = find(da=dpr.EnKS, upd_a='DEnKF' ,Lag=3) +\
         find(upd_a='Order1',Lag=3,nIter=1)
    assert _allsame(_rmse('u', ii))


# For nonlinear dynamics, the (non-iterative) EnKF (f/a/u stats)
# are reproduced by the iEnKS with Lag=0 (requires nIter==1 if Obs.mod is also nonlin).
# However, the 'u' stats of the non-iterative EnKS(Lag>0) are not reproduced.
# Re-use xps and test with:
# from dapper.mods.Lorenz96.sakov2008 import HMM
# HMM.t.KObs=100 # Here, must use >100 to avoid indistinguishable rmse stats.
