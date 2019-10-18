# Tests with the LA model, which are very useful for testing
# boundary cases of the iEnKS (e.g. nIter=1, Lag=0).

from dapper import *

sd0 = seed_init(3)

cfgs  = List_of_Configs(unique=True)

from dapper.mods.LA.small import HMM
HMM.t.BurnIn=0
HMM.t.KObs=10

cfgs +=  EnKF('Sqrt'        , N=20,                      )
cfgs +=  EnKF('PertObs'     , N=20,                      )
cfgs +=  EnKF('DEnKF'       , N=20,                      )
for Lag in [0,1,3]:
  cfgs +=  EnKS('Sqrt'      , N=20, Lag=Lag,             )
  cfgs +=  EnKS('PertObs'   , N=20, Lag=Lag,             )
  cfgs +=  EnKS('DEnKF'     , N=20, Lag=Lag,             )
  for nIter in [1,4]:
    for MDA in [False,True]:
      cfgs += iEnKS('Sqrt'    , N=20, Lag=Lag, nIter=nIter, MDA=MDA)
      cfgs += iEnKS('PertObs' , N=20, Lag=Lag, nIter=nIter, MDA=MDA)
      cfgs += iEnKS('Order1'  , N=20, Lag=Lag, nIter=nIter, MDA=MDA)


##############################
# Assimilate
##############################
xx,yy = HMM.simulate()

cfgs.assimilate(HMM,xx,yy,sd0+2,store_u=True)
cfgs.print_avrgs(['err.rms.u','err.rms.s'])


##############################
# Test aux functions
##############################
inds = cfgs.inds # lookup inds

def allsame(xx): 
  return np.allclose(xx, xx[0], atol=1e-10, rtol=0) # rtol=0 => only atol matters.

def gr(ii,sub): # grab_rmses
    def get_val(i):
        try: return deep_getattr(cfgs[i].avrgs,'err.rms.'+sub).val
        except AttributeError: return nan
    return [get_val(i) for i in ii]


##############################
# Tests
##############################
# These don't evaluate the actual numbers,
# just that they match with the results from other
# methods that are supposed to be equivalent.
# * Understanding why the matches should (and shouldn't) arise is very instructive.
# * Tests can be run with any initial seed.

# For Sqrt flavour:
# - All filter/analysis (f/a) stats should be equal for all EnKF/EnKS/iEnKS(nIter/Lag/MDA).
ii = inds(strict=1,upd_a='Sqrt')
def test_Sqrt(ii=ii):                                      assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - However, the u stats depend on the Lag, of course. Test together with non-iter filters:
ii = inds(strict=0,upd_a='Sqrt',Lag=0)
def test_Sqrt_u(ii=ii):                                    assert allsame(gr(ii,'u'))
# Idem for s stats (not def'd for filters):
ii = inds(strict=1,upd_a='Sqrt',Lag=1)
def test_Sqrt_Lag1_u(ii=ii):                               assert allsame(gr(ii,'u')) and allsame(gr(ii,'s'))
ii = inds(strict=1,upd_a='Sqrt',Lag=3)
def test_Sqrt_Lag3_u(ii=ii):                               assert allsame(gr(ii,'u')) and allsame(gr(ii,'s'))

# For PertObs flavour::
# - f/a stats all equal except for MDA with nIter>1.
ii = inds(strict=0,upd_a='PertObs',MDA=0) +\
     inds(strict=1,upd_a='PertObs',MDA=1,nIter=1)
def test_PertObs(ii=ii):                                   assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - Still with nIter=4, f/a stats of MDA does not depend on Lag.
ii = inds(strict=1,upd_a='PertObs',nIter=4,MDA=1)
def test_PertObs_nIter4(ii=ii):                            assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - u stats equal for filter and (iter/non-iter) smoothers with Lag=0, except MDA with nIter>1:
ii = inds(strict=0,upd_a='PertObs',MDA=0,Lag=0) +\
     inds(strict=1,upd_a='PertObs',MDA=1,Lag=0,nIter=1)
def test_PertObs_u(ii=ii):                                 assert allsame(gr(ii,'u'))
# - u stats equal for            (iter/non-iter) smoothers with Lag=1, except MDA with nIter>1:
ii =      inds(upd_a='PertObs',Lag=1)
ii.remove(inds(upd_a='PertObs',Lag=1,MDA=1,nIter=4)[0])
def test_PertObs_Lag1_u(ii=ii):                            assert allsame(gr(ii,'u')) and allsame(gr(ii,'s'))
# - u stats equal for            (iter/non-iter) smoothers with Lag=3, except MDA with nIter>1:
ii =      inds(upd_a='PertObs',Lag=3)
ii.remove(inds(upd_a='PertObs',Lag=3,MDA=1,nIter=4)[0])
def test_PertObs_Lag3_u(ii=ii):                            assert allsame(gr(ii,'u')) and allsame(gr(ii,'s'))

# For Order1 (DEnKF) flavour:
# - f/a stats all equal except for nIter>1:
ii = inds(upd_a='DEnKF') +\
     inds(upd_a='Order1',nIter=1)
def test_Order1(ii=ii):                                    assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - f/a stats independent of Lag for non-MDA and a given nIter:
ii = inds(upd_a='Order1',nIter=4,MDA=0)
def test_Order1_nIter4_MDA0(ii=ii):                        assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - f/a stats independent of Lag for     MDA and a given nIter:
ii = inds(upd_a='Order1',nIter=4,MDA=1)
def test_Order1_nIter4_MDA1(ii=ii):                        assert allsame(gr(ii,'a')) and allsame(gr(ii,'f'))
# - u   stats equal for EnKS/iEnKS(nIter=1) for a given Lag:
ii = inds(strict=0,upd_a='DEnKF' ,Lag=0) +\
     inds(         upd_a='Order1',Lag=0,nIter=1)
def test_Order1_nIter1_Lag0_u(ii=ii):                      assert allsame(gr(ii,'u'))
ii = inds(da=EnKS, upd_a='DEnKF' ,Lag=1) +\
     inds(         upd_a='Order1',Lag=1,nIter=1)
def test_Order1_nIter1_Lag1_u(ii=ii):                      assert allsame(gr(ii,'u'))
ii = inds(da=EnKS, upd_a='DEnKF' ,Lag=3) +\
     inds(         upd_a='Order1',Lag=3,nIter=1)
def test_Order1_nIter1_Lag3_u(ii=ii):                      assert allsame(gr(ii,'u'))


##############################
# Seed-dependent test
##############################
# Just stupidly compare the full table.
# ==> Test cannot be run with different seeds or computers.
# Seed used: sd0 = seed_init(3). Test run on my Mac.

import pytest
old = {}

parameters = [(faus, i) for faus in 'fasu' for i in arange(len(cfgs))]

@pytest.mark.parametrize(('fau_and_i'),parameters)
def test_value(fau_and_i):
  faus, i = fau_and_i
  oldval  = old[faus][i]
  newval  = gr([i],faus).pop()
  if np.isnan(oldval): assert np.isnan(newval)
  # else:              assert oldval == newval
  else:                assert np.isclose(oldval,newval,rtol=0, atol=1e-14) # rtol=0 => only atol matters.

# The following tables were printed using:
# for i,val in enumerate(gr(arange(len(cfgs)),faus)):
#     if i%3==0: print("")
#     print("%.14f , "%val, end="")

old['a'] = [
0.12766884966271 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966271 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.20352220860399 ,
0.12766884966271 , 0.14957193697425 , 0.13138618085556 ,
0.12766884966271 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.20352220860399 ,
0.12766884966271 , 0.14957193697425 , 0.13138618085556 ,
0.12766884966271 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966271 , 0.13892314812010 , 0.20352220860398 ,
0.12766884966271 , 0.14957193697425 , 0.13138618085556 ,
]


old['f'] = [
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.28249567656023 ,
0.20793122131992 , 0.22808905279646 , 0.21146242002686 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.28249567656023 ,
0.20793122131992 , 0.22808905279646 , 0.21146242002686 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.24164099430916 ,
0.20793122131992 , 0.21830084306957 , 0.28249567656021 ,
0.20793122131992 , 0.22808905279646 , 0.21146242002686 ,
]

old['s'] = [
nan , nan , nan ,
0.12766884966271 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.16345906375094 ,
0.12766884966270 , 0.13892314812010 , 0.20352220860399 ,
0.12766884966271 , 0.14957193697425 , 0.13138618085556 ,
0.07145667074868 , 0.08359564591382 , 0.10932732593591 ,
0.07145667074868 , 0.08359564591382 , 0.10932732593591 ,
0.07145667074868 , 0.08359564591382 , 0.10932732593591 ,
0.07145667074868 , 0.08359564591382 , 0.14990061446611 ,
0.07145667074868 , 0.09233546994813 , 0.07478635415345 ,
0.04985108855896 , 0.05965799854860 , 0.07868056744027 ,
0.04985108855896 , 0.05965799854860 , 0.07868056744027 ,
0.04985108855896 , 0.05965799854860 , 0.07868056744027 ,
0.04985108855896 , 0.05965799854860 , 0.07994971837720 ,
0.04985108855896 , 0.07124637471779 , 0.05282093277765 ,
]

old['u'] = [
0.19187874698847 , 0.20242530407968 , 0.22600460819751 ,
0.19187874698847 , 0.20242530407968 , 0.22600460819751 ,
0.19187874698847 , 0.20242530407968 , 0.22600460819751 ,
0.19187874698847 , 0.20242530407968 , 0.22600460819751 ,
0.19187874698847 , 0.20242530407968 , 0.26670098296898 ,
0.19187874698847 , 0.21238562963202 , 0.19544717219260 ,
0.11642641387990 , 0.12785764767885 , 0.15263271618794 ,
0.11642641387990 , 0.12785764767885 , 0.15263271618794 ,
0.11642641387990 , 0.12785764767885 , 0.15263271618794 ,
0.11642641387990 , 0.12785764767885 , 0.19279788977641 ,
0.11642641387990 , 0.13812464356902 , 0.12006621551514 ,
0.05905392234704 , 0.06969574764904 , 0.09163818415731 ,
0.05905392234704 , 0.06969574764904 , 0.09163818415731 ,
0.05905392234704 , 0.06969574764904 , 0.09163818415731 ,
0.05905392234704 , 0.06969574764904 , 0.10702955087963 ,
0.05905392234704 , 0.08024823550524 , 0.06232536252659 ,
]


# For nonlinear dynamics, the (non-iterative) EnKF (f/a/u stats)
# are reproduced by the iEnKS with Lag=0 (requires nIter==1 if Obs.mod is also nonlin).
# However, the 'u' stats of the non-iterative EnKS(Lag>0) are not reproduced.
# Re-use cfgs and test with:
# from dapper.mods.Lorenz96.sakov2008 import HMM
# HMM.t.KObs=100 # Here, must use >100 to avoid indistinguishable rmse stats.




