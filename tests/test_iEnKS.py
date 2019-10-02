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


for c in cfgs:
    c.liveplots=False
    c.store_u=True


##############################
# Assimilate
##############################
xx,yy = simulate(HMM)

cfgs.assimilate(HMM,xx,yy,sd0+2)
cfgs.print_avrgs(['rmse_u','rmse_s'])


##############################
# Test aux functions
##############################
inds = cfgs.inds # lookup inds

def allsame(xx): 
  return np.allclose(xx, xx[0], atol=1e-10, rtol=0) # rtol=0 => only atol matters.

def gr(ii,f_a_u): # grab_rmses
  return [cfgs[i].avrgs['rmse_'+f_a_u].val for i in ii]


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

parameters = [(fau, i) for fau in 'fasu' for i in arange(len(cfgs))]

@pytest.mark.parametrize(('fau_and_i'),parameters)
def test_value(fau_and_i):
  fau, i = fau_and_i
  oldval = old[fau][i]
  newval = gr([i],fau).pop()
  if np.isnan(oldval): assert np.isnan(newval)
  # else:              assert oldval == newval
  else:                assert np.isclose(oldval,newval,rtol=0, atol=1e-14) # rtol=0 => only atol matters.

# The following tables were printed using:
# for i,val in enumerate(gr(arange(len(cfgs)),fau)):
#     if i%3==0: print("")
#     print("%.14f , "%val, end="")

old['a'] = \
       [0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.19698937833747 ,
        0.11751799654458 , 0.15220018529924 , 0.11922523809866 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.19698937833746 ,
        0.11751799654458 , 0.15220018529923 , 0.11922523809866 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
        0.11751799654458 , 0.14487650979237 , 0.19698937833746 ,
        0.11751799654458 , 0.15220018529923 , 0.11922523809866 ]

old['f'] = \
       [0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.27653576499440 ,
        0.19720575243817 , 0.22909742847010 , 0.19906226974875 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.27653576499440 ,
        0.19720575243817 , 0.22909742847010 , 0.19906226974875 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.23869707498668 ,
        0.19720575243817 , 0.22161407590847 , 0.27653576499440 ,
        0.19720575243817 , 0.22909742847010 , 0.19906226974875 ]

old['s'] = \
         [             nan ,              nan ,              nan ,
          0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
          0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
          0.11751799654458 , 0.14487650979237 , 0.15956674158954 ,
          0.11751799654458 , 0.14487650979237 , 0.19698937833747 ,
          0.11751799654458 , 0.15220018529924 , 0.11922523809866 ,
          0.07315409742556 , 0.10346280045084 , 0.11576026496697 ,
          0.07315409742556 , 0.10346280045084 , 0.11576026496697 ,
          0.07315409742556 , 0.10346280045084 , 0.11576026496697 ,
          0.07315409742556 , 0.10346280045084 , 0.13592128415198 ,
          0.07315409742556 , 0.10859044786211 , 0.07484646780488 ,
          0.05909040587116 , 0.09166595931495 , 0.07053346415784 ,
          0.05909040587116 , 0.09166595931495 , 0.07053346415784 ,
          0.05909040587116 , 0.09166595931495 , 0.07053346415784 ,
          0.05909040587116 , 0.09166595931495 , 0.07537820504307 ,
          0.05909040587116 , 0.09328813968917 , 0.05814318598562 ]

old['u'] = \
         [0.18126820125945 , 0.20626656268525 , 0.22287100830725 ,
          0.18126820125945 , 0.20626656268525 , 0.22287100830725 ,
          0.18126820125945 , 0.20626656268525 , 0.22287100830725 ,
          0.18126820125945 , 0.20626656268525 , 0.22287100830725 ,
          0.18126820125945 , 0.20626656268525 , 0.26062648766301 ,
          0.18126820125945 , 0.21371797983593 , 0.18309486341873 ,
          0.10864521672078 , 0.13659376792406 , 0.15080544626503 ,
          0.10864521672078 , 0.13659376792406 , 0.15080544626503 ,
          0.10864521672078 , 0.13659376792406 , 0.15080544626503 ,
          0.10864521672078 , 0.13659376792406 , 0.18477575950037 ,
          0.10864521672078 , 0.14347823781181 , 0.11034948403990 ,
          0.06073856300593 , 0.09559556205105 , 0.08240207473401 ,
          0.06073856300593 , 0.09559556205106 , 0.08240207473401 ,
          0.06073856300593 , 0.09559556205106 , 0.08240207473401 ,
          0.06073856300593 , 0.09559556205105 , 0.09878194438072 ,
          0.06073856300593 , 0.09619878662669 , 0.06074381005457 ]


# For nonlinear dynamics, the (non-iterative) EnKF (f/a/u stats)
# are reproduced by the iEnKS with Lag=0 (requires nIter==1 if Obs.mod is also nonlin).
# However, the 'u' stats of the non-iterative EnKS(Lag>0) are not reproduced.
# Re-use cfgs and test with:
# from dapper.mods.Lorenz95.sak08 import HMM
# HMM.t.KObs=100 # Here, must use >100 to avoid indistinguishable rmse stats.




