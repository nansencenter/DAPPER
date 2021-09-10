"""Test graphics/plotting.

This won't automatically verify if the plots are correct,
only whether they cause errors or not.
"""

import numpy as np

import dapper as dpr
import dapper.da_methods as da


def test_L63():
    from dapper.mods.Lorenz63.sakov2012 import HMM as _HMM

    xps = dpr.xpList()
    xps += da.EnKF('Sqrt', N=10, infl=1.02, rot=True)
    xps += da.PartFilt(N=20, reg=2.4, NER=0.3)
    xps += da.OptInterp()
    # xps += da.iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)

    HMM = _HMM.copy()
    HMM.tseq.BurnIn = HMM.tseq.dto
    HMM.tseq.Ko = 1

    xps.launch(HMM,
               free=False,
               liveplots="all",
               store_u=False,
               fail_gently=False)

    for xp in xps:
        xp.stats.replay("all")

    xp.stats.replay(t2=1)
    xp.stats.replay(t2=0.0)
    xp.stats.replay(t2=0.3)
    xp.stats.replay(t2=0.8)
    xp.stats.replay(t2=0.8, t1=0.2)
    xp.stats.replay(t2=np.inf)
    xp.stats.replay(t2=np.inf, speed=1)
    xp.stats.replay(t2=np.inf, pause_a=0, pause_f=0)

    assert True  # An assertion for pytest to count
    return HMM, xps  # Return useful stuff


def test_L96():
    xps = dpr.xpList()

    from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM
    xps += da.EnKF('PertObs', N=40, infl=1.06)
    xps += da.EnKF('Serial', N=28, infl=1.02, rot=True)
    xps += da.OptInterp()
    xps += da.Var3D(xB=0.02)
    xps += da.ExtKF(infl=10)
    xps += da.LETKF(N=6, rot=True, infl=1.05, loc_rad=4, taper='Step')

    # from dapper.mods.Lorenz96.bocquet2015loc import HMM
    # xps += da.EnKF_N(  N=24, rot=True ,infl=1.01)
    # xps += da.PartFilt(N=3000,NER=0.20,reg=1.2)
    # xps += da.PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)

    HMM = _HMM.copy()
    # HMM.tseq.BurnIn = 10*HMM.tseq.dto
    # HMM.tseq.Ko = 30
    HMM.tseq.BurnIn = HMM.tseq.dto
    HMM.tseq.Ko = 2

    xps.launch(HMM,
               free=False,
               liveplots="all",
               store_u=False,
               fail_gently=False,
               save_as=False)

    for xp in xps:
        xp.stats.replay("all")

    xp.stats.replay(t2=1)
    xp.stats.replay(t2=0.0)
    xp.stats.replay(t2=0.3)
    xp.stats.replay(t2=0.8)
    xp.stats.replay(t2=0.8, t1=0.2)
    xp.stats.replay(t2=np.inf)
    xp.stats.replay(t2=np.inf, speed=1)
    xp.stats.replay(t2=np.inf, pause_a=0, pause_f=0)

    assert True  # An assertion for pytest to count
    return HMM, xps  # Return useful stuff


# Non py.test runs:
# HMM, xps = test_L63()
# HMM, xps = test_L96()
