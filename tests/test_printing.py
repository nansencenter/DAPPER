"""Test printing"""

import dapper as dpr
import dapper.tools.utils as utils
from dapper.tools.magic import spell_out

utils.disable_user_interaction = True


def test_L63():
    from dapper.mods.Lorenz63.sakov2012 import HMM

    xps = dpr.xpList()
    xps += dpr.EnKF("Sqrt", N=10, infl=1.02, rot=True)
    xps += dpr.PartFilt(N=20, reg=2.4, NER=0.3)
    xps += dpr.OptInterp()
    # xps += dpr.iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)

    HMM.t.BurnIn = HMM.t.dtObs
    HMM.t.KObs = 1

    xps.launch(
        HMM,
        free=False,
        statkeys=True,
        liveplots=None,
        store_u=False,
        fail_gently=False,
        save_as=False,
    )
    print(xps.tabulate_avrgs(["rmse.a"]))

    spell_out(HMM)
    spell_out(xps[-1])
    spell_out(xps[-1].stats)
    spell_out(xps[-1].avrgs)

    assert True  # An assertion for pytest to count
    return HMM, xps  # Return useful stuff


def test_L96():
    xps = dpr.xpList()

    from dapper.mods.Lorenz96.sakov2008 import HMM

    xps += dpr.EnKF("PertObs", N=40, infl=1.06)
    xps += dpr.EnKF("Serial", N=28, infl=1.02, rot=True)
    xps += dpr.OptInterp()
    xps += dpr.Var3D(xB=0.02)
    xps += dpr.ExtKF(infl=10)
    xps += dpr.LETKF(N=6, rot=True, infl=1.05, loc_rad=4, taper="Step")

    # from dapper.mods.Lorenz96.bocquet2015loc import HMM
    # xps += dpr.EnKF_N(  N=24, rot=True ,infl=1.01)
    # xps += dpr.PartFilt(N=3000,NER=0.20,reg=1.2)
    # xps += dpr.PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)

    # HMM.t.BurnIn = 10*HMM.t.dtObs
    # HMM.t.KObs = 30
    HMM.t.BurnIn = HMM.t.dtObs
    HMM.t.KObs = 2

    xps.launch(
        HMM,
        free=False,
        statkeys=True,
        liveplots=None,
        store_u=False,
        fail_gently=False,
        save_as=False,
    )
    print(xps.tabulate_avrgs(["rmse.a"]))

    spell_out(HMM)
    spell_out(xps[-1])
    spell_out(xps[-1].stats)
    spell_out(xps[-1].avrgs)

    assert True  # An assertion for pytest to count
    return HMM, xps  # Return useful stuff


# Non py.test runs:
# HMM, xx, yy, xps = test_L63()
# HMM, xx, yy, xps = test_L96()
