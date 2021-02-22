"""Numerical validation of TLM (d2x_dtdx)."""

import numpy as np

from dapper.mods.integration import FD_Jac

EPS = 1e-6


def _allclose(fun, jacob, x):
    # Eval.
    jac1 = jacob(x)
    jac2 = FD_Jac(fun, EPS)(x)
    # Compare. Note: rtol=0 => only atol matters.
    return np.allclose(jac1, jac2, atol=10*EPS, rtol=0)


def test_L63():
    from dapper.mods.Lorenz63 import d2x_dtdx, dxdt, x0
    assert _allclose(dxdt, d2x_dtdx, x0)


def test_L84():
    from dapper.mods.Lorenz84 import d2x_dtdx, dxdt, x0
    assert _allclose(dxdt, d2x_dtdx, x0)


def test_L96():
    from dapper.mods.Lorenz96 import d2x_dtdx, dxdt, x0
    assert _allclose(dxdt, d2x_dtdx, x0(40))


def test_LUV():
    from dapper.mods.LorenzUV import model_instance
    LUV = model_instance(nU=10, J=4, F=10)
    assert _allclose(LUV.dxdt, LUV.d2x_dtdx, LUV.x0)


def test_Ikeda():
    from dapper.mods.Ikeda import dstep_dx, step, x0
    x0 = np.random.randn(*x0.shape)
    def fun1(x): return step(x, np.nan, np.nan)
    def Jacob1(x): return dstep_dx(x, np.nan, np.nan)
    assert _allclose(fun1, Jacob1, x0)


def test_KS():
    from dapper.mods.KS import Model
    KS = Model()
    assert _allclose(KS.dxdt, KS.d2x_dtdx, KS.x0)


def test_LV():
    from dapper.mods.LotkaVolterra import d2x_dtdx, dxdt, x0
    assert _allclose(dxdt, d2x_dtdx, x0)


def test_VL20():
    from dapper.mods.VL20 import model_instance
    VL20 = model_instance(nX=18, F=10, G=10)
    assert _allclose(VL20.dxdt, VL20.d2x_dtdx, VL20.x0)
