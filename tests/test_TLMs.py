"""Numerical validation of TLM (d2x_dtdx)."""

import numpy as np
import dapper as dpr
from dapper.tools.math import FD_Jac

EPS = 1e-6


def _allclose(fun, jacob, x):
    # Eval.
    jac1 = jacob(x)
    jac2 = FD_Jac(fun, EPS)(x)
    # Compare. Note: rtol=0 => only atol matters.
    return np.allclose(jac1, jac2, atol=10*EPS, rtol=0)


def test_L63():
    from dapper.mods.Lorenz63.core import dxdt, d2x_dtdx, x0
    assert _allclose(dxdt, d2x_dtdx, x0)


def test_L84():
    from dapper.mods.Lorenz84.core import dxdt, d2x_dtdx, x0
    assert _allclose(dxdt, d2x_dtdx, x0)


def test_L96():
    from dapper.mods.Lorenz96.core import dxdt, d2x_dtdx, x0
    assert _allclose(dxdt, d2x_dtdx, x0(40))


def test_LUV():
    from dapper.mods.LorenzUV.core import model_instance
    LUV = model_instance(nU=10, J=4, F=10)
    assert _allclose(LUV.dxdt, LUV.d2x_dtdx, LUV.x0)


def test_Ikeda():
    from dapper.mods.Ikeda.core import step, dstep_dx, x0
    x0 = dpr.randn(x0.shape)
    def fun1(x): return step(x, np.nan, np.nan)
    def Jacob1(x): return dstep_dx(x, np.nan, np.nan)
    assert _allclose(fun1, Jacob1, x0)


def test_KS():
    from dapper.mods.KS.core import Model
    KS = Model()
    assert _allclose(KS.dxdt, KS.d2x_dtdx, KS.x0)


def test_LV():
    from dapper.mods.LotkaVolterra.core import dxdt, d2x_dtdx, x0
    assert _allclose(dxdt, d2x_dtdx, x0)
