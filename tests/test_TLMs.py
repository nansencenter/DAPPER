# Numerical validation of TLM (d2x_dtdx).

##
from dapper import *
import numpy as np

eps = 1e-6
def _allclose(fun, Jacob, x):
    # Eval.
    F1 = Jacob(x)
    F2 = FD_Jac(fun, eps)(x)
    # Compare. Note: rtol=0 => only atol matters.
    return np.allclose(F1, F2, atol=10*eps, rtol=0)

##
from dapper.mods.Lorenz63.core import dxdt, d2x_dtdx, x0
def test_L63(fun=dxdt,Jacob=d2x_dtdx,x=x0): # capture current values
    assert _allclose(fun, Jacob, x)

##
from dapper.mods.Lorenz84.core import dxdt, d2x_dtdx, x0
def test_L84(fun=dxdt,Jacob=d2x_dtdx,x=x0): # capture current values
    assert _allclose(fun, Jacob, x)

##
from dapper.mods.Lorenz96.core import dxdt, d2x_dtdx, x0
def test_L96(fun=dxdt,Jacob=d2x_dtdx,x=x0(40)): # capture current values
    assert _allclose(fun, Jacob, x)

##
from dapper.mods.LorenzUV.core import model_instance
LUV = model_instance(nU=10,J=4,F=10)
def test_LUV(fun=LUV.dxdt,Jacob=LUV.d2x_dtdx,x=LUV.x0): # capture current values
    assert _allclose(fun, Jacob, x)

##
from dapper.mods.Ikeda.core import step, dstep_dx, x0
x0 = randn(x0.shape)
def test_Ikeda(fun=step,Jacob=dstep_dx,x=x0): # capture current values
    fun1   = lambda x: fun  (x,nan,nan)
    Jacob1 = lambda x: Jacob(x,nan,nan)
    assert _allclose(fun1, Jacob1, x)

##
from dapper.mods.KS.core import Model
KS = Model()
def test_KS(fun=KS.dxdt,Jacob=KS.d2x_dtdx,x=KS.x0): # capture current values
    assert _allclose(fun, Jacob, x)

##
from dapper.mods.LotkaVolterra.core import dxdt, d2x_dtdx, x0
def test_LV(fun=dxdt,Jacob=d2x_dtdx,x=x0): # capture current values
    assert _allclose(fun, Jacob, x)


##
# test_LV()
# test_L63()
# test_L84()
# test_L96()

##
