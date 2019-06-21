# Numerical validation of TLM (derivatives).

##
from dapper import *

eps = 1e-6
def approx_jacob(fun, x, colwise=False):
  """Finite-diff.
  If colwise: fun(E) operates col-wise.
  else      : fun(E) operates row-wise.
  Anyways: return value has shape P-by-M.
  """
  E = x + eps*eye(len(x))
  if colwise: F = ( fun(E.T) - fun(x)[:,None] )   @ inv( (E-x).T )
  else      : F = ( fun(E)   - fun(x)         ).T @ inv(  E-x    ).T
  return F

# Transpose explanation:
# Let A be matrix whose cols are realizations. Abbrev: f=fun.
# Assume: f(A)-f(x) ≈ F @ (A-x).
# Then  : F ≈ [f(A)-f(x)] @ inv(A-x)         (v1)
#           = [f(A')-f(x')]' @ inv(A'-x')'.  (v2)
# Since DAPPER uses dxdt that is made for A', it's easier to apply v2.
# However, TLM should compute F (not F').

def compare(fun, Jacob, x):
  F1 = Jacob(x)
  F2 = approx_jacob(fun,x)
  # rtol=0 => only atol matters.
  return np.allclose(F1, F2, atol=10*eps, rtol=0)


##
from dapper.mods.LotkaVolterra.core import dxdt, TLM, x0
def test_LV(fun=dxdt,Jacob=TLM,x=x0): # capture current values
  assert compare(fun, Jacob, x)

##
from dapper.mods.Lorenz63.core import dxdt, TLM, x0
def test_L63(fun=dxdt,Jacob=TLM,x=x0): # capture current values
  assert compare(fun, Jacob, x)

##
from dapper.mods.Lorenz84.core import dxdt, TLM, x0
def test_L84(fun=dxdt,Jacob=TLM,x=x0): # capture current values
  assert compare(fun, Jacob, x)

##
from dapper.mods.Lorenz95.core import dxdt, TLM, x0
def test_L95(fun=dxdt,Jacob=TLM,x=x0(40)): # capture current values
  assert compare(fun, Jacob, x)

##
# TODO
# from dapper.mods.LorenzUV.core import model_instance
# LUV = model_instance(nU=10,J=4,F=10)
# x = 5 + randn(LUV.M)
# LUV.dfdt
# def test_LUV(fun=,Jacob=,x=): # capture current values
  # assert compare(fun, Jacob, x)

##
# test_LV()
# test_L63()
# test_L84()
# test_L95()

##


