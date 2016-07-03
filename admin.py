from common import *

class Operator:
  def __init__(self,m,model=None,noise=None):
    self.m = m

    if model is None:
      model = lambda x,t,dt: x
    self.model = model

    if noise is None:
      noise = 0
    if noise is 0:
      noise = GaussRV(0,m=m)
    self.noise = noise

class OSSE:
  def __init__(self,f,h,t,X0,**kwargs):
    if not isinstance(X0,GaussRV):
      X0 = GaussRV(**X0)
    if not isinstance(f,Operator):
      f = Operator(**f)
    if not isinstance(h,Operator):
      h = Operator(**h)
    if not isinstance(t,Chronology):
      t = Chronology(**t)
    self.X0 = X0
    self.f  = f
    self.h  = h
    self.t  = t
    for key, value in kwargs.items():
      setattr(self, key, value)

# Serves as dot-references dict
class Settings:
  pass



# DEPRECATED
import inspect
def dyn_import_all(modpath):
  """Incredibly hackish way to load into caller's global namespace"""
  exec('from ' + modpath + ' import *',inspect.stack()[1][0].f_globals)

# DEPRECATED
# Since it's not possible to
# import module as alias
# from alias import v1 v2 (or *)
def dyn_import_all_2(modpath,namespace):
  """Slightly less_hackish. Call as:
    dyn_import_all_2(modpath,globals())"""
  exec('from ' + modpath + ' import *',namespace)
  # Alternatively
  #modm = importlib.import_module(modpath)
  #namespace.update(modm.__dict__)
  # NB: __dict__ contains a lot of defaults



