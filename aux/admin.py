from common import *

class Printable:
  def __repr__(self):
    from pprint import pformat
    return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)


class Operator:
  """Class for operators (models)."""
  def __init__(self,m,model=None,noise=None,**kwargs):
    self.m = m

    if model is None:
      model = lambda x,t,dt: x
    self.model = model

    if noise is None:
      noise = 0
    if noise is 0:
      noise = GaussRV(0,m=m)
    self.noise = noise

    # Write the rest of parameters
    for key, value in kwargs.items(): setattr(self, key, value)

class OSSE:
  """Container for OSSE settings."""
  def __init__(self,f,h,t,X0,**kwargs):
    if not isinstance(X0,RV):
      # TODO: Pass through RV instead?
      X0 = GaussRV(**X0)
    if not isinstance(f,Operator):
      f = Operator(**f)
    if not isinstance(h,Operator):
      h = Operator(**h)
    if not isinstance(t,Chronology):
      t = Chronology(**t)
    self.X0 = X0
    self.f  = f
    if h.noise.C.rk != h.noise.C.m:
      raise ValueError("Rank-deficient R not supported")
    self.h  = h
    self.t  = t
    # Write the rest of parameters
    for key, value in kwargs.items(): setattr(self, key, value)

  def __repr__(self):
    s = 'OSSE(' + self.name
    for key,val in self.__dict__.items():
      if key != 'name':
        s += '\n' + key + '=' + str(val)
    return s + ')'

# TODO
#from json import JSONEncoder
#class DAC(JSONEncoder):
class DAC():
  """A fancy dict for DA Configuarations (settings)."""
  def __init__(self,da_driver,*upd_a,**kwargs):
    self.da_driver = da_driver
    if len(upd_a) == 1:
      self.upd_a = upd_a[0]
    elif len(upd_a) > 1:
      raise KeyError('Only upd_a is a non-keyword option')
    # Careful with defaults -- explicit is better than implicit!
    self.liveplotting = False
    # Write the rest of parameters
    for key, value in kwargs.items(): setattr(self, key, value)

  def __repr__(self):
    s = 'DAC(' + self.da_driver.__name__
    for key,val in self.__dict__.items():
      if key == 'da_driver': # Included above
        pass 
      elif key.startswith('_'):
        pass 
      elif key == 'name':
        if not getattr(self,'_name_auto_gen',False):
          s += ', ' + key + "='" + ' '.join(val.split()) + "'"
      else:
        s += ', ' + key + '=' + str(val)
    return s + ')'

def formatr(x):
  """Abbreviated formatting"""
  if hasattr(x,'__name__'): return x.__name__
  if isinstance(x,bool)   : return '1' if x else '0'
  if isinstance(x,float)  : return '{0:.5g}'.format(x)
  if x is None: return ''
  return str(x)

def typeset(lst,do_tab):
  """Convert lst elements to string. If do_tab: pad to min fixed width."""
  ss = list(map(formatr, lst))
  if do_tab:
    width = max([len(s)     for s in ss])
    ss    = [s.ljust(width) for s in ss]
  return ss

class DAC_list(list):
  """List containing DAC's"""
  def __init__(self,*args):
    """Init. Empty or a DAC or a list of cfgs"""
    if args != ():
      for bam in args:
        if isinstance(bam, DAC):
          self._add_DAC(bam)
        elif isinstance(bam, DAC_list):
          assert len(args)==1
          for b in bam: self._add_DAC(b)
        else: raise NotImplementedError
    #else: pass

  def _add_DAC(self,bam):
    """Append a DAC to list"""
    self.append(bam)
    self.set_distinct_names() # care about repeated overhead?
  def add(self,*kargs,**kwargs):
    """Declare and append a DAC"""
    config = DAC(*kargs,**kwargs)
    self._add_DAC(config)

  def set_distinct_names(self):
    """Generate a set of distinct names for DAC's."""
    self.distinct_attrs = {}
    self.common_attrs   = {}

    # Find all keys
    keys = {}
    for config in self:
      keys |= config.__dict__.keys()
    keys -= {'name'}
    keys  = list(keys)
    # Partition attributes into distinct and common
    for key in keys:
      vals = [getattr(config,key,None) for config in self]
      if all(v == vals[0] for v in vals) and len(self)>1:
        self.common_attrs[key] = vals[0]
      else:
        self.distinct_attrs[key] = vals
    # Sort
    def sf(item):
      ordering = ['da_driver','N','upd_a','infl','rot']
      try:    return ordering.index(item[0])
      except: return 99
    self.distinct_attrs = OrderedDict(sorted(self.distinct_attrs.items(), key=sf))
    # Process attributes into strings 
    names = ['']*len(self) # Init
    for i,(key,vals) in enumerate(self.distinct_attrs.items()):
      if i==0:
        key = ''
      else:
        key = ' ' + key[:2] + (key[-1] if len(key)>1 else '') + ':'
      lbls  = [' '*len(key) if v is None else key for v in vals]
      vals  = typeset(vals,do_tab=True)
      names = [''.join(x) for x in zip(names,lbls,vals)]
    # Assign to DAC_list
    self.distinct_names = names
   
  def __repr__(self):
    if len(self):
      headr = self.distinct_attrs.keys()
      mattr = self.distinct_attrs.values()
      s     = tabulate(mattr, headr)
      s    += "\n---\nAll: " + str(self.common_attrs)
    else: s = "DAC_list()"
    return s

  def assign_names(self,ow=False,do_tab=True):
    """Assign distinct_names to the individual DAC's. If ow: do_overwrite."""
    for name,config in zip(self.distinct_names,self):
      if ow or not getattr(config,'name',None):
        if not do_tab:
          name = ' '.join(name.split())
        config.name = name
        config._name_auto_gen = True
    

def assimilate(setup,config,xx,yy):
  """Call config.da_driver(), passing along all arguments."""
  args = locals()
  return config.da_driver(**args)

def simulate(setup):
  """Generate synthetic truth and observations"""
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  # truth
  xx = zeros((chrono.K+1,f.m))
  xx[0] = X0.sample(1)

  # obs
  yy = zeros((chrono.KObs+1,h.m))
  for k,kObs,t,dt in progbar(chrono.forecast_range,desc='Truth & Obs'):
    xx[k] = f.model(xx[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
    if kObs is not None:
      yy[kObs] = h.model(xx[k],t) + h.noise.sample(1)

  return xx,yy

class Bunch(dict):
  def __init__(self,**kw):
    dict.__init__(self,kw)
    self.__dict__ = self

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
  #import importlib
  #modm = importlib.import_module(modpath)
  #namespace.update(modm.__dict__)
  # NB: __dict__ contains a lot of defaults



