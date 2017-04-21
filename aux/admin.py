from common import *


class Operator(MLR_Print):
  """Class for operators (models)."""
  def __init__(self,m,model=None,noise=None,**kwargs):
    self.m = m

    # None => Identity model
    if model is None:
      model = lambda x,t,dt: x
    self.model = model

    # None/0 => No noise
    if noise is None:
      noise = 0
    if noise is 0:
      noise = GaussRV(0,m=m)
    self.noise = noise

    # Write the rest of parameters
    for key, value in kwargs.items(): setattr(self, key, value)
  
  def __call__(self,*args,**kwargs):
    return self.model(*args,**kwargs)


class OSSE(MLR_Print):
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


class AssimFailedError(RuntimeError):
    pass

from functools import wraps
def DA_Config(da_driver):
  """
  Wraps a da_driver to an instance of the DAC (DA Configuration) class.
  1) inserts arguments to da_driver as attributes in a DAC object, and
  2) wraps assimilate() so as to fail_gently and pre-init stats.

  Features:
   - Provides light-weight interface (makes da_driver very readable).
   - Allows args (and defaults) to be defined in da_driver's signature.
   - assimilate() nested under da_driver.
     => args accessible without unpacking or 'self'.
   - stats initialized by assimilate(),
     not when running da_driver itself => save memory.
   - Involves minimal hacking/trickery.
  """
  # Note: I did consider unifying DA_Config and the DAC class,
  # since they "belong together", but I saw little other benefit,
  # and it would then be harder to use functools.wraps

  f_arg_names = da_driver.__code__.co_varnames[
      :da_driver.__code__.co_argcount]

  @wraps(da_driver)
  def wrapr(*args,**kwargs):
    ############################
    # Validate signature/return
    #---------------------------
    assimilate = da_driver(*args,**kwargs)
    if assimilate.__name__ != 'assimilate':
      raise Exception("DAPPER convention requires that "
      + da_driver.__name__ + " return a function named 'assimilate'.")
    run_args = ('stats','twin','xx','yy')
    if assimilate.__code__.co_varnames[:len(run_args)] != run_args:
      raise Exception("DAPPER convention requires that "+
      "the arguments of 'assimilate' be " + str(run_args))

    ############################
    # Make assimilation caller
    #---------------------------
    def assim_caller(setup,xx,yy):
      name_hook = da_driver.__name__ # for pdesc of progbar
      # Init stats
      stats = Stats(cfg,setup,xx,yy)
      # Put assimilate inside try/catch to allow gentle failure
      try:
        assimilate(stats,setup,xx,yy)
      except AssimFailedError:
        pass # => return output anyway below
      # Return statas
      return stats
    assim_caller.__doc__ = "Calls assimilate() from " +\
        da_driver.__name__ +", passing it the (output) stats object. " +\
        "Returns stats (even if an AssimFailedError is caught)."

    ############################
    # Grab argument names/values
    #---------------------------
    # Process abbreviations
    abbrevs = [('LP','liveplotting')]
    for a,b in abbrevs:
      if a in kwargs:
        kwargs[b] = kwargs[a]
        del kwargs[a]

    cfg = OrderedDict()
    # 1) Insert args into cfg with signature-names.
    for i,val in enumerate(args):
      cfg[f_arg_names[i]] = val
    # 2) Insert kwargs, ordered as in signature.
    for key in f_arg_names[i:]:
      try:
        cfg[key] = kwargs.pop(key)
      except KeyError:
        pass
    # 3) Insert kwargs not listed in signature.
    cfg.update(kwargs)

    ############################
    # Wrap
    #---------------------------
    cfg['da_driver']  = da_driver
    cfg['assimilate'] = assim_caller
    cfg = DAC(cfg)
    return cfg
  return wrapr


class DAC():
  """
  DA Configs (settings).
  Basically just a fancy dict.
  """

  # Defaults
  dflts = {
      'liveplotting': False,
      'store_u'     : False,
      }

  import re
  excluded = \
      list(dflts.keys()) +\
      ['assimilate',re.compile('^_')]

  def __init__(self,odict):
    # Ordering is kept for printing
    self._ordering = odict.keys()
    for key, value in self.dflts.items(): setattr(self, key, value)
    for key, value in      odict.items(): setattr(self, key, value)

  def __repr__(self):
    def format_(key,val):
      # default printing
      s = repr(val)
      # wrap s
      s = key + '=' + s + ", "
      return s
    s = self.da_driver.__name__ + '('
    # Print ordered
    for key in filter_out(self._ordering,*self.excluded,'da_driver'):
      s += format_(key,getattr(self,key))
    # Print remaining
    for key in filter_out(self.__dict__,*self.excluded,*self._ordering):
      s += format_(key,getattr(self,key))
    return s[:-2]+')'

class List_of_Configs(list):
  """List for DAC's"""

  # Print settings
  excluded = DAC.excluded + ['name']
  ordering = ['da_driver','N','upd_a','infl','rot']

  def __init__(self,*args):
    """
    List_of_Configs() -> new empty list
    List_of_Configs(iterable) -> new list initialized from iterable's items
    """
    for cfg in args:
      if isinstance(cfg, DAC):
        self.append(cfg)
      elif isinstance(cfg, list):
        for b in cfg:
          self.append(b)

  def __iadd__(self,val):
    if not hasattr(val,'__iter__'):
      val = [val]
    for item in val:
      self.append(item)
    return self

  def distinct_attrs(self): return self.separate_distinct_common()[0]
  def common_attrs  (self): return self.separate_distinct_common()[1]

  def separate_distinct_common(self):
    """Generate a set of distinct names for DAC's."""
    dist = {}
    comn = {}

    # Find all keys
    keys = {}
    for config in self:
      keys |= config.__dict__.keys()
    keys  = list(keys)

    # Partition attributes into distinct and common
    for key in keys:
      vals = [getattr(config,key,None) for config in self]
      if all(v == vals[0] for v in vals) and len(self)>1:
        comn[key] = vals[0]
      else:
        dist[key] = vals

    # Sort. Do it here so that the same sort is used for
    # repr(DAC_config) and print_averages().
    def sf(item):
      try:    return self.ordering.index(item[0])
      except: return 99
    dist = OrderedDict(sorted(dist.items(), key=sf))

    return dist, comn
   
  def __repr__(self):
    if len(self):
      # Prepare
      s = '<List_of_Configs>:\n'
      dist,comn = self.separate_distinct_common()
      # Distinct
      headr = filter_out(dist,*self.excluded)
      mattr = [dist[key] for key in headr]
      s    += tabulate(mattr, headr)
      # Common
      keys  = filter_out(comn,*self.excluded)
      comn  = {k: formatr(comn[k]) for k in keys}
      s    += "\n---\nCommon: " + str(comn)
    else:
      s = "List_of_Configs([])"
    return s

  def assign_names(self,ow=False,do_tab=True):
    """
    Assign distinct_names to the individual DAC's.
    If ow: do_overwrite.
    """
    # Process attributes into strings 
    names = ['']*len(self) # Init
    dist  = self.distinct_attrs()
    keys  = filter_out(dist, *self.excluded)
    for i,k in enumerate(keys):
      vals  = dist[k]                                              # Get data
      lbls  = '' if i==0 else ' '                                  # Spacing 
      if len(k)<=4: lbls += k + ' '                                # Standard label
      else:         lbls += k[:3] + '~' + k[4] + ' '               # Abbreviated label
      lbls  = [' '*len(lbls) if v is None else lbls for v in vals] # Skip label if val=None
      vals  = typeset(vals,do_tab=True)                            # Format data
      names = [''.join(x) for x in zip(names,lbls,vals)]           # Join
    
    # Assign strings to configs
    for name,config in zip(names,self):
      if ow or not getattr(config,'name',None):
        if not do_tab:
          name = ' '.join(name.split())
        config.name = name
        config._name_auto_gen = True
    

def print_averages(cfgs,Avrgs,attrkeys=(),statkeys=()):
  """
  For c in cfgs:
    Print c[attrkeys], Avrgs[c][statkeys]
  - attrkeys: list of attributes to include.
      - if -1: only print da_driver.
      - if  0: print distinct_attrs
  - statkeys: list of statistics to include.
  """

  # Convert single cfg to list
  if isinstance(cfgs,DAC):
    cfgs  = List_of_Configs(cfgs)
    Avrgs = [Avrgs]

  # Defaults averages
  if not statkeys:
    #statkeys = ['rmse_a','rmv_a','logp_m_a']
    statkeys = ['rmse_a','rmv_a','rmse_f']

  # Defaults attributes
  if not attrkeys:       headr = list(cfgs.distinct_attrs())
  elif   attrkeys == -1: headr = ['da_driver']
  else:                  headr = list(attrkeys)

  # Filter excluded
  headr = filter_out(headr, *cfgs.excluded)
  
  # Get attribute values
  mattr = [cfgs.distinct_attrs()[key] for key in headr]

  # Add separator
  headr += ['|']
  mattr += [['|']*len(cfgs)]

  # Get stats.
  # Format stats_with_conf.
  # Use #'s to avoid auto-cropping by tabulate().
  for key in statkeys:
    col = ['{0:#>9} ±'.format(key)]
    for i in range(len(cfgs)):
      try:
        val  = Avrgs[i][key].val
        conf = Avrgs[i][key].conf
        col.append('{0:#>9.4g} {1: <6g} '.format(val,round2sigfig(conf)))
      except KeyError:
        col.append(' '*16)
    crop= min([s.count('#') for s in col])
    col = [s[crop:]         for s in col]
    headr.append(col[0])
    mattr.append(col[1:])
  table = tabulate(mattr, headr).replace('#',' ')
  print(table)


def formatr(x):
  """Abbreviated formatting"""
  if hasattr(x,'__name__'): return x.__name__
  if isinstance(x,bool)   : return '1' if x else '0'
  if isinstance(x,float)  : return '{0:.5g}'.format(x)
  if x is None: return ''
  return str(x)

def typeset(lst,do_tab):
  """
  Convert lst elements to string.
  If do_tab: pad to min fixed width.
  """
  ss = list(map(formatr, lst))
  if do_tab:
    width = max([len(s)     for s in ss])
    ss    = [s.ljust(width) for s in ss]
  return ss


