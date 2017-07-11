from common import *

class TwinSetup(MLR_Print):
  """
  Container for Twin experiment (OSSE) settings.
  OSSE: "observing system simulation experiment"
  """
  def __init__(self,f,h,t,X0,**kwargs):
    self.f  = f  if isinstance(f,  Operator)   else Operator  (**f)
    self.h  = h  if isinstance(h,  Operator)   else Operator  (**h)
    self.t  = t  if isinstance(t,  Chronology) else Chronology(**t)
    self.X0 = X0 if isinstance(X0, RV)         else RV        (**X0)
    # Write the rest of parameters
    for key, value in kwargs.items():
      setattr(self, key, value)
    # Validation
    if self.h.noise.C==0 or self.h.noise.C.rk!=self.h.noise.C.m:
        raise ValueError("Rank-deficient R not supported.")

  def prepare_setup(self,mat,assimcycles):
    #Extend the assimilation to get 10**4 cycles
    self.t.T=assimcycles*self.t.dtObs
    self.h.noise.C=mat

    return self

class Operator(MLR_Print):
  """
  Container for operators (models).
  """
  def __init__(self,m,model=None,noise=None,**kwargs):
    self.m = m

    # None => Identity model
    if model is None:
      model = Id_op()
      kwargs['jacob'] = Id_mat(m)
    self.model = model

    # None/0 => No noise
    if isinstance(noise,RV):
      self.noise = noise
    else:
      if noise is None: noise = 0
      if np.isscalar(noise):
        self.noise = GaussRV(C=noise,m=m)
      else:
        self.noise = GaussRV(C=noise)

    # Write attributes
    for key, value in kwargs.items():
      setattr(self, key, value)
  
  def __call__(self,*args,**kwargs):
    return self.model(*args,**kwargs)



def DA_Config(da_method):
  """
  Wraps a da_method to an instance of the DAC (DA Configuration) class.
  1) inserts arguments to da_method as attributes in a DAC object, and
  2) wraps assimilator() so as to fail_gently and pre-init stats.

  Features:
   - Provides light-weight interface (makes da_method very readable).
   - Allows args (and defaults) to be defined in da_method's signature.
   - assimilator() nested under da_method.
     => args accessible without unpacking or 'self'.
   - stats initialized by assimilator(),
     not when running da_method itself => save memory.
   - Involves minimal hacking/trickery.
  """
  # Note: I did consider unifying DA_Config and the DAC class,
  # since they "belong together", but I saw little other benefit,
  # and it would then be harder to use functools.wraps

  f_arg_names = da_method.__code__.co_varnames[
      :da_method.__code__.co_argcount]

  @functools.wraps(da_method)
  def wrapr(*args,**kwargs):
    ############################
    # Validate signature/return
    #---------------------------
    assimilator = da_method(*args,**kwargs)
    if assimilator.__name__ != 'assimilator':
      raise Exception("DAPPER convention requires that "
      + da_method.__name__ + " return a function named 'assimilator'.")
    run_args = ('stats','twin','xx','yy')
    if assimilator.__code__.co_varnames[:len(run_args)] != run_args:
      raise Exception("DAPPER convention requires that "+
      "the arguments of 'assimilator' be " + str(run_args))

    ############################
    # Make assimilation caller
    #---------------------------
    def assim_caller(setup,xx,yy):
      name_hook = da_method.__name__ # for pdesc of progbar
      # Init stats
      stats = Stats(cfg,setup,xx,yy)
      # Put assimilator inside try/catch to allow gentle failure
      try:
        assimilator(stats,setup,xx,yy)
      except (AssimFailedError,ValueError) as err:
        msg  = "Caught exception during assimilation. Printing traceback:"
        msg += "\n" + "<"*20 + "\n\n"
        msg += "\n".join(s for s in traceback.format_tb(err.__traceback__))
        msg += "\n" + str(err)
        msg += "\n" + ">"*20 + "\n"
        msg += "Returning stats object in its current (incompleted) state.\n"
        print(msg)
      return stats
    assim_caller.__doc__ = "Calls assimilator() from " +\
        da_method.__name__ +", passing it the (output) stats object. " +\
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
    i   = 0
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
    cfg['da_method']  = da_method
    cfg['assimilate'] = assim_caller
    cfg = DAC(cfg)
    return cfg
  return wrapr


class DAC():
  """
  DA Configs (settings).
  """

  # Defaults
  dflts = {
      'liveplotting': False,
      'store_u'     : False,
      }

  excluded =  ['assimilate',re.compile('^_')]

  def __init__(self,odict):
    # Ordering is kept for printing
    self._ordering = odict.keys()
    for key, value in self.dflts.items(): setattr(self, key, value)
    for key, value in      odict.items(): setattr(self, key, value)

  def update_settings(self,**kwargs):
    """Returns new DAC with new "instance" of the da_method with the updated setting."""
    old = list(self._ordering) + filter_out(self.__dict__,*self._ordering,*self.excluded,'da_method')
    dct = {**{key: getattr(self,key) for key in old}, **kwargs}
    return DA_Config(self.da_method)(**dct)

  def __repr__(self):
    def format_(key,val):
      # default printing
      s = repr(val)
      # wrap s
      s = key + '=' + s + ", "
      return s
    s = self.da_method.__name__ + '('
    # Print ordered
    keys = list(self._ordering) + filter_out(self.__dict__,*self._ordering)
    keys = filter_out(keys,*self.excluded,*self.dflts,'da_method')
    for key in keys:
      s += format_(key,getattr(self,key))
    return s[:-2]+')'

class List_of_Configs(list):
  """List for DAC's"""

  # Print settings
  excluded = DAC.excluded + ['name']
  ordering = ['da_method','N','upd_a','infl','rot']

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
    keys = list(keys)

    # Partition attributes into distinct and common
    for key in keys:
      vals = [getattr(config,key,None) for config in self]
      if all(v == vals[0] for v in vals) and len(self)>1:
        comn[key] = vals[0]
      else:
        dist[key] = vals

    # Sort. Do it here so that the same sort is used for
    # repr(List_of_Configs) and print_averages().
    def sortr(item):
      key = item[0]
      try:
        # Find index in self.ordering.
        # Do chr(65+) to compare alphabetically,
        # for keys not in ordering list.
        return chr(65+self.ordering.index(key))
      except:
        return key.upper()
    dist = OrderedDict(sorted(dist.items(), key=sortr))

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
    names = [config.da_method.__name__+' ' for config in self]
    dist  = self.distinct_attrs()
    keys  = filter_out(dist, *self.excluded,'da_method')
    for i,k in enumerate(keys):
      vals  = dist[k]                                              # Get data
      lbls  = '' if i==0 else ' '                                  # Spacing 
      if len(k)<=4: lbls += k + ' '                                # Standard label
      else:         lbls += k[:3] + '~' + k[-1] + ' '              # Abbreviated label
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
      - if -1: only print da_method.
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
  elif   attrkeys == -1: headr = ['da_method']
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




