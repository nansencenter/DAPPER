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

  Purpose: make a da_method brief and readable. Features:
   - argument treatment: since assimilator() is nested under a da_method,
     this enables builtin argument default systemization (via da_method's signature),
     and enables processing before the assimilation run.
     In contrast to the system of passing dicts, this avoids unpacking.
   - stats auto-initialized here, before calling assimilator()
   - provides fail_gently
  We could have achieved much the same with less hacking (less 'inspect')
  using classes for the da_methods. But that would have required onerous
  read/write via 'self'.
  """

  f_arg_names = da_method.__code__.co_varnames[:da_method.__code__.co_argcount]

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

        def crop_traceback(ERR,lvl):
            msg = []
            try:
              # If IPython, use its coloring functionality
              __IPYTHON__
              from IPython.core.debugger import Pdb
              import traceback as tb
              pdb_instance = Pdb()
              pdb_instance.curframe = inspect.currentframe() # first frame: this one
              for i, frame_lineno in enumerate(tb.walk_tb(ERR.__traceback__)):
                if i<lvl: continue # skip first frame
                msg += [pdb_instance.format_stack_entry(frame_lineno,context=5)]
            except (NameError,ImportError):
              # No coloring
              msg += ["\n".join(s for s in traceback.format_tb(ERR.__traceback__))]
            return msg

        # Put assimilator inside try/catch to allow gentle failure
        try:
          try:
              assimilator(stats,setup,xx,yy)
          except (AssimFailedError,ValueError,np.linalg.LinAlgError) as ERR:
              if getattr(cfg,'fail_gently',True):
                msg  = ["\nCaught exception during assimilation. Printing traceback:"]
                msg += ["<"*20 + "\n"]
                msg += crop_traceback(ERR,1) + [str(ERR)]
                msg += ["\n" + ">"*20]
                msg += ["Returning stats (time series) object in its"+\
                    " current (incompleted) state\n, and resuming program execution.\n"]
                for s in msg:
                  print(s,file=sys.stderr)
              else: # Don't fail gently.
                raise ERR
        except Exception as ERR:
              #print(*crop_traceback(ERR,2), str(ERR))
              # How to avoid duplicate traceback printouts?
              # Don't want to replace 'raise' by 'sys.exit(1)',
              # coz then %debug would start here.
              raise ERR

        return stats

      assim_caller.__doc__ = "Calls assimilator() from " +\
          da_method.__name__ +", passing it the (output) stats object. " +\
          "Returns stats (even if an AssimFailedError is caught)."

      ############################
      # Grab argument names/values
      #---------------------------
      # Process abbreviations, aliases
      abbrevs = [('LP','liveplotting'),('store_intermediate','store_u')]
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


# Adapted from stackoverflow.com/a/3603824
class ImmutableAttributes():
  """
  Freeze (make immutable) attributes of class instance.
  Applies to 
  """
  __isfrozen = False
  __keys     = None
  def __setattr__(self, key, value):
    #if self.__isfrozen and hasattr(self, key):
    if self.__isfrozen and key in self.__keys:
      raise AttributeError(
          "The attribute %r of %r has been frozen."%(key,type(self)))
    object.__setattr__(self, key, value)
  def _freeze(self,keys):
    self.__keys     = keys
    self.__isfrozen = True


class DAC(ImmutableAttributes):
  """
  DA configs (settings).

  This class just contains the parameters grabbed by the DA_Config wrapper.

  NB: re-assigning these would only change their value in this container,
      (i.e. not as they are known by the assimilator() funtions)
      and has therefore been disabled ("frozen").
      However, parameter changes can be made using update_settings().
  """

  # Defaults
  dflts = {
      'liveplotting': False,
      'store_u'     : False,
      }

  excluded =  ['assimilate',re.compile('^_')]

  def __init__(self,odict):
    """Assign dict items to attributes"""
    # Ordering is kept for printing
    self._ordering = odict.keys()
    for key, value in self.dflts.items(): setattr(self, key, value)
    for key, value in      odict.items(): setattr(self, key, value)
    self._freeze(filter_out(odict.keys(),*self.dflts,'name'))

  def update_settings(self,**kwargs):
    """
    Returns new DAC with new "instance" of the da_method with the updated setting.

    Example:
    >>> for iC,C in enumerate(cfgs):
    >>>   cfgs[iC] = C.update_settings(liveplotting=True)
    """
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

  def _is(self,decorated_da_method):
    "Test if cfg is an instance of the decorator of the da_method."
    return self.da_method.__name__ == decorated_da_method.__name__


class List_of_Configs(list):
  """
  List of DA configs.

  Purpose: presentation (facilitate printing tables of attributes, results, etc).
  Also implement += operator for easy use.
  """

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

  def sublist(self,inds):
    """
    List only supports slice indexing.
    This enables accessing (i.e. getitem) by list of inds
    """
    return List_of_Configs([self[i] for i in inds])

  @property
  def da_names(self):
    return [config.da_method.__name__ for config in self]

  # HUGE WASTE OF TIME.
  # Better to rely on manual (but brief) filter_out(xcld) to use gen_names().
  # def distinct_attrs(self,grouped=0):
  #   """
  #   Yields list of the attributes that are distinct (not all the same or absent/None).

  #   grouped:
  #     Does (various levels of) group-wise comparisons (grouping by da_name)
  #     to eliminate some attrs from the list.
  #     NB: Setting grouped>0 can be quite useful, but "comes with no guarantees".
  #         I.e. the behaviour is only strictly predictable for grouped=0.
  #     NB: If grouped>0, then some attributes may be eliminated,
  #         in which case: {common UNION distinct} < {all attributes}.
  #   """
  #   attrs = self.separate_distinct_common()[0]

  #   if grouped>=1: # Eliminate single-appearance attrs that belong to single-appearance names.
  #       names = self.da_names
  #       for key in list(attrs):
  #         nn_inds = [i for i,x in enumerate(attrs[key]) if x is not None]        # not-None indices
  #         if len(nn_inds)==1:                                                    # if  ∃! not-None val
  #           if names.count(names[nn_inds[0]])==1:                                # and ∃! of the corresponding name 
  #             del attrs[key]

  #   if grouped>=2: # Elim if constant within all non-singleton groups.
  #       groups  = list(keep_order_unique(array(names)))                          # groups: unique names
  #       g_inds  = [ [i for i,n in enumerate(names) if n==g] for g in groups ]    # get indices per group 
  #       g_attrs = [self.sublist(inds).distinct_attrs() for inds in g_inds ]      # distinct_attrs per group

  #       # Here be dragons!
  #       for key in list(attrs):
  #         nn_inds  = [i for i,x in enumerate(attrs[key]) if x is not None]       # not-None indices
  #         is_const = []                                                          # list where duplicates were found
  #         for gi,inds in enumerate(g_inds):                                      # Loop over groups
  #           if len(inds)>1:                                                      #   ensure non-singleton group
  #             if all([i in nn_inds for i in inds]):                              #   ensure vals are not all None (globally)
  #               is_const.append( key not in g_attrs[gi] )                        #   check if constant
  #         if len(is_const)>0:                                                    # if non-singleton/None groups were found
  #           if all(is_const):                                                    # if all were constant 
  #             del attrs[key]                                                     # eliminate attribute

  #   if grouped>=3: # Eliminate those that are not distinct in any group.
  #       # Get distinct_attrs per group.
  #       g_keys  = [list(attrs.keys()) for attrs in g_attrs ]                     # use keys only
  #       g_keys  = keep_order_unique(array([a for keys in g_keys for a in keys])) # Merge (flatten, unique)

  #       # Eliminate, but retain ordering.
  #       for key in list(attrs): 
  #         if key not in g_keys: del attrs[key]

  #   return attrs

  def distinct_attrs(self): return self.separate_distinct_common()[0]
  def   common_attrs(self): return self.separate_distinct_common()[1]

  def separate_distinct_common(self):
    """
    Compile the attributes of the DAC's in the List_of_Confgs,
    and partition them in two sets: distinct and common.
    Insert None's for cfgs that don't have that attribute.
    """
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
      if all(v == vals[0] for v in vals): # and len(self)>1:
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

  def gen_names(self,abbrev=4,trim=False,do_tab=False,xcld=[]):

    # 1st column: da_method's names
    columns = self.da_names
    MxWidth = max([len(n) for n in columns])
    columns = [n.ljust(MxWidth) + ' ' for n in columns]

    # Get distinct attributes
    dist  = self.distinct_attrs()
    keys  = filter_out(dist, *self.excluded,'da_method',*xcld)

    # Process attributes into strings 
    for i,k in enumerate(keys):
      vals = dist[k]

      # Make label strings
      A = 4 if abbrev is True else 99 if abbrev==False else abbrev # Set abbrev length A
      if A: lbls = k                     + ':'                     # Standard label
      else: lbls = k[:A-1] + '~' + k[-1] + ':'                     # Abbreviated label
      lbls = ('' if i==0 else ' ') + lbls                          # Column spacing 
      lbls = [' '*len(lbls) if v is None else lbls for v in vals]  # Erase label where val=None

      # Make value strings
      if trim and all(v in [find_1st(vals),None] for v in vals):
        # If all values  are identical (or None): only keep label.
        lbls = [x[:-1] for x in lbls] # Remove colon
        vals = [''     for x in lbls] # Make empty val
      else: # Format data
        vals = typeset(vals,do_tab=True)

      # Form column: join columns, lbls and vals.
      columns = [''.join(x) for x in zip(columns,lbls,vals)]           

    # Undo all tabulation inside column all at once:
    if not do_tab: columns = [" ".join(n.split()) for n in columns]

    return columns

  def assign_names(self,ow=False,do_tab=False):
    """
    Assign distinct_names to the individual DAC's.
    If ow: do_overwrite.
    """
    # Process attributes into strings 
    names = gen_names(self,do_tab)
    
    # Assign strings to configs
    for name,config in zip(names,self):
      t = getattr(config,'name',None)
      if ow is False:
        if t: s = t
      elif ow == 'append':
        if t: s = t+' '+s
      elif ow == 'prepend':
        if t: s = s+' '+t
      config.name = s
    

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
    cfgs     = List_of_Configs(cfgs)
    Avrgs    = [Avrgs]

  # Set excluded attributes
  excluded = list(cfgs.excluded)
  if len(cfgs)==1:
    excluded += list(cfgs[0].dflts)

  # Defaults averages
  if not statkeys:
    #statkeys = ['rmse_a','rmv_a','logp_m_a']
    statkeys = ['rmse_a','rmv_a','rmse_f']

  # Defaults attributes
  if not attrkeys:       headr = list(cfgs.distinct_attrs())
  elif   attrkeys == -1: headr = ['da_method']
  else:                  headr = list(attrkeys)

  # Filter excluded
  headr = filter_out(headr, *excluded)
  
  # Get attribute values
  mattr = [cfgs.distinct_attrs()[key] for key in headr]

  # Add separator
  headr += ['|']
  mattr += [['|']*len(cfgs)]

  # Fill in stats
  for key in statkeys:
    # Generate column, including header (for cropping purposes)
    col = ['{0:@>9} ±'.format(key)]
    for i in range(len(cfgs)):
      # Format entry
      try:
        val  = Avrgs[i][key].val
        conf = Avrgs[i][key].conf
        col.append('{0:@>9.4g} {1: <6g} '.format(val,round2sigfig(conf)))
      except KeyError:
        col.append(' ') # gets filled by tabulate
    # Crop
    crop= min([s.count('@') for s in col])
    col = [s[crop:]         for s in col]
    # Split column into headr/mattr
    headr.append(col[0])
    mattr.append(col[1:])

  # Used @'s to avoid auto-cropping by tabulate().
  table = tabulate(mattr, headr).replace('@',' ')
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


