"""Define high-level objects frequently used in DAPPER."""

from dapper import *

class HiddenMarkovModel(NestedPrint):
  """Container for attributes of a Hidden Markov Model (HMM).
  
  This container contains the specification of a "twin experiment",
  i.e. an "OSSE (observing system simulation experiment)".
  """

  def __init__(self,Dyn,Obs,t,X0,**kwargs):
    self.Dyn = Dyn if isinstance(Dyn, Operator)   else Operator  (**Dyn)
    self.Obs = Obs if isinstance(Obs, Operator)   else Operator  (**Obs)
    self.t   = t   if isinstance(t  , Chronology) else Chronology(**t)
    self.X0  = X0  if isinstance(X0 , RV)         else RV        (**X0)

    # Assign name by file (using inspect magic)
    # Fails if used after running a script from the model dir (e.g. demo.py),
    # (but only then?). Buggy?
    name = inspect.getfile(inspect.stack()[1][0])
    self.name = os.path.relpath(name,'mods/')

    # Write the rest of parameters
    de_abbreviate(kwargs, [('LP','liveplotters')])
    for key, value in kwargs.items():
      setattr(self, key, value)

    # Allow running LETKF, SL_EAKF etc without localization
    if not hasattr(self.Obs,"localizer"):
      self.Obs.localizer = no_localization(self.Nx, self.Ny)

    # Validation
    if self.Obs.noise.C==0 or self.Obs.noise.C.rk!=self.Obs.noise.C.M:
        raise ValueError("Rank-deficient R not supported.")
  
  # ndim shortcuts
  @property
  def Nx(self): return self.Dyn.M
  @property
  def Ny(self): return self.Obs.M

  # Print options
  ordering = ['Dyn','Obs','t','X0']


class Operator(NestedPrint):
  """Container for operators (models)."""
  def __init__(self,M,model=None,noise=None,**kwargs):
    self.M = M

    # None => Identity model
    if model is None:
      model = Id_op()
      kwargs['jacob'] = Id_mat(M)
    self.model = model

    # None/0 => No noise
    if isinstance(noise,RV):
      self.noise = noise
    else:
      if noise is None: noise = 0
      if np.isscalar(noise):
        self.noise = GaussRV(C=noise,M=M)
      else:
        self.noise = GaussRV(C=noise)

    # Write attributes
    for key, value in kwargs.items():
      setattr(self, key, value)
  
  def __call__(self,*args,**kwargs):
    return self.model(*args,**kwargs)

  # Print options
  ordering = ['M','model','noise']



def DA_Config(da_method):
  """Wraps a da_method to an instance of the DAC (DA Configuration) class.

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
      run_args = ('stats','HMM','xx','yy')
      if assimilator.__code__.co_varnames[:len(run_args)] != run_args:
        raise Exception("DAPPER convention requires that "+
        "the arguments of 'assimilator' be " + str(run_args))

      ############################
      # Make assimilation caller
      #---------------------------
      def assim_caller(HMM,xx,yy):
        name_hook = da_method.__name__ # for pdesc of progbar

        # Init stats
        stats = Stats(cfg,HMM,xx,yy)

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
              assimilator(stats,HMM,xx,yy)
          except (AssimFailedError,ValueError,np.linalg.LinAlgError) as ERR:
              if getattr(cfg,'fail_gently',rc['fail_gently']):
                msg  = ["\n\nCaught exception during assimilation. Printing traceback:"]
                msg += ["<"*20 + "\n"]
                msg += crop_traceback(ERR,1) + [str(ERR)]
                msg += ["\n" + ">"*20]
                msg += ["Returning stats (time series) object in its "+\
                    "current (incompleted) state,\nand resuming program execution.\n"+\
                    "Turn off the fail_gently attribute of the DA config to fully raise the exception.\n"]
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
      de_abbreviate(kwargs, [('LP','liveplotting')])

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



class DAC(ImmutableAttributes):
  """DA configs (settings).

  This class just contains the parameters grabbed by the DA_Config wrapper.

  NB: re-assigning these would only change their value in this container,
      (i.e. not as they are known by the assimilator() funtions)
      and has therefore been disabled ("frozen").
      However, parameter changes can be made using update_settings().
  """

  # Defaults
  dflts = {
      'liveplotting': rc['liveplotting_enabled'],
      'store_u'     : rc['store_u'],
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

  def __eq__(self, config):
    prep = lambda obj: {k:v for k,v in obj.__dict__.items() if
        k!="assimilate" and not k.startswith("_")}
    eq = prep(self)==prep(config)
    # EDIT: deepdiff causes horrible input bug with QtConsole.
    # eq = not DeepDiff(self,config) # slow, but works well.
    return eq

  def _is(self,decorated_da_method):
    "Test if cfg is an instance of the decorator of the da_method."
    return self.da_method.__name__ == decorated_da_method.__name__


class List_of_Configs(list):
  """List of DA configs.

  This class is quite hackey. But convenience is king for its purposes:
   - pretty printing (using common/distinct attrs)
   - make += accept (a single) item
   - unique kw.
   - indexing with lists.
   - searching for indices by attributes [inds()].
  """

  # Print settings
  excluded = DAC.excluded + ['name']
  ordering = ['da_method','N','upd_a','infl','rot']

  def __init__(self,*args,unique=False):
    """
    List_of_Configs() -> new empty list
    List_of_Configs(iterable) -> new list initialized from iterable's items

    If unique: don't append duplicate entries.
    """
    self.unique = unique
    for cfg in args:
      if isinstance(cfg, DAC):
        self.append(cfg)
      elif isinstance(cfg, list):
        for b in cfg:
          self.append(b)

  def __iadd__(self,cfg):
    if not hasattr(cfg,'__iter__'):
      cfg = [cfg]
    for item in cfg:
      self.append(item)
    return self

  def append(self,cfg):
    "Implemented in order to support 'unique'"
    if self.unique and cfg in self:
      return
    else:
      super().append(cfg)

  def __getitem__(self, keys):
    """Implement indexing by a list"""
    try:              B=List_of_Configs([self[k] for k in keys]) # list
    except TypeError: B=list.__getitem__(self, keys)             # int, slice
    return B


  # NB: In principle, it is possible to list fewer attributes as distinct,
  #     by using groupings. However, doing so intelligently is difficult,
  #     and I wasted a lot of time trying. So don't go there...
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

      try:
        allsame = all(v == vals[0] for v in vals) # and len(self)>1:
      except ValueError:
        allsame = False

      if allsame: comn[key] = vals[0]
      else:       dist[key] = vals

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

  def distinct_attrs(self): return self.separate_distinct_common()[0]
  def   common_attrs(self): return self.separate_distinct_common()[1]


  def __repr__(self):
    if len(self):
      # Prepare
      s = '<List_of_Configs> with attributes:\n'
      dist,comn = self.separate_distinct_common()
      # Distinct
      headr = filter_out(dist,*self.excluded)
      mattr = [dist[key] for key in headr]
      s    += tabulate(mattr, headr)
      # Common
      keys  = filter_out(comn,*self.excluded)
      comn  = {k: formatr(comn[k]) for k in keys}
      s    += "\n---\nCommon attributes:\n" + str(AlignedDict(comn))
    else:
      s = "List_of_Configs([])"
    return s

  @property
  def da_names(self):
    return [config.da_method.__name__ for config in self]

  def gen_names(self,abbrev=4,trim=False,tab=False,xcld=[]):

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
        vals = typeset(vals,tab=True)

      # Form column: join columns, lbls and vals.
      columns = [''.join(x) for x in zip(columns,lbls,vals)]           

    # Undo all tabulation inside column all at once:
    if not tab: columns = [" ".join(n.split()) for n in columns]

    return columns

  def assign_names(self,ow=False,tab=False):
    """
    Assign distinct_names to the individual DAC's.
    If ow: do_overwrite.
    """
    # Process attributes into strings 
    names = self.gen_names(tab)
    
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


  def inds(self,strict=True,da=None,**kw):
    """Find indices of configs with attributes matching the kw dict.
     - strict: If True, then configs lacking a requested attribute will match.
     - da: the da_method.
     """

    def fill(fillval):
      return 'empties_dont_match' if strict else fillval

    def matches(C, kw):
      kw_match = all( getattr(C,k,fill(v))==v for k,v in kw.items())
      da_match = True if da is None else C._is(da)
      return (kw_match and da_match)

    return [i for i,C in enumerate(self) if matches(C,kw)]


    

def _print_averages(cfgs,avrgs,attrkeys=(),statkeys=()):
  """Pretty print the structure containing the averages.

  Essentially, for c in cfgs:
    Print c[attrkeys], avrgs[c][statkeys]

  - attrkeys: list of attributes to include.
      - if -1: only print da_method.
      - if  0: print distinct_attrs
  - statkeys: list of statistics to include.
  """
  # Convert single cfg to list
  if isinstance(cfgs,DAC):
    cfgs     = List_of_Configs(cfgs)
    avrgs    = [avrgs]

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
        val  = avrgs[i][key].val
        conf = avrgs[i][key].conf
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
  return table

@functools.wraps(_print_averages)
def print_averages(*args,**kwargs):
  print(_print_averages(*args,**kwargs))


def formatr(x):
  """Abbreviated formatting"""
  if hasattr(x,'__name__'): return x.__name__
  if isinstance(x,bool)   : return '1' if x else '0'
  if isinstance(x,float)  : return '{0:.5g}'.format(x)
  if x is None: return ''
  return str(x)

def typeset(lst,tab):
  """
  Convert lst elements to string.
  If tab: pad to min fixed width.
  """
  ss = list(map(formatr, lst))
  if tab:
    width = max([len(s)     for s in ss])
    ss    = [s.ljust(width) for s in ss]
  return ss


