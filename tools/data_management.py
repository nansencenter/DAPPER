# Utilities (non-math)

from common import *

#########################################
# 
#########################################
from copy import deepcopy, copy

class ResultsTable():
  """
  Main purpose: collect result data (avrgs) from separate (e.g. parallelized) experiments.
  Supports merging datasets with distinct xticks (abscissa) and labels.

  Load avrgs (array of dicts of fields of time-average statistics)
    from .npz files which also contain arrays 'abscissa' and 'labels'.
    Assumes avrgs.shape == (len(abscissa),nRepeat,len(labels)).
    But the avrgs of different source files can have entirely different
    abscissa, nRepeat, labels. The sources will be properly handled
    also allowing for nan values. This flexibility allows working with
    a patchwork of "inhomogenous" sources.
  Merge (stack) into a TABLE with shape (len(labels),len(abscissa)).
    Thus, all results for a given label/abscissa are easily indexed,
    and don't even have to be of the same length
    (TABLE[iC,iS] is a list of the avrgs for that (label,absissa)).
  Also provides functions that partition the TABLE,
    (but nowhere near the power of a full database).
  NB: the TABLE is just convenience:
      the internal state of ResultsTable is the dict of datasets.

  Examples:

  # COMPOSING THE DATABASE OF RESULTS
  >>> R = ResultsTable('data/AdInf/bench_LUV/c_run[1-3]')                         # Load by regex
  >>> R.load('data/AdInf/bench_LUV/c_run7')                                       # More loading
  >>> R.mv(r'tag (\d+)',r'tag\1')                                                 # change "tag 50" to "tag50" => merge such labels (configs)
  >>> R.rm([0, 1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])                            # rm uninteresting labels (configs)
  >>> R.rm('EnKF[^_]')                                                            # rm EnKF but not EnKF_N
  >>> cond = lambda s: s.startswith('EnKF_N') and not re.search('(FULL|CHEAT)',s) # Define more involved criterion
  >>> R1 = R.split('mytag')                                                       # R1: labels with 'mytag'.   R <-- R\R1
  >>> R1, R2 = R.split2(cond)                                                     # R1: labels satisfying cond. R2 = R\R1

  # PRESENTING RESULTS
  >>> R.print_frame(R.field('rmse_a')[iR])                                        # print frame of exprmnt#iR of 'rmse_a' field.
  >>> R.print_field(R.field('rmse_a'))                                            # print all experiment frames
  >>> R.print_frame(R.mean_field('rmse_a')[0].tolist())                           # print frame of mean of field
  >>> R.print_mean_field('rmse_a',show_fail=True,show_conf=False,cols=None)       # This gives more options
  >>> R.plot_mean_field('rmse_a',leg=False)                                       # plot
  >>> check = toggle_lines()                                                      # check boxes

  See AdInf/present_results.py for further examples.
  """

  def __init__(self,*args,**kwargs):
    self.load(*args,**kwargs)

  def load(self,pattern):
    """
    Load datasets into the "datasets",
    Call regen_table.
    """
    self.patterns = getattr(self,'patterns',[]) + [pattern]
    self.datasets = getattr(self,'datasets',OrderedDict())
    # NB: Don't declare any more attributes here; put them in regen_table().

    DIR,regex = os.path.split(pattern)
    keys      = sorted_human(os.listdir(DIR))
    keys      = [os.path.join(DIR,f) for f in keys if re.search(regex,f)]

    if len(keys)==0:
      raise Exception("No files found that match with the given pattern")

    for f in keys:
      if f in self.datasets:
        print("Warning: re-loading",f)
      if 0==os.path.getsize(f):
        print("Encountered placeholder file:",f)
        continue
      self.datasets[f] = dict(np.load(f))
    self.regen_table()

    return self # for chaining

  def rm_dataset(self,pattern):
    for key in list(self.datasets):
      if re.search(pattern,key):
        del self.datasets[key]
    self.regen_table()


  def regen_table(self):
      """
      from datasets, do:
       - assemble labels and abscissa
       - generate corresponding TABLE 
       - validate xlabel, tuneLabel
      """
      # abscissa, labels
      abscissa = [] # <--> xlabel
      labels   = [] # <--> tuneLabel
      # Grab from datasets
      for ds in self.datasets.values():
        abscissa += [ds['abscissa']]
        labels   += [ds['labels']]
      # Make labels and abscissa unique
      abscissa = np.sort(np.unique(ccat(*abscissa)))
      labels   = keep_order_unique(ccat(*labels))
      # Assign
      self.abscissa = abscissa
      self.labels   = labels
      # Init TABLE
      TABLE  = np.empty(self.shape,object)
      for i,j in np.ndindex(TABLE.shape):
        TABLE[i,j] = []
      # Fill TABLE
      fields = set()
      for ds in self.datasets.values():
        for iC,C in enumerate(ds['labels']):
          for iS,S in enumerate(ds['abscissa']):
            avrgs = ds['avrgs'][iS,:,iC].tolist()
            TABLE[labels==C,abscissa==S][0] += avrgs
            fields |= set().union(*(a.keys() for a in avrgs))
      self.TABLE  = TABLE
      self.fields = fields

      # Non-array attributes (i.e. must be the same in all datasets).
      self._scalars = ['xlabel', 'tuneLabel'] # Register attributes.
      # NB: If you add a new attribute but not by registering them in _scalars,
      #     then you must also manage it in __deepcopy__().
      scalars = {key:[] for key in self._scalars} # Init
      # Ensure consistency
      def validate_homogeneity(key,vals):
        if not all(vals[0] == x for x in vals):
          raise Exception("The loaded datasets have different %s."%key)
        # Check if some datasets lack the tag.
        if 0<len(vals)<len(self.datasets):
          # Don't bother to warn in len==0 case.
          print("Warning: some of the loaded datasets don't specify %s."%key)
      # Grab from datasets
      for ds in self.datasets.values():
        for key in scalars:
          if key in ds:
            scalars[key] += [ds[key].item()]
      # Assign
      for key,vals in scalars.items():
        if vals:
          validate_homogeneity(key,vals)
          setattr(self,key,vals[0])
        else:
          setattr(self,key,None)
          if key is not 'tuneLabel':
            print("Warning: none of the datasets specify %s."%key)



  @property
  def shape(self):
    return (len(self.labels),len(self.abscissa))

  # The number of experiments for a given Config and Setting [iC,iS]
  # may differ (and may be 0). Generate 2D table counting it.
  @property
  def nRepeats(self):
    return np.vectorize(lambda x: len(x))(self.TABLE)


  def rm(self,cond,INV=False):
    """
    Delete configs where cond is True.
    Also, cond can be indices or a regex.
    Examples:
    delete if inflation>1.1:   Res.rm('infl 1\.[1-9]')
    delete if contains tag 50: Res.rm('tag 50')
    """

    if isinstance(cond,int):
      cond = [cond]

    def _cond(name):
      if hasattr(cond,'__call__'):
        match = cond(name)
      elif isinstance(cond,str):
        match = bool(re.search(cond,name))
      else: # assume indices
        match = name in self.labels[cond]
      # Use xnor to inverse (if INV)
      return not (not(INV) ^ match) 

    for ds in self.datasets.values():
      ii = [i for i,name in enumerate(ds['labels']) if _cond(name)]
      ds['labels'] = np.delete(ds['labels'], ii)
      ds['avrgs']  = np.delete(ds['avrgs'] , ii, axis=-1)
      # stackoverflow.com/q/46611571
      ds['avrgs']  = np.ascontiguousarray(ds['avrgs'])

    self.regen_table()

  def split2(self,cond):
    """
    Split.
    Example:
    >>> R1 = R.split('mytag') # R1 <-- labels with 'mytag'. R <-- R\R1.
    """
    C1 = deepcopy(self); C1.rm(cond,INV=True)
    C2 = deepcopy(self); C2.rm(cond)
    return C1, C2

  def split(self,cond):
    """
    Split. In-place version.
    Example:
    >>> R1, R2 = R.split2(cond) # R1 <-- labels satisfying cond. R2 = R\R1
    """
    C1 = deepcopy(self); C1.rm(cond,INV=True)
    self.rm(cond)
    return C1

  def mv(self,regex,sub,inds=None):
    """
    Rename labels.
    sub:   substitution pattern
    inds:  restrict to these inds of table's labels
    """
    if isinstance(inds,int): inds = [inds]
    for ds in self.datasets.values():
      ds['labels'] = list(ds['labels']) # coz fixed string limits
      for i,cfg in enumerate(ds['labels']):
        if inds is None or cfg in self.labels[inds]:
          ds['labels'][i] = re.sub(regex, sub, cfg)
    self.regen_table()

  def rm_abcsissa(self,inds):
    """
    Remove abscissa with indices inds.
    """
    D = self.abscissa[inds] # these points will be removed
    for ds in self.datasets.values():
      keep = [i for i,a in enumerate(ds['abscissa']) if a not in D]
      ds['abscissa'] = ds['abscissa'][keep]
      ds['avrgs']    = ds['avrgs']   [keep]
      ds['avrgs']    = np.ascontiguousarray(ds['avrgs'])
    self.regen_table()


  def __deepcopy__(self, memo):
    """
    Implement __deepcopy__ to make it faster.

    We only need to copy the datasets.
    Then regen_table essentially re-inits the object.

    The speed-up is achieved by stopping the 'deep' copying
    at the level of the arrays containing the avrgs.
      This is admissible because the entries of avrgs
      should never be modified, only deleted.
    """
    cls = self.__class__
    new = cls.__new__(cls)
    memo[id(self)] = new
    new.patterns = deepcopy(self.patterns)
    new.datasets = OrderedDict()

    for k, ds in self.datasets.items():
      # deepcopy
      new.datasets[k] = {
          'abscissa':deepcopy(ds['abscissa']),
          'labels'  :deepcopy(ds['labels'])
          }
      for tag in self._scalars:
        if tag in ds:
          new.datasets[k][tag] = deepcopy(ds[tag])
      # 'shallow' copy for avrgs:
      new.datasets[k]['avrgs'] = np.empty(ds['avrgs'].shape,dict)
      for idx, avrg in np.ndenumerate(ds['avrgs']):
        new.datasets[k]['avrgs'][idx] = copy(avrg)

    new.regen_table()
    return new

  def __len__(self):
    # len(self) == len(self.labels) == len(self.TABLE)
    return len(self.TABLE)

  def __repr__(self):
    s = "ResultsTable based on datasets matching patterns:\n" + "\n".join(self.patterns)
    if hasattr(self,'abscissa'):
      s +="\n\nfields:\n"     + str(self.fields)    +\
          "\n\nxlabel: "      + str(self.xlabel)    +\
          "\nabscissa: "      + str(self.abscissa)  +\
          "\n\ntuneLabel: "   + str(self.tuneLabel) +\
          "\nlabels:\n"+\
          "\n".join(["[{0:2d}] {1:s}".format(i,name) for i,name in enumerate(self.labels)])
    return s
        
        


  def field(self,field):
    """
    Extract a given field from TABLE.
    Insert in 3D list "field3D",
    but with a fixed shape (like an array), where empty <--> None.
    Put iRepeat dimension first,
    so that repr(field3D) prints nRepeats.max() 2D-tables.
    """
    shape   = (self.nRepeats.max(),)+self.shape
    field3D = np.full(shape, -999.9) # Value should be overwritten below
    field3D = field3D.tolist()
    # Lists leave None as None, as opposed to a float ndarray.
    # And Tabulate uses None to identify 'missingval'. 
    # So we stick with lists here, to be able to print directly, e.g.
    # Results.print_field(Results.field('rmse_a')
    for iR,iC,iS in np.ndindex(shape):
      try:
        field3D[iR][iC][iS] = self.TABLE[iC,iS][iR][field].val
      except (IndexError,KeyError):
        field3D[iR][iC][iS] = None
    return field3D

  def mean_field(self,field):
    "Extract field"
    field3D = self.field(field)
    field3D = array(field3D,float) # converts None to Nan (not to be counted as fails!)
    mu   = zeros(self.shape)
    conf = zeros(self.shape)
    nSuc = zeros(self.shape,int) # non-fails
    for (iC,iS),_ in np.ndenumerate(mu):
      nRep = self.nRepeats[iC,iS]
      f    = field3D[:nRep,iC,iS]
      f    = f[np.logical_not(np.isnan(f))]
      mu  [iC,iS] = f.mean()                   if len(f)   else None
      conf[iC,iS] = f.std(ddof=1)/sqrt(len(f)) if len(f)>3 else np.nan
      nSuc[iC,iS] = len(f)
    return mu, conf, nSuc


  def print_frame(self,frame):
    "Print single frame"
    for iC,row in enumerate(frame):
      row.insert(0,self.labels[iC])
    print(tabulate_orig.tabulate(frame,headers=self.abscissa,missingval=''))

  def print_field(self,field3D):
    "Loop over repetitions, printing Config-by-Setting tables."
    for iR,frame in enumerate(field3D):
      print_c("\nRep: ",iR)
      self.print_frame(frame)


  def print_mean_field(self,field,show_conf=False,show_fail=False,cols=None):
    """
    Print mean frame, including nRep (#) for each value.
    Don't print nan's when nRep==0 (i.e. print nothing).
    show_conf: include confidence estimate (±).
    show_fail: include number of runs that yielded NaNs (#).
               if False but NaNs are present: print NaN for the mean value.
    s: indices of columns (experiment abscissa) to include.
         - Default          : all
         - tuple of length 2: value range
         - a number         : closest match
    """

    mu, conf, nSuc = self.mean_field(field)
    nReps = self.nRepeats
    nFail = nReps-nSuc

    # Num of figures required to write nRepeats.max()
    NF = str(int(floor(log10(nReps.max()))+1))

    # Set mean values to NaN wherever NaNs are present
    if not show_fail: mu[nFail.astype(bool)] = np.nan

    # Determine columns to print
    if cols is None:
      # All
      cols = arange(len(self.abscissa))
    if isinstance(cols,slice):
      # Slice
      cols = arange(len(self.abscissa))[cols]
    if isinstance(cols,(int,float)):
      # Find closest
      cols = [abs(self.abscissa - cols).argmin()]
    if isinstance(cols,tuple):
      # Make range
      cols = np.where( (cols[0]<=self.abscissa) & (self.abscissa<=cols[1]) )[0]

    # mattr[0]: names
    mattr = [self.labels.tolist()]
    # headr: name \ setting, with filling spacing:
    MxLen = max([len(x) for x in mattr[0]])
    Space = max(0, MxLen - len(self.xlabel) - 2)
    headr = ['name' + " "*(Space//2) + "\\" + " "*-(-Space//2) + self.xlabel + ":"]

    # Fill in stats
    for iS in cols:
      S = self.abscissa[iS]
      # Generate column. Include header for cropping purposes
      col = [('{0:@>6g} {1: <'+NF+'s}').format(S,'#')]
      if show_fail: col[0] += (' {0: <'+NF+'s}').format('X')
      if show_conf: col[0] += ' ±'
      for iC in range(len(self.labels)):
        # Format entry
        nRep = nReps[iC][iS]
        val  = mu   [iC][iS]
        c    = conf [iC][iS]
        nF   = nFail[iC][iS]
        if nRep:
          s = ('{0:@>6.3g} {1: <'+NF+'d} '    ).format(val,nRep)
          if show_fail: s += ('{0: <'+NF+'d} ').format(nF)
          if show_conf: s += ('{0: <6g} '     ).format(c)
        else:
          s = ' ' # gets filled by tabulate
        col.append(s) 
      # Crop
      crop= min([s.count('@') for s in col])
      col = [s[crop:]         for s in col]
      # Split column into headr/mattr
      headr.append(col[0])
      mattr.append(col[1:])

    # Used @'s to avoid auto-cropping by tabulate().
    print(tabulate(mattr,headr,inds=False).replace('@',' '))


  def plot_1d(self,field,legloc='best',**kwargs):
    fig, ax = plt.gcf(), plt.gca()

    Z = self.mean_field(field)[0]

    if self.tuneLabel:
      from cycler import cycler
      colors = plt.get_cmap('jet')(linspace(0,1,len(self.labels)))
      ax.set_prop_cycle(cycler('color',colors))

    lhs = []
    for iC,(row,name) in enumerate(zip(Z,self.labels)): 
      lhs += [ ax.plot(self.abscissa,row,'-o',label=name,**kwargs)[0] ]

    ax.set_xlabel(self.xlabel)
    ax.set_ylabel(field)
    ax.set_title("datasets matching pattern:\n" + "\n".join(self.patterns))
    if legloc: ax.legend(loc=legloc)

    # Set xticks from abscissa data.
    #xt = self.abscissa
    #if len(xt)>15:
      #xt = xt[ceil(linspace(0,len(xt)-1,9)).astype(int)]
    #ax.set_xticks(xt)
    #ax.set_xticklabels(xt)

    return lhs


  def plot_2d(self,field,log=False,cMax=None,**kwargs):
    fig, ax = plt.gcf(), plt.gca()

    Z = self.mean_field(field)[0]
#tuneLabel = 'loc_rad' # 'loc_rad', 'rot'
#tuneVals  = R.pprop(tuneLabel,float)

    cMax = Z.max() if cMax is None else cMax

    # Colormap
    cmap = plt.get_cmap('nipy_spectral',200)
    cmap.set_over('w')
    if log: trfm = mpl.colors.LogNorm(0.95*Z.min(), cMax)
    else:   trfm = None

    # 
    mesh = ax.pcolormesh(Z,cmap=cmap,norm=trfm)

    # Colorbar
    #cb = fig.colorbar(mesh,shrink=0.9)
    #l = mpl.ticker.AutoLocator()
    #l.create_dummy_axis()
    #l.tick_values(, 2)


    ct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 4, 5]
    cb.set_ticks(ct); cb.set_ticklabels(ct)
    cb.ax.tick_params(length=4, direction='out',width=1, colors='k')

    ax.set_xticks(0.5+arange(len(self.abscissa))); ax.set_xticklabels(self.abscissa); ax.set_xlabel(self.xlabel)
    ax.set_yticks(0.5+arange(len(tuneVals     ))); ax.set_yticklabels(tuneVals)  

    plt.xlabel(self.xlabel)
    if self.tuneLabel is not None:
      tuneLabel = 'Lag' # 'loc_rad', 'rot'
      tuneVals  = self.pprop(tuneLabel,float)
      plt.ylabel(tuneLabel)

    plt.title("datasets matching pattern:\n" + "\n".join(self.patterns))

    return mesh


  def pprop(self,propID,cast=float,fillval=np.nan):
    """
    Parse property (propID) values from labels.
    Example:
    >>> pprop(R.labels,'infl',float)
    """
    props = []
    for s in self.labels:
      x = re.search(r'.*'+propID+':(.+?)(\s|$)',s)
      props += [cast(x.group(1))] if x else [fillval]
    return array(props)




