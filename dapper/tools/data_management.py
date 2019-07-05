# Utilities (non-math)

from dapper import *

#########################################
# 
#########################################
from copy import deepcopy, copy

class ResultsTable():
  """Main purpose: collect result data (avrgs) from separate (e.g. parallelized) experiments.

  Supports merging datasets with distinct xticks and labels.

  Load avrgs (array of dicts of fields of time-average statistics)
    from .npz files which also contain arrays 'xticks' and 'labels'.
    Assumes avrgs.shape == (len(xticks),nRepeat,len(labels)).
    But the avrgs of different source files can have entirely different
    xticks, nRepeat, labels. The sources will be properly handled
    also allowing for nan values. This flexibility allows working with
    a patchwork of "inhomogenous" sources.
  Merge (stack) into a TABLE with shape (len(labels),len(xticks)).
    Thus, all results for a given label/xticks are easily indexed,
    and don't even have to be of the same length
    (TABLE[iC,iX] is a list of the avrgs for that (label,absissa)).
  Also provides functions that partition the TABLE,
    (but nowhere near the power of a full database).
  NB: the TABLE is just convenience:
      the internal state of ResultsTable is the dict of datasets.

  Examples:

  # COMPOSING THE DATABASE OF RESULTS
  >>> R = ResultsTable(dirs['data']+'/AdInf/bench_LUV/c_run[1-3]')                    # Load by regex
  >>> R.load(dirs['data']+'/AdInf/bench_LUV/c_run7')                                  # More loading
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

  See AdInf/present_results.py for further examples.
  """

  def __init__(self,*args,**kwargs):
    self.load(*args,**kwargs)

  def load(self,pattern):
    """Load datasets into the "datasets". Call regen_table."""
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

  # NB: Writes to file. Dangerous!
  def write2all(self,**kwargs):
    for key in list(self.datasets):
      tmp = np.load(key)
      np.savez(key, **tmp, **kwargs)

  def rm_dataset(self,pattern):
    for key in list(self.datasets):
      if re.search(pattern,key):
        del self.datasets[key]
    self.regen_table()


  def regen_table(self):
      """Regenerate table.

      From datasets, do:
       - assemble labels and xticks
       - generate corresponding TABLE 
       - validate xlabel, tuning_tag
      """

      # xticks, labels
      # Also see section below for self._scalars.
      # -------------------
      xticks = [] # <--> xlabel
      labels = [] # <--> tuning_tag (if applicable)
      # Grab from datasets
      for ds in self.datasets.values():
        xticks += [ds['xticks']]
        labels += [ds['labels']]
      # Make labels and xticks unique
      xticks = np.sort(np.unique(ccat(*xticks)))
      labels = keep_order_unique(ccat(*labels))

      # Assign
      self.xticks = xticks
      self.labels = labels

      # Init TABLE of avrgs
      # -------------------
      TABLE  = np.empty(self.shape,object)
      for i,j in np.ndindex(TABLE.shape):
        TABLE[i,j] = []
      # Fill TABLE, fields
      fields = set()
      for ds in self.datasets.values():
        for iC,C in enumerate(ds['labels']):
          for iX,X in enumerate(ds['xticks']):
            avrgs = ds['avrgs'][iX,:,iC].tolist()
            TABLE[labels==C,xticks==X][0] += avrgs
            fields |= set().union(*(a.keys() for a in avrgs))
      self.TABLE  = TABLE
      self.fields = fields

      # Non-array attributes (i.e. must be the same in all datasets).
      # --------------------------------------------------------------
      self._scalars = ['xlabel', 'tuning_tag', 'meta'] # Register attributes.
      # NB: If you add a new attribute but not by registering them in _scalars,
      #     then you must also manage it in __deepcopy__().
      scalars = {key:[] for key in self._scalars} # Init
      # Grab from datasets
      for ds in self.datasets.values():
        for key in scalars:
          if key in ds:
            scalars[key] += [ds[key].item()]
      # Assign, having ensured consistency
      for key,vals in scalars.items():
        if vals:
          #def validate_homogeneity(key,vals):
          if not all(vals[0] == x for x in vals):
            raise Exception("The loaded datasets have different %s."%key)
          # Check if some datasets lack the tag.
          if 0<len(vals)<len(self.datasets):
            # Don't bother to warn in len==0 case.
            print("Warning: some of the loaded datasets don't specify %s."%key)
          #validate_homogeneity(key,vals)
          setattr(self,key,vals[0])
        else:
          setattr(self,key,None)
          if key is not 'tuning_tag':
            print("Warning: none of the datasets specify %s."%key)



  @property
  def shape(self):
    return (len(self.labels),len(self.xticks))

  # The number of experiments for a given Config and Setting [iC,iX]
  # may differ (and may be 0). Generate 2D table counting it.
  @property
  def nRepeats(self):
    return np.vectorize(lambda x: len(x))(self.TABLE)


  def rm(self,conditions,INV=False):
    """Delete configs where conditions is True.

    Also, conditions can be indices or a regex.

    Examples:

    - delete if inflation>1.1:   Res.rm('infl 1\.[1-9]')
    - delete if contains tag 50: Res.rm('tag 50')
    """

    if not isinstance(conditions,list):
      conditions = [conditions]

    def check_conds(ind, label):
      for cond in conditions:
        if hasattr(cond,'__call__'): match = cond(label)
        elif isinstance(cond,str):   match = bool(re.search(cond,label))
        else:                        match = ind==cond # works for inds
        if match:
          break
      if INV: match = not match
      return match

    for ds in self.datasets.values():
      inds = [i for i,label in enumerate(ds['labels']) if check_conds(i,label)]
      ds['labels'] = np.delete(ds['labels'], inds)
      ds['avrgs']  = np.delete(ds['avrgs'] , inds, axis=-1)
      ds['avrgs']  = np.ascontiguousarray(ds['avrgs'])
      # for ascontiguousarray, see stackoverflow.com/q/46611571

    self.regen_table()

  def split2(self,cond):
    """Split.

    Example:
    >>> R1, R2 = R.split2(cond) # R1 <-- labels satisfying cond. R2 = R\R1
    """
    C1 = deepcopy(self); C1.rm(cond,INV=True)
    C2 = deepcopy(self); C2.rm(cond)
    return C1, C2

  def split(self,cond):
    """Split. In-place version.

    Example:
    >>> R1 = R.split('mytag') # R1 <-- labels with 'mytag'. R <-- R\R1.
    """
    C1 = deepcopy(self); C1.rm(cond,INV=True)
    self.rm(cond)
    return C1

  def mv(self,regex,sub,inds=None):
    """Rename labels.

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
    """Remove xticks with indices inds."""
    D = self.xticks[inds] # these points will be removed
    for ds in self.datasets.values():
      keep = [i for i,a in enumerate(ds['xticks']) if a not in D]
      ds['xticks'] = ds['xticks'][keep]
      ds['avrgs']  = ds['avrgs'] [keep]
      ds['avrgs']  = np.ascontiguousarray(ds['avrgs'])
    self.regen_table()


  def __deepcopy__(self, memo):
    """Implement __deepcopy__ to make it faster.

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
          'xticks':deepcopy(ds['xticks']),
          'labels':deepcopy(ds['labels'])
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

  def _headr(self):
    return "ResultsTable from datasets matching patterns:\n" + "\n".join(self.patterns)

  def __repr__(self):
    s = self._headr()
    if hasattr(self,'xticks'):
      s +="\n\nfields:\n"      + str(self.fields)     +\
          "\n\nmeta: "         + str(self.meta)       +\
          "\n\nxlabel: "       + str(self.xlabel)     +\
          "\nxticks: "         + str(self.xticks)     +\
          "\n\ntuning_tag: "   + str(self.tuning_tag) +\
          "\nlabels:\n"+\
          "\n".join(["[{0:2d}] {1:s}".format(i,name) for i,name in enumerate(self.labels)])
    return s
        
        


  def field(self,field):
    """Get field.

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
    for iR,iC,iX in np.ndindex(shape):
      try:
        field3D[iR][iC][iX] = self.TABLE[iC,iX][iR][field].val
      except (IndexError,KeyError):
        field3D[iR][iC][iX] = None
    return field3D

  def mean_field(self,field):
    "Extract field"
    field3D = self.field(field)
    field3D = array(field3D,float) # converts None to Nan (not to be counted as fails!)
    mu   = zeros(self.shape)
    conf = zeros(self.shape)
    nSuc = zeros(self.shape,int) # non-fails
    for (iC,iX),_ in np.ndenumerate(mu):
      nRep = self.nRepeats[iC,iX]
      f    = field3D[:nRep,iC,iX]
      f    = f[np.logical_not(np.isnan(f))]
      mu  [iC,iX] = f.mean()                   if len(f)   else None
      conf[iC,iX] = f.std(ddof=1)/sqrt(len(f)) if len(f)>3 else np.nan
      nSuc[iC,iX] = len(f)
    return mu, conf, nSuc


  def print_frame(self,frame):
    "Print single frame"
    for iC,row in enumerate(frame):
      row.insert(0,self.labels[iC])
    print(tabulate_orig.tabulate(frame,headers=self.xticks,missingval=''))

  def print_field(self,field3D):
    "Loop over repetitions, printing Config-by-Setting tables."
    for iR,frame in enumerate(field3D):
      print_c("\nRep: ",iR)
      self.print_frame(frame)


  def print_mean_field(self,field,show_conf=False,show_fail=False,cols=None):
    """Print mean frame, including nRep (#) for each value.

    Don't print nan's when nRep==0 (i.e. print nothing).

    - show_conf: include confidence estimate (±).
    - show_fail: include number of runs that yielded NaNs (#).
      if False but NaNs are present: print NaN for the mean value.

    - cols: indices of columns (experiment xticks) to include.

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
      cols = arange(len(self.xticks))
    if isinstance(cols,slice):
      # Slice
      cols = arange(len(self.xticks))[cols]
    if isinstance(cols,(int,float)):
      # Find closest
      cols = [abs(self.xticks - cols).argmin()]
    if isinstance(cols,tuple):
      # Make range
      cols = np.where( (cols[0]<=self.xticks) & (self.xticks<=cols[1]) )[0]

    # mattr[0]: names
    mattr = [self.labels.tolist()]
    # headr: name \ setting, with filling spacing:
    MxLen = max([len(x) for x in mattr[0]])
    Space = max(0, MxLen - len(self.xlabel) - 2)
    headr = ['name' + " "*(Space//2) + "\\" + " "*-(-Space//2) + self.xlabel + ":"]

    # Fill in stats
    for iX in cols:
      X = self.xticks[iX]
      # Generate column. Include header for cropping purposes
      col = [('{0:@>6g} {1: <'+NF+'s}').format(X,'#')]
      if show_fail: col[0] += (' {0: <'+NF+'s}').format('X')
      if show_conf: col[0] += ' ±'
      for iC in range(len(self.labels)):
        # Format entry
        nRep = nReps[iC][iX]
        val  = mu   [iC][iX]
        c    = conf [iC][iX]
        nF   = nFail[iC][iX]
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


  def plot_1d(self,field='rmse_a',**kwargs):
    fig, ax = plt.gcf(), plt.gca()

    Z = self.mean_field(field)[0]

    labels = self.labels
    title_ = ""
    if self.tuning_tag and not any(np.isnan(self.tuning_vals())):
      labels = self.tuning_vals()
      title_ = "\nLegend: " + self.tuning_tag
      #
      from cycler import cycler
      colors = plt.get_cmap('jet')(linspace(0,1,len(labels)))
      ax.set_prop_cycle(cycler('color',colors))

    lhs = []
    for iC,(row,name) in enumerate(zip(Z,labels)): 
      lhs += [ ax.plot(self.xticks,row,'-o',label=name,**kwargs)[0] ]

    if ax.is_last_row():
      ax.set_xlabel(self.xlabel)
    ax.set_ylabel(field)
    ax.set_title(self._headr() + title_)

    return lhs



  def plot_1d_minz(self,field='rmse_a',**kwargs):
    fig = plt.gcf()
    fig.clear()
    fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3, 1]},num=fig.number)
    ax , ax_   = axs
    lhs, lhs_  = [], []

    c  = deepcopy(plt.rcParams["axes.prop_cycle"])
    c += plt.cycler(marker=(['o','s','x','+','*'   ]*99)[:len(c)])
    c += plt.cycler(ls    =(['-','--','-.',':','--']*99)[:len(c)])
    ax .set_prop_cycle(c)
    ax_.set_prop_cycle(c)

    unique, _, tuning_vals, fieldvals = self.select_optimal(field)

    for group_name, tunings, vals in zip(unique, tuning_vals, fieldvals):
      
      lhs  += [ ax .plot(self.xticks,vals   ,label=group_name,**kwargs)[0] ]
      lhs_ += [ ax_.plot(self.xticks,tunings,label=group_name,**kwargs)[0] ]

    ax_.set_xlabel(self.xlabel)
    ax_.set_ylabel("opt. " + self.tuning_tag)
    ax .set_ylabel(field)
    ax .set_title(self._headr())
    plt.sca(ax)

    return ax, ax_, lhs, lhs_


  def plot_2d(self,field='rmse_a',log=False,cMin=None,cMax=None,
      show_fail=True,minz=True,**kwargs):
    fig, ax = plt.gcf(), plt.gca()

    # Get plotting data
    Z, _, nSuc = self.mean_field(field)
    nFail      = self.nRepeats - nSuc
    if show_fail: pass # draw crosses (see further below)
    else: Z[nFail.astype(bool)] = np.nan # Create nan (white?) pixels

    # Color range limit
    cMin = 0.95*Z.min() if cMin is None else cMin
    cMax = Z.max()      if cMax is None else cMax
    CL   = cMin, cMax

    # Colormap
    cmap = plt.get_cmap('nipy_spectral',200)
    cmap.set_over('w') # white color for out-of-range values
    if log: trfm = mpl.colors.LogNorm  (*CL)
    else:   trfm = mpl.colors.Normalize(*CL)

    # Plot 
    mesh = ax.pcolormesh(Z,
        cmap=cmap,norm=trfm,
        edgecolor=0.0*ones(3),linewidth=0.3,
        **kwargs)

    if show_fail: # draw crosses on top of colored pixels
      d = array([0,1])
      for (i,j), failed in np.ndenumerate(nFail):
        if failed:
          plt.plot(j+d,i+d  ,'w-',lw=2)
          plt.plot(j+d,i+d  ,'k:',lw=1.5)
          plt.plot(j+d,i-d+1,'w-',lw=2)
          plt.plot(j+d,i-d+1,'k:',lw=1.5)

    # Colorbar and its ticks.
    # Caution: very tricky in log-case. Don't mess with this.
    cb = fig.colorbar(mesh,shrink=0.9)
    cb.ax.tick_params(length=4, direction='out',width=1, color='k')
    if log:
      ct = round2sigfig(LogSp(   max(Z.min(),CL[0]), min(CL[1],Z.max()), 10  ), 2)
      ct = [x for x in ct if CL[0] <= x <= CL[1]] # Cannot go outside of clim! Buggy as hell!
      cb.set_ticks(  ct   )
      cb.set_ticklabels(ct)
    else:
      pass
    cb.set_label(field)

    # Mark optimal tuning
    if minz:
      tuning_inds, _, _ = self.minz_tuning(field)
      ax.plot(0.5 + arange(len(self.xticks)), 0.5 + tuning_inds, 'w*' , ms=12)
      ax.plot(0.5 + arange(len(self.xticks)), 0.5 + tuning_inds, 'b:*', ms=5)

    # title
    ax.set_title(self._headr())
    # xlabel
    ax.set_xlabel(self.xlabel)
    # ylabel:
    if self.tuning_tag is None:
      ax.set_ylabel('labels')
      ylbls = self.labels
    else:
      ax.set_ylabel(self.tuning_tag)
      ylbls = self.tuning_vals()

    # Make xticks less dense, if needed
    nXGrid = len(self.xticks)
    step = 1 if nXGrid <= 16 else nXGrid//10
    # Set ticks
    ax.set_xticks(0.5+arange(nXGrid)[::step]); ax.set_xticklabels(self.xticks[::step]);
    ax.set_yticks(0.5+arange(len(ylbls)));     ax.set_yticklabels(ylbls)  

    # Reverse order
    #ax.invert_yaxis()

    return mesh


  def tuning_vals(self,**kwargs):
    return pprop(self.labels, self.tuning_tag, **kwargs)

  def minz_tuning(self,field='rmse_a'):
    Z = self.mean_field(field)[0]
    tuning_inds = np.nanargmin(Z,0)
    tuning_vals = self.tuning_vals()[tuning_inds]
    fieldvals   = Z[tuning_inds,arange(len(tuning_inds))]
    return tuning_inds, tuning_vals, fieldvals 

  def select_optimal(self,field='rmse_a'):
    assert self.tuning_tag
    from numpy import ma

    # Remove tuning tag (repl by spaces to avoid de-alignment in case of 1.1 vs 1.02 for ex).
    pattern  = '(?<!\S)'+self.tuning_tag+':\S*' # (no not whitespace), tuning_tag, (not whitespace)
    repl_fun = lambda m: ' '*len(m.group())     # replace by spaces of same length
    names    = [re.sub(pattern, repl_fun, n) for n in self.labels]
    names    = trim_table(names)
    names    = [n.strip() for n in names]

    Z = self.mean_field(field)[0]

    unique      = OrderedDict.fromkeys(n for n in names)
    tuning_inds = []
    tuning_vals = []
    fieldvals   = []

    for group_name in unique:
      gg           = [i for i, n in enumerate(names) if n==group_name]   # Indices  of group
      Zg           = ma.masked_invalid(Z[gg])                            # Vals     of group, with invalids masked.
      fieldvalsg   = Zg.min(axis=0)                                      # Minima   of group; yields invalid if all are invalid
      tuning_indsg = Zg.argmin(axis=0)                                   # Indices     of minima
      tuning_valsg = self.tuning_vals()[gg][tuning_indsg]                # Tuning vals of minmia
      tuning_valsg = ma.masked_array(tuning_valsg, mask=fieldvalsg.mask) # Apply mask for invalids
      # Append
      tuning_inds += [tuning_indsg]
      tuning_vals += [tuning_valsg]
      fieldvals   += [fieldvalsg]

    return unique, tuning_inds, tuning_vals, fieldvals

def trim_table(list_of_strings):
  """Make (narrow) columns with only whitespace to width 1."""
  # Definitely not an efficient implementation.
  table = list_of_strings
  j = 0
  while True:
    if j==min(len(row) for row in table):
      break
    if all(row[j-1:j+1]=="  " for row in table):    # check if col has 2 spaces
      table = [row[:j]+row[j+1:] for row in table]  # remove column j
    else:
      j += 1
  return table


def pprop(labels,propID,cast=float,fillval=np.nan):
  """Parse property (propID) values from labels.

  Example:
  >>> pprop(R.labels,'infl',float)
  """
  props = []
  for s in labels:
    x = re.search(r'.*'+propID+':(.+?)(\s|$)',s)
    props += [cast(x.group(1))] if x else [fillval]
  return array(props)


