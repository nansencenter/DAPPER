# Utilities (non-math)

from common import *

#########################################
# 
#########################################
from copy import deepcopy, copy

class ResultsTable():
  """
  Load avrgs (array of dicts of fields of time-average statistics)
    from .npz files which also contain arrays 'abscissa' and 'labels'.
    Assumes avrgs.shape == (len(abscissa),nRepeat,len(labels)).
  Merge (stack) into a TABLE with shape (len(labels),len(abscissa)).
    THUS, all results for a given label/abscissa are easily indexed,
    and don't even have to be of the same length
    (TABLE[iC,iS] is a list of the avrgs for that (label,absissa)).
  Also provides functions taht partitions the TABLE.

  NB: the TABLE is just convenience:
      the internal state of ResultsTable is the dict of datasets.

  Examples:

  # COMPOSING THE DATABASE OF RESULTS
  # Res = ResultsTable('data/AdInf/bench_LXY/c_run[1-3]')                       # Load by regex
  # Res.load('data/AdInf/bench_LXY/c_run7')                                     # More loading
  # Res.mv(r'tag (\d+)',r'tag\1')                                               # change "tag 50" to "tag50" => merge such configs
  # Res.rm([0, 1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])                          # rm uninteresting configs
  # Res.rm('EnKF[^_]')                                                          # rm EnKF but not EnKF_N
  # cond = lambda s: s.startswith('EnKF_N') and not re.search('(FULL|CHEAT)',s) # Define more involved criterion
  # R2, Res = Res.split(cond)                                                   # Split into EnKF_N and rest

  # PRESENTING RESULTS
  # Res.print_frame(Res.mean_field('rmse_a')[0].tolist())                       # re-use print_frame to print mean_field
  # Res.print_mean_field('rmse_a',show_fail=True,show_conf=False,col_inds=...)  # print_mean_field has more options
  # #Res.print_field(Res.field('rmse_a'))                                       # print raw data
  # Res.plot_mean_field('rmse_a')                                               # plot
  # check = toggle_lines()                                                      # check boxes

  Also see AdInf/present_results.py for further examples.
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

    DIR,regex = os.path.split(pattern)
    keys      = sorted_human(os.listdir(DIR))
    keys      = [os.path.join(DIR,f) for f in keys if re.search(regex,f)]
    for f in keys:
      if f in self.datasets:
        print("Warning: re-loading",f)
      if 0==os.path.getsize(f):
        print("Skipping empty file",f)
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
    """
    abscissa = []
    labels   = []
    for ds in self.datasets.values():
      abscissa += [ds['abscissa']]
      labels   += [ds['labels']]
    # Make labels and abscissa unique
    def retain_order_uniq(ar):
      _, inds = np.unique(ar,return_index=True)
      return ar[np.sort(inds)]
    abscissa = np.sort(np.unique(ccat(*abscissa)))
    labels   = retain_order_uniq(ccat(*labels))
    self.abscissa = abscissa
    self.labels   = labels

    # Init
    TABLE  = np.empty(self.shape,object)
    for i,j in np.ndindex(TABLE.shape):
      TABLE[i,j] = []
    # Fill
    fields = set()
    for ds in self.datasets.values():
      for iC,C in enumerate(ds['labels']):
        for iS,S in enumerate(ds['abscissa']):
          avrgs = ds['avrgs'][iS,:,iC].tolist()
          TABLE[labels==C,abscissa==S][0] += avrgs
          fields |= set().union(*(a.keys() for a in avrgs))
    self.TABLE  = TABLE
    self.fields = fields

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

  def split(self,cond):
    C1 = deepcopy(self); C1.rm(cond,INV=True)
    C2 = deepcopy(self); C2.rm(cond)
    return C1, C2

  def mv(self,regex,sub,inds=None):
    """
    Rename configs. Examples:
    Remove space between "tag XX": Res.mv(r'tag (\d+)',r'tag\1')                                          # change "tag 50" to "tag50" => merge such configsregex: search pattern

    sub:   substitution pattern
    inds:  restrict to these inds of table's labels
    """
    if isinstance(inds,int): inds = [inds]
    for ds in self.datasets.values():
      for i,cfg in enumerate(ds['labels']):
        if inds is None or cfg in self.labels[inds]:
          ds['labels'][i] = re.sub(regex, sub, cfg)
    self.regen_table()


  def __deepcopy__(self, memo):
    """
    Implement __deepcopy__ to make it faster.

    We only need to copy the datasets.
    Then regen_table essentially re-inits the object.

    The speed-up is obtained by stopping the 'deep' copying
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
      new.datasets[k] = {
          'abscissa':deepcopy(ds['abscissa']),
          'labels'  :deepcopy(ds['labels']),
          'avrgs'   :np.empty(ds['avrgs'].shape,dict)
          }
      for idx, avrg in np.ndenumerate(ds['avrgs']):
        # 'shallow' copy of the entry
        new.datasets[k]['avrgs'][idx] = copy(avrg)

    new.regen_table()
    return new

  def __repr__(self):
    s = "datasets from " + str(self.patterns)
    if hasattr(self,'abscissa'):
      s +="\nabscissa: "      +str(self.abscissa)+\
          "\nfields: "        +str(self.fields)+\
          "\n\n"+\
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
    # Extract field
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
      conf[iC,iS] = f.std(ddof=1)/sqrt(len(f)) if len(f)>5 else np.nan
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


  def print_mean_field(self,field,show_conf=False,show_fail=False,col_inds=None):
    """
    Print mean frame, including nRep (#) for each value.
    Don't print nan's when nRep==0 (i.e. print nothing).
    show_conf: include confidence estimate (±).
    show_fail: include number of runs that yielded NaNs (#).
               if False but NaNs are present: print NaN for the mean value.
    jj: indices of columns (experiment abscissa) to include. Default: all
    """

    mu, conf, nSuc = self.mean_field(field)
    nReps = self.nRepeats
    nFail = nReps-nSuc

    # Num of figures required to write nRepeats.max()
    NF = str(int(floor(log10(nReps.max()))+1))

    # Set mean values to NaN wherever NaNs are present
    if not show_fail: mu[nFail.astype(bool)] = np.nan

    if col_inds is None: col_inds = arange(len(self.abscissa))

    headr = ['name']
    mattr = [self.labels.tolist()]

    # Fill in stats
    for iS in col_inds:
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


  def plot_mean_field(self,field):
    mu = self.mean_field(field)[0]
    for iC,(row,name) in enumerate(zip(mu,self.labels)): 
      plt.plot(self.abscissa,row,'-o',label=name)





