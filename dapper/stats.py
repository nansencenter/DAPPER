"""Provide the stats class which defines the "builtin" stats to be computed."""

from dapper import *

class Stats(NestedPrint):
  """Contains and computes statistics of the DA methods.

  Use new_series() to register your own stat time series.
  """

  printopts = {'precision' : 3, 'ordr_by_linenum' : -1}
 
  def __init__(self,config,HMM,xx,yy):
    """Init the default statistics.

    Note: Python allows dynamically creating attributes, so you can easily
    add custom stat. series to a Stat instance within a particular method,
    for example. Use ``new_series`` to get automatic averaging too.
    """

    ######################################
    # Preamble
    ######################################
    self.config  = config
    self.xx      = xx
    self.yy      = yy
    self.HMM     = HMM

    # Shapes
    K    = xx.shape[0]-1
    Nx   = xx.shape[1]
    KObs = yy.shape[0]-1
    Ny   = yy.shape[1]
    self.K   , self.Nx = K   , Nx
    self.KObs, self.Ny = KObs, Ny

    # Methods for summarizing multivariate stats ("fields") as scalars
    self.field_summaries = dict(
            # Don't use nanmean here; nan's should get propagated!
            # suffix              formula
            m         = lambda x: mean(x)           , # mean-field
            rms       = lambda x: sqrt(mean(x**2))  , # root-mean-square
            ma        = lambda x: mean(abs(x))      , # mean-absolute
            gm        = lambda x: exp(mean(log(x))) , # geometric mean
            )
    # Only keep the methods listed in rc
    self.field_summaries = {k:v for k,v in self.field_summaries.items()
            if k in rc['stat']['field_summary_methods'].split(',')}

    # Define similar methods, but restricted to sectors
    self.sector_summaries = {}
    restrict = lambda fun, inds: (lambda x: fun(x[inds]))
    for suffix, formula in self.field_summaries.items():
        for sector, inds in HMM.sectors.items():
            f = restrict(formula,inds)
            self.sector_summaries['%s.%s'%(suffix,sector)] = f


    ######################################
    # Allocate time series of various stats
    ######################################
    self.new_series('mu'    ,Nx, MS='sec') # Mean
    self.new_series('std'   ,Nx, MS='sec') # Std. dev. ("spread")
    self.new_series('err'   ,Nx, MS='sec') # Error (mu - truth)
    self.new_series('gscore',Nx, MS='sec') # Gaussian (log) score

    # To save memory, we only store these field means:
    self.new_series('mad' ,1) # Mean abs deviations
    self.new_series('skew',1) # Skewness
    self.new_series('kurt',1) # Kurtosis

    if hasattr(config,'N'):
      N            = config.N
      self.new_series('w',N, MS=True)    # Importance weights
      self.new_series('rh',Nx,dtype=int) # Rank histogram

      self._is_ens = True
      minN         = min(Nx,N)
      do_spectral  = sqrt(Nx*N) <= rc['comp_threshold_b']
    else:
      self._is_ens = False
      minN         = Nx
      do_spectral  = Nx <= rc['comp_threshold_b']

    if do_spectral:
      # Note: the mean-field and RMS time-series of
      # (i) svals and (ii) umisf should match the corresponding series of
      # (i) std and (ii) err.
      self.new_series('svals',minN) # Principal component (SVD) scores
      self.new_series('umisf',minN) # Error in component directions


    ######################################
    # Allocate a few series for outside use
    ######################################
    self.new_series('trHK'  , 1, KObs+1)
    self.new_series('infl'  , 1, KObs+1)
    self.new_series('iters' , 1, KObs+1)

    # Weight-related
    self.new_series('N_eff' , 1, KObs+1)
    self.new_series('wroot' , 1, KObs+1)
    self.new_series('resmpl', 1, KObs+1)


  def new_series(self,name,shape,length='FAUSt',MS=False,**kwargs):
      """Create (and register) a statistics time series.

      Series are initialized with nan's.

      Example: Create ndarray of length KObs+1 for inflation time series:
      >>> self.new_series('infl', 1, KObs+1)

      NB: The ``sliding_diagnostics`` liveplotting relies on detecting ``nan``'s
          to avoid plotting stats that are not being used.
          => Cannot use ``dtype=bool`` or ``int`` for stats that get plotted.
      """

      # Convert int shape to tuple
      if not hasattr(shape, '__len__'):
        if shape==1: shape = ()
        else:        shape = (shape,)

      def make_series(shape,**kwargs):
          if length=='FAUSt':
              return FAUSt(self.K, self.KObs, shape,
                      self.config.store_u, self.config.store_s, **kwargs)
          else:
              return DataSeries((length,)+shape,**kwargs)

      # Principal series
      series = make_series(shape)
      setattr(self, name, series)

      # Summary (scalar) series:
      if shape!=():
          if MS:
              for suffix in self.field_summaries:
                  setattr(series,suffix,make_series(()))
          # Make a nested level for sectors
          if MS=='sec':
              for ss in self.sector_summaries:
                  suffix, sector = ss.split('.')
                  sector = setattr(getattr(series,suffix),sector,make_series(()))


  @property
  def data_series(self):
      return [k for k in vars(self) if isinstance(getattr(self,k),DataSeries)]

  @property
  def included(self):
      return self.data_series



  def assess(self,k,kObs=None,faus=None,
      E=None,w=None,mu=None,Cov=None):
    """Common interface for both assess_ens and _ext.

    The _ens assessment function gets called if E is not None,
    and _ext if mu is not None.

    faus: One or more of ['f',' a', 'u'], indicating
          that the result should be stored in (respectively)
          the forecast/analysis/universal attribute.
          Default: 'u' if kObs is None else 'au' ('a' and 'u').
    """

    # Initial consistency checks.
    if k==0:
      if kObs is not None:
        raise KeyError("DAPPER convention: no obs at t=0. Helps avoid bugs.")
      if faus is None:
        faus = 'u'
      if self._is_ens==True:
        def rze(a,b,c):
          raise TypeError("Expected "+a+" input, but "+b+" is "+c+" None")
        if E is None:      rze("ensemble","E","")
        if mu is not None: rze("ensemble","my/Cov","not")
      else:
        if E is not None:  rze("mu/Cov","E","not")
        if mu is None:     rze("mu/Cov","mu","")

    # Default. Don't add more defaults. It just gets confusing.
    if faus is None:
      faus = 'u' if kObs is None else 'au'

    # Select assessment call and arguments
    if self._is_ens:
      _assess = self.assess_ens
      _prms   = {'E':E,'w':w}
    else:
      _assess = self.assess_ext
      _prms   = {'mu':mu,'P':Cov}

    for sub in faus:
        # Skip assessment?
        if kObs==None and not self.config.store_u:
          try:
            if (not rc['liveplotting_enabled']) or (not self.LP_instance.any_figs):
              continue
          except AttributeError:
            pass # LP_instance not yet created

        # Avoid repetitive warnings caused by zero variance
        with np.errstate(divide='call',invalid='call'):
            np.seterrcall(warn_zero_variance)

            # Call assessment
            stats_now = Bunch()
            _assess(stats_now, self.xx[k], **_prms)
            self.derivative_stats(stats_now)
            self.summarize_marginals(stats_now)

        # Write current stats to series
        for name,val in stats_now.items():
            stat = deep_getattr(self,name)
            if isinstance(stat,FAUSt): stat[(k,kObs,sub)] = val
            else:                      stat[kObs]         = val

        # LivePlot -- Both initiation and update must come after the assessment.
        if rc['liveplotting_enabled']:
          if not hasattr(self,'LP_instance'): # -- INIT --
            self.LP_instance = LivePlot(self, self.config.liveplots, (k,kObs,sub), E, Cov)
          else: # -- UPDATE --
            self.LP_instance.update((k,kObs,sub),E,Cov)


  def summarize_marginals(self,now):
    "Compute Mean-field and RMS values"
    formulae = {**self.field_summaries, **self.sector_summaries}

    with np.errstate(divide='ignore',invalid='ignore'):
        for stat in list(now):
            field = now[stat]
            for suffix, formula in formulae.items():
                statpath = stat+'.'+suffix
                if deep_hasattr(self, statpath):
                    now[statpath] = formula(field)


  def derivative_stats(self,now):
    """Stats that derive from others (=> not specific for _ens or _ext)."""
    now.gscore = 2*log(now.std) + (now.err/now.std)**2


  def assess_ens(self,now,x,E,w):
    """Ensemble and Particle filter (weighted/importance) assessment."""
    N,Nx = E.shape

    if w is None: 
      w = ones(N)/N # All equal. Also, rm attr from stats:
      if hasattr(self,'w'):
        delattr(self,'w')
    else:
      now.w = w
      if abs(w.sum()-1) > 1e-5:    raise_AFE("Weights did not sum to one.")
    if not np.all(np.isfinite(E)): raise_AFE("Ensemble not finite.")
    if not np.all(np.isreal(E)):   raise_AFE("Ensemble not Real.")

    now.mu  = w @ E
    now.err = now.mu - x
    A = E - now.mu

    # While A**2 is approx as fast as A*A,
    # A**3 is 10x slower than A**2 (or A**2.0).
    # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
    # But, to save memory, only use A_pow.
    A_pow = A**2

    # Compute variances
    var  = w @ A_pow
    ub   = unbias_var(w,avoid_pathological=True)
    var *= ub

    # Compute standard deviation ("Spread")
    std = sqrt(var) # NB: biased (even though var is unbiased)
    now.std = std
    
    # For simplicity, use naive (biased) formulae, derived
    # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
    # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
    A_pow *= A
    now.skew = np.nanmean( w @ A_pow / (std*std*std) )
    A_pow *= A
    now.kurt = np.nanmean( w @ A_pow / var**2 - 3 )

    now.mad  = np.nanmean( w @ abs(A) )

    if hasattr(self,'svals'):
      if N<=Nx:
        _,s,UT    = svd( (sqrt(w)*A.T).T, full_matrices=False)
        s        *= sqrt(ub) # Makes s^2 unbiased
        now.svals = s
        now.umisf = UT @ now.err
      else:
        P         = (A.T * w) @ A
        s2,U      = eigh(P)
        s2       *= ub
        now.svals = sqrt(s2.clip(0))[::-1]
        now.umisf = U.T[::-1] @ now.err

      # For each state dim [i], compute rank of truth (x) among the ensemble (E)
      E_x = np.sort(np.vstack((E,x)),axis=0,kind='heapsort')
      now.rh = np.asarray([np.where(E_x[:,i]==x[i])[0][0] for i in range(Nx)])


  def assess_ext(self,now,x,mu,P):
    """Kalman filter (Gaussian) assessment."""
    Nx = len(mu)

    if not np.all(np.isfinite(mu)): raise_AFE("Estimates not finite.")
    if not np.all(np.isreal(mu)):   raise_AFE("Estimates not Real.")
    # Don't check the cov (might not be explicitly availble)

    now.mu  = mu
    now.err = now.mu - x

    var = P.diag if isinstance(P,CovMat) else diag(P)
    now.std = sqrt(var)

    # Here, sqrt(2/pi) is the ratio, of MAD/STD for Gaussians
    now.mad = np.nanmean( now.std ) * sqrt(2/pi)

    if hasattr(self,'svals'):
      P         = P.full if isinstance(P,CovMat) else P
      s2,U      = nla.eigh(P)
      now.svals = sqrt(np.maximum(s2,0.0))[::-1]
      now.umisf = (U.T @ now.err)[::-1]


  def average_in_time(self,kk=None,kkObs=None):
      """Avarage all univariate (scalar) time series.

      - ``kk``    time inds for averaging
      - ``kkObs`` time inds for averaging obs
      """
      chrono = self.HMM.t
      if kk    is None: kk     = chrono.mask_BI
      if kkObs is None: kkObs  = chrono.maskObs_BI

      def average(series):
          avrgs = Bunch(printopts=FAUSt.printopts)

          def average_multivariate(): return avrgs
          # Plain averages of nd-series are rarely interesting.
          # => Leave for manual computations

          if isinstance(series,FAUSt):
              # Average series for each subscript
              if series.item_shape != ():
                  return average_multivariate()
              for sub in [ch for ch in 'fas' if hasattr(series,ch)]:
                  avrgs[sub] = series_mean_with_conf(series[kkObs,sub])
              if series.store_u:
                  avrgs['u'] = series_mean_with_conf(series[kk,'u'])

          elif isinstance(series,DataSeries):
              if series.array.shape[1:] != ():
                  return average_multivariate()
              elif len(series.array) == self.KObs+1:
                  avrgs = series_mean_with_conf(series[kkObs])
              elif len(series.array) == self.K+1:
                  avrgs = series_mean_with_conf(series[kk])
              else: raise ValueError

          return avrgs

      def recurse_average(stat_parent,avrgs_parent):
          for key,series in vars(stat_parent).items(): # Loop data_series
              if not isinstance(series,DataSeries): continue

              avrgs = average(series)
              avrgs_parent[key] = avrgs

              recurse_average(series,avrgs)

      avrgs = Bunch(printopts=FAUSt.printopts)
      recurse_average(self,avrgs)
      return avrgs

@do_once
def warn_zero_variance(err,flag):
    """In case of degeneracy, variance might be 0,
    causing warnings in computing skew/kurt/MGLS
    (which all normalize by variance).
    This should and will yield nan's, but we don't want
    mere diagnostics computations to cause repetitive warnings,
    so we only warn once."""
    msg = "Numerical error in stat comps.\n"+\
            "Probably caused by a sample variance of 0."
    warnings.warn(msg)


# TODO: do something about key
def raise_AFE(msg,key=None):
  if key is not None:
    msg += "\n(k,kObs,fau) = " + str(key) + ". "
  raise AssimFailedError(msg)


def tabulate_avrgs(avrgs_list,statkeys=(),decimals=None,pad=' '):
    """Tabulate avrgs (val±conf).

    - ``statkeys``: list of keys of statistics to include.
    """
    # Defaults averages
    if not statkeys:
      statkeys = ['rmse.a','rmv.a','rmse.f']
  
    # Abbreviations
    abbrevs = {'rmse':'err.rms', 'rmss':'std.rms', 'rmv':'std.rms'}
    de_abbrev = lambda k: '.'.join(abbrevs.get(l,l) for l in k.split('.'))
  
    def tabulate_column(col,header):
        """Align single column (on decimal pt) using tabulate().
        Pad for equal length."""
        col = tabulate_orig.tabulate(col,[header],'plain').splitlines()
        mxW = max(len(s) for s in col)
        col = [s + ' '*(mxW-len(s)) for s in col]
        col = [s.replace(' ',pad)   for s in col]
        return col
  
    # Fill in
    headr, mattr = [], []
    for column in statkeys:
        # Get vals, confs
        vals, confs = [], []
        for avrgs in avrgs_list:
            uq = deep_getattr(avrgs,de_abbrev(column),None)
            if uq is None:         val,conf = None,None
            elif decimals is None: val,conf = uq.round(mult=0.2)
            else:                  val,conf = np.round([uq.val, uq.conf],decimals)
            vals .append([val])
            confs.append([conf])
        # Align
        vals  = tabulate_column(vals , column)
        confs = tabulate_column(confs, '1σ')
        # Enter in headr, mattr
        headr.append(vals[0]+'  1σ')
        mattr.append([v +' ±'+c for v,c in zip(vals,confs)][1:])
    return headr, mattr

