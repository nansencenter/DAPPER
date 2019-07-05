"""Provide the stats class which defines the "builtin" stats to be computed."""

from dapper import *

class Stats(NestedPrint):
  """
  Contains and computes statistics of the DA methods.
  """

  # Used by NestedPrint
  excluded  = NestedPrint.excluded +\
      ['HMM','config','xx','yy','style1','style2','LP_instance']
  precision = 3
  ordr_by_linenum = -1
 
  def __init__(self,config,HMM,xx,yy):
    """
    Init the default statistics.

    Note: you may also allocate & compute individual stats elsewhere
          (Python allows dynamic class attributes).
          For example at the top of your experimental DA method,
          which avoids "polluting" this space.
    """

    ######################################
    # Save twin experiment settings 
    ######################################
    self.config = config
    self.HMM    = HMM
    self.xx     = xx
    self.yy     = yy

    # Validations
    Nx   = HMM.Nx      ; assert Nx   ==xx.shape[1]
    Ny   = HMM.Ny      ; assert Ny   ==yy.shape[1]
    K    = HMM.t.K     ; assert K    ==xx.shape[0]-1
    KObs = HMM.t.KObs  ; assert KObs ==yy.shape[0]-1


    ######################################
    # Declare time series of various stats
    ######################################
    new_series = self.new_FAU_series

    self.mu     = new_series(Nx) # Mean
    self.var    = new_series(Nx) # Variances
    self.mad    = new_series(Nx) # Mean abs deviations
    self.err    = new_series(Nx) # Error (mu-truth)
    self.logp_m = new_series(1)  # Marginal, Gaussian Log score
    self.skew   = new_series(1)  # Skewness
    self.kurt   = new_series(1)  # Kurtosis
    self.rmv    = new_series(1)  # Root-mean variance
    self.rmse   = new_series(1)  # Root-mean square error

    if hasattr(config,'N'):
      # Ensemble-only init
      self._had_0v = False
      self._is_ens = True
      N            = config.N
      minN         = min(Nx,N)
      self.w       = new_series(N)            # Importance weights
      self.rh      = new_series(Nx,dtype=int) # Rank histogram
      do_spectral  = sqrt(Nx*N) <= rc['comp_threshold_b']
    else:
      # Linear-Gaussian assessment
      self._is_ens = False
      minN         = Nx
      do_spectral  = Nx <= rc['comp_threshold_b']

    if do_spectral:
      self.svals = new_series(minN) # Principal component (SVD) scores
      self.umisf = new_series(minN) # Error in component directions


    ######################################
    # Declare non-FAU (i.e. normal) series
    ######################################
    self.trHK   = np.full (KObs+1, nan)
    self.infl   = np.full (KObs+1, nan)
    self.iters  = np.full (KObs+1, nan)

    # Weight-related
    self.N_eff  = np.full (KObs+1, nan)
    self.wroot  = np.full (KObs+1, nan)
    self.resmpl = np.full (KObs+1, nan)


    ######################################
    # Define which stats get plotted as diagnostics in liveplotting, and how.
    ######################################
    # NB: The diagnostic liveplotting relies on detecting nan's to avoid
    #     plotting stats that are not being used.
    #     => Cannot use dtype bool or int for those that may be plotted.

    def lin(a,b): return lambda x: a + b*x
    def Id(x)   : return x
    def divN(x) :
      try: return x/N
      except NameError: return nan

    # RMS
    self.style1 = {
        'rmse'    : [Id          , None   , dict(c='k'      , label='Error'            )],
        'rmv'     : [Id          , None   , dict(c='b'      , label='Spread', alpha=0.6)],
      }

    # OTHER         transf       , shape  , plt kwargs
    self.style2 = OrderedDict([
        ('skew'   , [Id          , None   , dict(c=     'g' , label=star+'Skew/$\sigma^3$'        )]),
        ('kurt'   , [Id          , None   , dict(c=     'r' , label=star+'Kurt$/\sigma^4{-}3$'    )]),
        ('trHK'   , [Id          , None   , dict(c=     'k' , label=star+'HK'                     )]),
        ('infl'   , [lin(-10,10) , 'step' , dict(c=     'c' , label='10(infl-1)'                  )]),
        ('N_eff'  , [divN        , 'dirac', dict(c=RGBs['y'], label='N_eff/N'             ,lw=3   )]),
        ('iters'  , [lin(0,.1)   , 'dirac', dict(c=     'm' , label='iters/10'                    )]),
        ('resmpl' , [Id          , 'dirac', dict(c=     'k' , label='resampled?'                  )]),
      ])


  def assess(self,k,kObs=None,f_a_u=None,
      E=None,w=None,mu=None,Cov=None):
    """
    Common interface for both assess_ens and _ext.

    f_a_u: One or more of ['f',' a', 'u'], indicating
           that the result should be stored in (respectively)
           the forecast/analysis/universal attribute.
           Default: 'u' if kObs is None else 'au' ('a' and 'u').
    """

    # Initial consistency checks.
    if k==0:
      if kObs is not None:
        raise KeyError("DAPPER convention: no obs at t=0. Helps avoid bugs.")
      if f_a_u is None:
        f_a_u = 'u'
      if self._is_ens==True:
        def rze(a,b,c):
          raise TypeError("Expected "+a+" input, but "+b+" is "+c+" None")
        if E is None:      rze("ensemble","E","")
        if mu is not None: rze("ensemble","my/Cov","not")
      else:
        if E is not None:  rze("mu/Cov","E","not")
        if mu is None:     rze("mu/Cov","mu","")

    # Default. Don't add more defaults. It just gets confusing.
    if f_a_u is None:
      f_a_u = 'u' if kObs is None else 'au'

    # Prepare assessment call and arguments
    if self._is_ens:
      # Ensemble assessment
      alias = self.assess_ens
      state_prms = {'E':E,'w':w}
    else:
      # Moment assessment
      alias = self.assess_ext
      state_prms = {'mu':mu,'P':Cov}

    for fau in f_a_u:
      # Assemble key
      key = (k,kObs,fau)

      # Skip assessment?
      if kObs==None and not self.config.store_u:
        try:
          if (not rc['liveplotting_enabled']) or (not self.LP_instance.any_figs):
            continue
        except AttributeError:
          pass # LP_instance not yet created

      # Call assessment
      with np.errstate(divide='ignore',invalid='ignore'):
        alias(key,**state_prms)

      # In case of degeneracy, variance might be 0,
      # causing warnings in computing skew/kurt/MGLS
      # (which all normalize by variance).
      # This should and will yield nan's, but we don't want
      # the diagnostics computations to cause too many warnings,
      # so we turned them off above. But we'll manually warn ONCE here.
      if not getattr(self,'_had_0v',False) \
          and np.allclose(sqrt(self.var[key]),0):
        self._had_0v = True
        warnings.warn("Sample variance was 0 at (k,kObs,fau) = " + str(key))

      # LivePlot -- Both initiation and update must come after the assessment.
      if rc['liveplotting_enabled']:
        if not hasattr(self,'LP_instance'): # -- INIT --
          self.LP_instance = LivePlot(self, self.config.liveplotting, key,E,Cov)
        else: # -- UPDATE --
          self.LP_instance.update(key,E,Cov)


  def assess_ens(self,k,E,w=None):
    """Ensemble and Particle filter (weighted/importance) assessment."""
    # Unpack
    N,Nx = E.shape
    x = self.xx[k[0]]

    # Validate weights
    if w is None: 
      try:                    delattr(self,'w')
      except AttributeError:  pass
      finally:                w = 1/N
    if np.isscalar(w):
      assert w != 0
      w = w*ones(N)
    if hasattr(self,'w'):
      self.w[k] = w

    if abs(w.sum()-1) > 1e-5:      raise_AFE("Weights did not sum to one.",k)
    if not np.all(np.isfinite(E)): raise_AFE("Ensemble not finite.",k)
    if not np.all(np.isreal(E)):   raise_AFE("Ensemble not Real.",k)

    self.mu[k]   = w @ E
    A            = E - self.mu[k]

    # While A**2 is approx as fast as A*A,
    # A**3 is 10x slower than A**2 (or A**2.0).
    # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
    # But, to save memory, only use A_pow.
    A_pow        = A**2

    self.var[k]  = w @ A_pow
    self.mad[k]  = w @ abs(A)  # Mean abs deviations

    ub           = unbias_var(w,avoid_pathological=True)
    self.var[k] *= ub
    

    # For simplicity, use naive (biased) formulae, derived
    # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
    # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
    A_pow       *= A
    self.skew[k] = np.nanmean( w @ A_pow / self.var[k]**(3/2) )
    A_pow       *= A # idem.
    self.kurt[k] = np.nanmean( w @ A_pow / self.var[k]**2 - 3 )

    self.derivative_stats(k,x)

    if hasattr(self,'svals'):
      if N<=Nx:
        _,s,UT         = svd( (sqrt(w)*A.T).T, full_matrices=False)
        s             *= sqrt(ub) # Makes s^2 unbiased
        self.svals[k]  = s
        self.umisf[k]  = UT @ self.err[k]
      else:
        P              = (A.T * w) @ A
        s2,U           = eigh(P)
        s2            *= ub
        self.svals[k]  = sqrt(s2.clip(0))[::-1]
        self.umisf[k]  = U.T[::-1] @ self.err[k]

      # For each state dim [i], compute rank of truth (x) among the ensemble (E)
      Ex_sorted     = np.sort(np.vstack((E,x)),axis=0,kind='heapsort')
      self.rh[k]    = [np.where(Ex_sorted[:,i] == x[i])[0][0] for i in range(Nx)]


  def assess_ext(self,k,mu,P):
    """Kalman filter (Gaussian) assessment."""

    isFinite = np.all(np.isfinite(mu)) # Do not check covariance
    isReal   = np.all(np.isreal(mu))   # (coz might not be explicitly availble)
    if not isFinite: raise_AFE("Estimates not finite.",k)
    if not isReal:   raise_AFE("Estimates not Real.",k)

    Nx = len(mu)
    x  = self.xx[k[0]]

    self.mu[k]  = mu
    self.var[k] = P.diag if isinstance(P,CovMat) else diag(P)
    self.mad[k] = sqrt(self.var[k])*sqrt(2/pi)
    # ... because sqrt(2/pi) = ratio MAD/STD for Gaussians

    self.derivative_stats(k,x)

    if hasattr(self,'svals'):
      P             = P.full if isinstance(P,CovMat) else P
      s2,U          = nla.eigh(P)
      self.svals[k] = sqrt(np.maximum(s2,0.0))[::-1]
      self.umisf[k] = (U.T @ self.err[k])[::-1]


  def derivative_stats(self,k,x):
    """Stats that apply for both _w and _ext paradigms and derive from the other stats."""
    self.err[k]  = self.mu[k] - x
    self.rmv[k]  = sqrt(mean(self.var[k]))
    self.rmse[k] = sqrt(mean(self.err[k]**2))
    self.MGLS(k)
    
  def MGLS(self,k):
    # Marginal Gaussian Log Score.
    Nx             = len(self.err[k])
    ldet           = log(self.var[k]).sum()
    nmisf          = self.var[k]**(-1/2) * self.err[k]
    logp_m         = (nmisf**2).sum() + ldet
    self.logp_m[k] = logp_m/Nx


  def average_in_time(self):
    """
    Avarage all univariate (scalar) time series.
    """
    avrg = AlignedDict()
    for key,series in vars(self).items():
      if key.startswith('_'):
        continue
      try:
        # FAU_series
        if isinstance(series,FAU_series):
          # Compute
          f_a_u = series.average()
          # Add the sub-fields as sub-scripted fields
          for sub in f_a_u: avrg[key+'_'+sub] = f_a_u[sub]
        # Array
        elif isinstance(series,np.ndarray):
          if series.ndim > 1:
            raise NotImplementedError
          t = self.HMM.t
          if len(series) == len(t.kkObs):
            inds = t.maskObs_BI
          elif len(series) == len(t.kk):
            inds = t.kk_BI
          else:
            raise ValueError
          # Compute
          avrg[key] = series_mean_with_conf(series[inds])
        # Scalars
        elif np.isscalar(series):
          avrg[key] = series
        else:
          raise NotImplementedError
      except NotImplementedError:
        pass
    return avrg


  def average_subset(self,ii):
    """
    Produce time-averages from subsets (ii) of the state indices.
    Then average in time.
    This is a mediocre solution, and should be systematized somehow.
    """ 
    avrg = AlignedDict()
    # Compute univariate time series from subset of state variables
    for fa in 'fa':
      avrg['rmse_'+fa] = sqrt(mean(getattr(self.err,fa)[:,ii]**2,1))
      avrg['rmv_' +fa] = sqrt(mean(getattr(self.var,fa)[:,ii]   ,1))
    # Average in time:
    for key,series in avrg.items():
      avrg[key] = series_mean_with_conf(series[self.HMM.t.maskObs_BI])
    return avrg



  def new_FAU_series(self,M,**kwargs):
    "Convenience FAU_series constructor."
    store_u = self.config.store_u
    return FAU_series(self.HMM.t, M, store_u=store_u, **kwargs)



def average_each_field(table,axis=1):
  "Average each field in a 2D table of dicts along a given axis."
  if isinstance(table,list):
    table = array(table)
  if axis == 0:
    table = np.transpose(table)
  assert table.ndim == 2

  M,N = table.shape
  avrg = np.empty(M,dict)

  for i,row in enumerate(table):
    avrg[i] = dict()
    for key in table[i][0].keys():
      avrg[i][key] = val_with_conf(
          val  = mean([s_ij[key].val  for s_ij in row]),
          conf = mean([s_ij[key].conf for s_ij in row])/sqrt(N))
      # NB: This is a rudimentary averaging of confidence intervals
      # Should be checked against variance of avrg[i][key].val
  return avrg



