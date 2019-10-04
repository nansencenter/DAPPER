"""Provide the stats class which defines the "builtin" stats to be computed."""

from dapper import *

class Stats(NestedPrint):
  """Contains and computes statistics of the DA methods.

  Use new_series() to register your own stat time series.
  """

  # Used by NestedPrint
  precision = 3
  ordr_by_linenum = -1
 
  def __init__(self,config,HMM,xx,yy):
    """Init the default statistics.

    Note: you may also allocate & compute individual stats elsewhere
          (Python allows dynamic class attributes).
          For example at the top of your experimental DA method,
          which avoids "polluting" this space.
    """

    ######################################
    # Save twin experiment settings 
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

    ######################################
    # Declare time series of various stats
    ######################################
    self.data_register = [] # Register of names of series
    self.included = self.data_register # Repr: only these

    self.new_series('mu' ,Nx) # Mean
    self.new_series('var',Nx) # Variances
    self.new_series('std',Nx) # Spread
    self.new_series('mad',Nx) # Mean abs deviations
    self.new_series('err',Nx) # Error (mu - truth)

    self.new_series('logp_m',1)  # Marginal ,Gaussian Log score
    self.new_series('skew'  ,1)  # Skewness
    self.new_series('kurt'  ,1)  # Kurtosis
    self.new_series('rmv'   ,1)  # Root-mean variance
    self.new_series('rmse'  ,1)  # Root-mean square error

    if hasattr(config,'N'):
      # Ensemble-only init
      self._is_ens = True
      N            = config.N
      minN         = min(Nx,N)
      self.new_series('w',N)             # Importance weights
      self.new_series('rh',Nx,dtype=int) # Rank histogram
      do_spectral  = sqrt(Nx*N) <= rc['comp_threshold_b']
    else:
      # Linear-Gaussian assessment
      self._is_ens = False
      minN         = Nx
      do_spectral  = Nx <= rc['comp_threshold_b']

    if do_spectral:
      self.new_series('svals',minN) # Principal component (SVD) scores
      self.new_series('umisf',minN) # Error in component directions


    ######################################
    # Declare a few series for outside use
    ######################################
    self.new_series('trHK'  , (), KObs+1)
    self.new_series('infl'  , (), KObs+1)
    self.new_series('iters' , (), KObs+1)

    # Weight-related
    self.new_series('N_eff' , (), KObs+1)
    self.new_series('wroot' , (), KObs+1)
    self.new_series('resmpl', (), KObs+1)


  def new_series(self,name,shape,length='FAUSt',**kwargs):
      """Create (and register) a statistics time series.

      Example: Create ndarray of length KObs+1 for inflation time series:
      >>> self.new_series('infl', (), KObs+1)
      """

      if length=='FAUSt': series = FAUSt(self.K,self.KObs,shape,self.config.store_u, **kwargs)
      else              : series = np.full((length,)+shape, nan, **kwargs)
      setattr(self, name, series)     # Write
      self.data_register.append(name) # Register

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
            _assess(stats_now,self.xx[k],**_prms)

        # Write current stats to series
        for stat,val in stats_now.items():
            getattr(self,stat)[(k,kObs,sub)] = val

        # LivePlot -- Both initiation and update must come after the assessment.
        if rc['liveplotting_enabled']:
          if not hasattr(self,'LP_instance'): # -- INIT --
            self.LP_instance = LivePlot(self, self.config.liveplots, (k,kObs,sub), E, Cov)
          else: # -- UPDATE --
            self.LP_instance.update((k,kObs,sub),E,Cov)


  def assess_ens(self,now,x,E,w):
    """Ensemble and Particle filter (weighted/importance) assessment."""
    N,Nx = E.shape

    if w is None: 
      w = ones(N)/N # No weights. Also, rm attr from stats:
      if hasattr(self,'w'):
        delattr(self,'w')
        self.data_register.remove('w')
    else:
      now.w = w
      if abs(w.sum()-1) > 1e-5:    raise_AFE("Weights did not sum to one.")
    if not np.all(np.isfinite(E)): raise_AFE("Ensemble not finite.")
    if not np.all(np.isreal(E)):   raise_AFE("Ensemble not Real.")

    now.mu = w @ E
    A = E - now.mu

    # While A**2 is approx as fast as A*A,
    # A**3 is 10x slower than A**2 (or A**2.0).
    # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
    # But, to save memory, only use A_pow.
    A_pow = A**2

    now.var = w @ A_pow
    now.mad = w @ abs(A) # Mean abs deviations

    ub       = unbias_var(w,avoid_pathological=True)
    now.var *= ub
    
    # For simplicity, use naive (biased) formulae, derived
    # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
    # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
    A_pow *= A
    now.skew = np.nanmean( w @ A_pow / now.var**(3/2) )
    A_pow *= A
    now.kurt = np.nanmean( w @ A_pow / now.var**2 - 3 )

    self.derivative_stats(now,x)

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
      Ex_sorted = np.sort(np.vstack((E,x)),axis=0,kind='heapsort')
      now.rh    = [np.where(Ex_sorted[:,i] == x[i])[0][0] for i in range(Nx)]


  def assess_ext(self,now,x,mu,P):
    """Kalman filter (Gaussian) assessment."""
    Nx = len(mu)

    if not np.all(np.isfinite(mu)): raise_AFE("Estimates not finite.")
    if not np.all(np.isreal(mu)):   raise_AFE("Estimates not Real.")
    # Don't check the cov (might not be explicitly availble)

    now.mu  = mu
    now.var = P.diag if isinstance(P,CovMat) else diag(P)
    now.mad = sqrt(now.var)*sqrt(2/pi) # sqrt(2/pi): ratio, Gaussian MAD/STD

    self.derivative_stats(now,x)

    if hasattr(self,'svals'):
      P         = P.full if isinstance(P,CovMat) else P
      s2,U      = nla.eigh(P)
      now.svals = sqrt(np.maximum(s2,0.0))[::-1]
      now.umisf = (U.T @ now.err)[::-1]


  def derivative_stats(self,now,x):
    """Stats that apply derive from the others, and apply for both _ens and _ext"""
    now.err  = now.mu - x

    now.rmv  = sqrt(mean(now.var))
    now.rmse = sqrt(mean(now.err**2))
    self.MGLS(now)
    
  def MGLS(self,now):
    "Marginal Gaussian Log Score."
    Nx         = len(now.err)
    ldet       = log(now.var).sum()
    nmisf      = now.var**(-1/2) * now.err
    logp_m     = (nmisf**2).sum() + ldet
    now.logp_m = logp_m/Nx


  def average_in_time(self,kk=None,kkObs=None):
      """Avarage all univariate (scalar) time series.

      - ``kk``    time inds for averaging
      - ``kkObs`` time inds for averaging obs
      """
      avrg = AlignedDict()

      chrono = self.HMM.t
      if kk    is None: kk     = chrono.mask_BI
      if kkObs is None: kkObs  = chrono.maskObs_BI

      def average_multivariate():
        raise NotImplementedError(
        """Plain averages of nd-series are rarely interesting.
        => Leave for manual computations.""")

      for key in self.data_register:

          if hasattr(self,key):
              series = getattr(self,key)
          else:
              continue

          try:
              if isinstance(series,FAUSt):
                  # Average series for each subscript
                  if series.item_shape != ():
                      average_multivariate()
                  for sub in 'afs':
                      avrg[key+'_'+sub] = series_mean_with_conf(series[kkObs,sub])
                  if series.store_u:
                      avrg[key+'_u'] = series_mean_with_conf(series[kk,'u'])

              elif isinstance(series,np.ndarray):
                  # Average the array
                  if series.ndim > 1:
                      avrg[key] = average_multivariate()
                  elif len(series) == self.KObs+1:
                      avrg[key] = series_mean_with_conf(series[kkObs])
                  elif len(series) == self.K+1:
                      avrg[key] = series_mean_with_conf(series[kk])
                  else:
                      raise ValueError
              
              else:
                  raise NotImplementedError

          except NotImplementedError:
            pass

      return avrg

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


