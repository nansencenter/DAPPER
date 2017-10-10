from common import *

class Stats(MLR_Print):
  """
  Contains and computes statistics of the DA methods.
  """

  # Adjust this to omit heavy computations
  comp_threshold_3 = 51

  # Used by MLR_Print
  excluded  = MLR_Print.excluded + ['setup','config','xx','yy']
  precision = 3
  ordr_by_linenum = -1
 
  def __init__(self,config,setup,xx,yy):
    """
    Init the default statistics.
    Note: you may well allocate & compute individual stats elsewhere,
          and simply assigne them as an attribute to the stats instance.
    """

    self.config = config
    self.setup  = setup
    self.xx     = xx
    self.yy     = yy

    m    = setup.f.m    ; assert m   ==xx.shape[1]
    K    = setup.t.K    ; assert K   ==xx.shape[0]-1
    p    = setup.h.m    ; assert p   ==yy.shape[1]
    KObs = setup.t.KObs ; assert KObs==yy.shape[0]-1

    # time-series constructor alias
    fs = self.new_FAU_series 
    self.mu     = fs(m) # Mean
    self.var    = fs(m) # Variances
    self.mad    = fs(m) # Mean abs deviations
    self.err    = fs(m) # Error (mu-truth)
    self.logp_m = fs(1) # Marginal, Gaussian Log score
    self.skew   = fs(1) # Skewness
    self.kurt   = fs(1) # Kurtosis
    self.rmv    = fs(1) # Root-mean variance
    self.rmse   = fs(1) # Root-mean square error

    if hasattr(config,'N'):
      # Ensemble-only init
      self._had_0v = False
      self._is_ens = True
      N            = config.N
      m_Nm         = min(m,N)
      self.w       = fs(N)           # Importance weights
      self.rh      = fs(m,dtype=int) # Rank histogram
      #self.N      = N               # Use w.shape[1] instead
    else:
      # Linear-Gaussian assessment
      self._is_ens = False
      m_Nm         = m

    self.svals = fs(m_Nm) # Principal component (SVD) scores
    self.umisf = fs(m_Nm) # Error in component directions

    # Other. 
    self.trHK = np.full(KObs+1, nan)
    self.infl = np.full(KObs+1, nan)


  def assess(self,k,kObs=None,f_a_u=None,
      E=None,w=None,mu=None,Cov=None):
    """
    Common interface for both assess_ens and _ext.
    f_a_u: One or more of ['f',' a', 'u'], indicating
           that the result should be stored in (respectively)
           the forecast/analysis/universal attribute.
           Defaults: see source code.
    If 'u' in f_a_u: call/update LivePlot.
    """

    # Initial consistency checks.
    if k==0:
      if kObs is not None:
        raise KeyError("Should not have any obs at initial time."+
            "This very easily leads to bugs, and not 'DA convention'.")
      if self._is_ens==True:
        def rze(a,b,c):
          raise TypeError("Expected "+a+" input, but "+b+" is "+c+" None")
        if E is None:      rze("ensemble","E","")
        if mu is not None: rze("ensemble","my/Cov","not")
      else:
        if E is not None:  rze("mu/Cov","E","not")
        if mu is None:     rze("mu/Cov","mu","")
    

    # Defaults for f_a_u
    if f_a_u is None:
      if kObs is None:
        f_a_u = 'u'
      else:
        f_a_u = 'au'
    elif f_a_u == 'fau':
      if kObs is None:
        f_a_u = 'u'

    key = (k,kObs,f_a_u)

    LP      = self.config.liveplotting
    store_u = self.config.store_u

    if not (LP or store_u) and kObs==None:
      pass # Skip assessment
    else:
      # Prepare assessment call and arguments
      if self._is_ens:
        # Ensemble assessment
        alias = self.assess_ens
        state_prms = {'E':E,'w':w}
      else:
        # Linear-Gaussian assessment
        alias = self.assess_ext
        state_prms = {'mu':mu,'P':Cov}

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


      # LivePlot
      if LP:
        if k==0:
          self.lplot = LivePlot(self,**state_prms,only=LP)
        elif 'u' in f_a_u:
          self.lplot.update(k,kObs,**state_prms)


  def assess_ens(self,k,E,w=None):
    """Ensemble and Particle filter (weighted/importance) assessment."""
    # Unpack
    N,m = E.shape
    x = self.xx[k[0]]

    # Process weights
    if w is None: 
      self._has_w = False
      w           = 1/N
    else:
      self._has_w = True
    if np.isscalar(w):
      assert w   != 0
      w           = w*ones(N)

    if abs(w.sum()-1) > 1e-5:      raise_AFE("Weights did not sum to one.",k)
    if not np.all(np.isfinite(E)): raise_AFE("Ensemble not finite.",k)
    if not np.all(np.isreal(E)):   raise_AFE("Ensemble not Real.",k)

    self.w[k]    = w
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
    self.skew[k] = mean( w @ A_pow / self.var[k]**(3/2) )
    A_pow       *= A # idem.
    self.kurt[k] = mean( w @ A_pow / self.var[k]**2 - 3 )

    self.derivative_stats(k,x)

    if sqrt(m*N) <= Stats.comp_threshold_3:
      if N<=m:
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
      self.rh[k]    = [np.where(Ex_sorted[:,i] == x[i])[0][0] for i in range(m)]


  def assess_ext(self,k,mu,P):
    """Kalman filter (Gaussian) assessment."""

    isFinite = np.all(np.isfinite(mu)) # Do not check covariance
    isReal   = np.all(np.isreal(mu))   # (coz might not be explicitly availble)
    if not isFinite: raise_AFE("Estimates not finite.",k)
    if not isReal:   raise_AFE("Estimates not Real.",k)

    m = len(mu)
    x = self.xx[k[0]]

    self.mu[k]  = mu
    self.var[k] = P.diag if isinstance(P,CovMat) else diag(P)
    self.mad[k] = sqrt(self.var[k])*sqrt(2/pi)
    # ... because sqrt(2/pi) = ratio MAD/STD for Gaussians

    self.derivative_stats(k,x)

    if m <= Stats.comp_threshold_3:
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
    m              = len(self.err[k])
    ldet           = log(self.var[k]).sum()
    nmisf          = self.var[k]**(-1/2) * self.err[k]
    logp_m         = (nmisf**2).sum() + ldet
    self.logp_m[k] = logp_m/m


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
          t = self.setup.t
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
      avrg[key] = series_mean_with_conf(series[self.setup.t.maskObs_BI])
    return avrg



  def new_FAU_series(self,m,**kwargs):
    "Convenience FAU_series constructor."
    store_u = self.config.store_u
    return FAU_series(self.setup.t, m, store_u=store_u, **kwargs)

  # TODO: Provide frontend initializer 

  # Better to initialize manually (np.full...)
  # def new_array(self,f_a_u,m,**kwargs):
  #   "Convenience array constructor."
  #   t = self.setup.t
  #   # Convert int-len to shape-tuple
  #   if is_int(m):
  #     if m==1: m = ()
  #     else:    m = (m,)
  #   # Set length
  #   if f_a_u=='a':
  #     K = t.KObs
  #   elif f_a_u=='u':
  #     K = t.K
  #   #
  #   return np.full((K+1,)+m,**kwargs)



def average_each_field(ss,axis=None):
  "Average a 2D table (where T[i,j] is a dict of fields) along a given axis."
  assert ss.ndim == 2
  if axis == 0:
    ss = np.transpose(ss)
  m,N = ss.shape
  avrg = np.empty(m,dict)
  for i,row in enumerate(ss):
    avrg[i] = dict()
    for key in ss[i][0].keys():
      avrg[i][key] = val_with_conf(
          val  = mean([s_ij[key].val  for s_ij in row]),
          conf = mean([s_ij[key].conf for s_ij in row])/sqrt(N))
      # NB: This is a rudimentary averaging of confidence intervals
      # Should be checked against variance of avrg[i][key].val
  return avrg



