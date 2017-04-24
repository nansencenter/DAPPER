from common import *

def auto_cov(xx,L=5):
    """Auto covariance function.
    For scalar time series.
    L: lags (offsets) for which to compute acf."""
    assert is1d(xx)
    N = len(xx)
    if N<=L:
      raise ValueError('L (=len(ACF)) must be <= len(xx)')
    mu = mean(xx)
    acovf = array([
      np.sum((xx[:N-i]-mu)*(xx[i:]-mu))/(N-1-i)
      for i in range(L)])
    return acovf

def auto_cov_periodic(xx,L=5):
    """
    Periodic version.
    Assumes that the length of the signal = its period.
    """
    assert is1d(xx)
    N = len(xx)
    mu = mean(xx)
    acovf = array([
      np.sum(np.roll(xx-mu,i)*(xx-mu))/(N-1)
      for i in range(L)])
    return acovf


#def geometric_mean(xx):
  #return exp(mean(log(xx)))
geometric_mean = ss.mstats.gmean

def mean_ratio(xx):
  return geometric_mean([xx[i]/xx[i-1] for i in range(1,len(xx))])

def fit_acf_by_AR1(acf_empir,L=None):
  """
  Fit an empirical auto cov function (ACF) by that of an AR1 process.
  acf_empir: auto-corr/cov-function.
  L: length of ACF to use in AR(1) fitting
  """
  if L is None:
    L = len(acf_empir)
  # Negative correlation => Truncate ACF
  neg_ind   = find_1st_ind(array(acf_empir)<=0)
  acf_empir = acf_empir[:neg_ind]
  if len(acf_empir) <= 1:
    return 0
  return mean_ratio(acf_empir)

def estimate_corr_length(xx):
  """
  For explanation, see mods.LA.core: homogeneous_1D_cov().
  Also note that, for exponential corr function, as assumed here,
  corr(L) = exp(-1) = ca 0.368
  """
  assert is1d(xx)
  acovf = auto_cov(xx,min(100,len(xx)-2))
  a     = fit_acf_by_AR1(acovf)
  if a == 0:
    L = 0
  else:
    L = 1/log(1/a)
  return L


class val_with_conf():
  def __init__(self,val,conf):
    self.val  = val
    self.conf = conf
  def _str(self):
    with np.errstate(divide='ignore'):
      conf = round2sigfig(self.conf)
      nsig = floor(log10(conf))
      return str(round2(self.val,10**(nsig))), str(conf)
  def __str__(self):
    val,conf = self._str()
    return val+' ±'+conf
  def __repr__(self):
    val,conf = self._str()
    return type(self).__name__ + "(val="+val+", conf="+conf+")"

def series_mean_with_conf(xx):
  """
  Compute series mean.
  Also provide confidence of mean,
  as estimated from its correlation-corrected variance.
  """
  mu    = mean(xx)
  N     = len(xx)
  if np.allclose(xx,mu):
    return val_with_conf(mu, 0)
  if (not np.isfinite(mu)) or N<5:
    return val_with_conf(mu, np.nan)
  acovf = auto_cov(xx,5)
  v     = acovf[0]
  v    /= N
  # Estimate (fit) ACF
  a = fit_acf_by_AR1(acovf)
  # If xx[k] where independent of xx[k-1],
  # then std_of_mu is the end of the story.
  # The following corrects for the correlation in the time series.
  #
  # See stats.stackexchange.com/q/90062
  # c = sum([(N-k)*a**k for k in range(1,N)])
  # But this series is analytically tractable:
  c = ( (N-1)*a - N*a**2 + a**(N+1) ) / (1-a)**2
  confidence_correction = 1 + 2/N * c
  v*= confidence_correction
  #sig_fig_std = float('%.1g' % sqrt(decorr_var_of_mu))
  vc = val_with_conf(mu, round2sigfig(sqrt(v)))
  return vc


class FAU_series(MLR_Print):
  """
  Container for time series of a statistic from filtering.
  Data is indexed with key (k,kObs,f_a_u) or simply k.
  The accessing then categorizes the result as
   - forecast   (.f, len: KObs+1)
   - analysis   (.a, len: KObs+1)
   - universial (.u, len: K+1)
     - also contains time instances where there are no obs.
       These intermediates are nice for plotting.
     - may also be hijacked to store "smoothed" values.
  Data may also be accessed through raw attributes [.a, .f, .u].
  NB: if time series is only from analysis instances (len KObs+1),
      then you should use a simple np.array instead.
  """

  # Used by MLR_Print
  included = MLR_Print.included + ['f','a','store_u']
  aliases  = {
      'f':'Forecast (.f)',
      'a':'Analysis (.a)',
      'u':'All      (.u)'}
  aliases  = {**MLR_Print.aliases, **aliases}

  def __init__(self,chrono,m,store_u=True,**kwargs):
    """
    Constructor.
     - chrono  : a Chronology object.
     - m       : len (or shape) of items in series. 
     - store_u : if False: only the current value is stored.
     - kwargs  : passed on to ndarrays.
    """

    self.store_u = store_u
    self.chrono  = chrono

    # Convert int-len to shape-tuple
    self.m = m # store first
    if is_int(m):
      if m==1: m = ()
      else:    m = (m,)

    self.a   = np.full((chrono.KObs+1,)+m, nan, **kwargs)
    self.f   = np.full((chrono.KObs+1,)+m, nan, **kwargs)
    if self.store_u:
      self.u = np.full((chrono.K   +1,)+m, nan, **kwargs)
    else:
      self.tmp   = np.full(m, nan, **kwargs)
      self.k_tmp = None
  
  def validate_key(self,key):
    try:
      # Assume key = (k,kObs,fau)
      if not isinstance(key,tuple):                    raise ValueError
      k,kObs,fau = key                      #Will also raise ValueError
      if not isinstance(fau,str):                      raise ValueError
      if not all([letter in 'fau' for letter in fau]): raise ValueError
      #
      if kObs is None:
        for ltr in 'af':
          if ltr in fau:
            raise KeyError("Accessing ."+ltr+" series, but kObs is None.")
      elif k != self.chrono.kkObs[kObs]:
        raise KeyError("kObs indicated, but k!=kkObs[kObs]")
    except ValueError:
      # Assume key = k
      key = (key,None,'u')
    return key

  def split_dims(self,k):
    if isinstance(k,tuple):
      k1 = k[1:]
      k0 = k[0]
    elif is_int(k):
      k1 = ...
      k0 = k
    else:
      raise KeyError
    return k0, k1

  def __setitem__(self,key,item):
    k,kObs,fau = self.validate_key(key)
    if 'f' in fau:
      self.f[kObs]   = item
    if 'a' in fau:
      self.a[kObs]   = item
    if 'u' in fau:
      if self.store_u:
        self.u[k]    = item
      else:
        k0, k1       = self.split_dims(k)
        self.k_tmp   = k0
        self.tmp[k1] = item

  def __getitem__(self,key):
    k,kObs,fau = self.validate_key(key)

    # Check consistency. NB: Somewhat time-consuming.
    for sub in fau[1:]:
      i1 = self[k,kObs,sub]
      i2 = self[k,kObs,fau[0]]
      if np.any(i1!=i2):
        if not (np.all(np.isnan(i1)) and np.all(np.isnan(i2))):
          raise RuntimeError(
            "Requested item from multiple ('."+fau+"') series, " +\
            "But the items are not equal.")
    if 'f' in fau:
      return self.f[kObs]
    elif 'a' in fau:
      return self.a[kObs]
    else:
      if self.store_u:
        return self.u[k]
      else:
        k0, k1 = self.split_dims(k)
        if self.k_tmp is not k0:
          msg = "Only item [" + str(self.k_tmp) + "] is available from "+\
          "the universal (.u) series. One possible source of error "+\
          "is that the data has not been computed for k="+str(k)+". "+\
          "Another possibility is that it has been cleared; "+\
          "if so, a fix might be to set store_u=True, "+\
          "or to use analysis (.a) or forecast (.f) arrays instead."
          raise KeyError(msg)
        return self.tmp[k1]

  def average(self):
    """
    Avarage series,
    but only if it's univariate (scalar).
    """
    if self.m > 1:
      raise NotImplementedError
    avrg = {}
    t = self.chrono
    for sub in 'afu':
      if sub=='u':
        inds = t.kk_BI
      else:
        inds = t.maskObs_BI
      if hasattr(self,sub):
        series = getattr(self,sub)[inds]
        avrg[sub] = series_mean_with_conf(series)
    return avrg

  def __repr__(self):
    if self.store_u:
      # Create instance version of 'included'
      self.included = self.included + ['u']
    return super().__repr__()



