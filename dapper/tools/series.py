from dapper import *

def auto_cov(xx,L=5,periodic=False,corr=False):
  """
  Auto covariance function, computed along axis 0.
  L   : max lag (offset) for which to compute acf.
  mode: use 'wrap' to assume xx periodic (circular).
  corr: normalize acf by acf[0] so as to return auto-CORRELATION.
  """
  assert xx.ndim <= 2
  assert L<len(xx)

  N     = len(xx)
  mu    = mean(xx,0)
  A     = xx - mu
  acovf = zeros((L,)+np.shape(mu))

  for i in range(L):
    Left     = A.take(arange(N-i),0)
    Right    = A.take(arange(i,N),0)
    acovf[i] = (Left*Right).sum(0)/(N-i-1)

  if corr:
    acovf /= acovf[0].copy()

  return acovf


def fit_acf_by_AR1(acf_empir,L=None):
  """
  Fit an empirical auto cov function (ACF) by that of an AR1 process.
  acf_empir: auto-corr/cov-function.
  L: length of ACF to use in AR(1) fitting
  """
  if L is None:
    L = len(acf_empir)

  geometric_mean = ss.mstats.gmean # = exp(mean(log(xx)))
  def mean_ratio(xx):
    return geometric_mean([xx[i]/xx[i-1] for i in range(1,len(xx))])

  # Negative correlation => Truncate ACF
  neg_ind   = find_1st_ind(array(acf_empir)<=0)
  acf_empir = acf_empir[:neg_ind]
  if   len(acf_empir) == 0: return 0
  elif len(acf_empir) == 1: return 0.01
  else:                     return mean_ratio(acf_empir)

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
    with np.errstate(all='ignore'):
      conf = round2sigfig(self.conf)
      nsig = max(-10,floor(log10(conf)))
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
  if np.allclose(xx,mu):            return val_with_conf(mu, 0)
  if (not np.isfinite(mu)) or N<=5: return val_with_conf(mu, np.nan)
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


class FAU_series(NestedPrint):
  """Container for time series of a statistic from filtering.

  Data is indexed with key (k,kObs,f_a_u) or simply k.
  The accessing then categorizes the result as

   - forecast   (.f, len: KObs+1)
   - analysis   (.a, len: KObs+1)
   - smoothed   (.s, len: KObs+1)

   - universial (.u, len: K+1)

     - also contains time instances where there are no obs.
       These intermediates are nice for plotting.
     - may also be hijacked to store "smoothed" values.

  Data may also be accessed through raw attributes [.a, .f, .s, .u].

  .. note:: if time series is only from analysis instances (len KObs+1),
            then you should use a simple np.array instead.
  """

  # Printing options (cf. NestedPrint)
  included = NestedPrint.included + ['f','a','s','store_u']
  aliases  = {
      'f':'Forecast (.f)',
      'a':'Analysis (.a)',
      's':'Smoothed (.s)',
      'u':'All      (.u)'}
  aliases  = {**NestedPrint.aliases, **aliases}

  def __init__(self,chrono,M,store_u=True,**kwargs):
    """
    Constructor.
     - chrono  : a Chronology object.
     - M       : len (or shape) of items in series. 
     - store_u : if False: only the current value is stored.
     - kwargs  : passed on to ndarrays.
    """

    self.store_u = store_u
    self.chrono  = chrono

    # Convert int-len to shape-tuple
    self.M = M # store first
    if is_int(M):
      if M==1: M = ()
      else:    M = (M,)

    self.a   = np.full((chrono.KObs+1,)+M, nan, **kwargs)
    self.f   = np.full((chrono.KObs+1,)+M, nan, **kwargs)
    self.s   = np.full((chrono.KObs+1,)+M, nan, **kwargs)
    if self.store_u:
      self.u = np.full((chrono.K   +1,)+M, nan, **kwargs)
    else:
      self.tmp   = np.full(M, nan, **kwargs)
      self.k_tmp = None
  
  def validate_key(self,key):
    try:
      # Assume key = (k,kObs,fau)
      if not isinstance(key,tuple):                     raise ValueError
      k,kObs,fau = key                       #Will also raise ValueError
      if not isinstance(fau,str):                       raise ValueError
      if not all([letter in 'fasu' for letter in fau]): raise ValueError
      #
      if kObs is None:
        for ltr in 'afs':
          if ltr in fau:
            raise KeyError("Accessing ."+ltr+" series, but kObs is None.")
      # NB: The following check has been disabled, because
      # it is actually very time consuming when kkObs is long (e.g. 10**4):
      # elif k != self.chrono.kkObs[kObs]: raise KeyError("kObs indicated, but k!=kkObs[kObs]")
    except ValueError:
      # Assume key = k
      assert not hasattr(key, '__getitem__'), "Key must be 1-dimensional."
      key = (key,None,'u')
    return key

  def split_dims(self,k):
    "Split (k,kObs,fau) into k, (kObs,fau)"
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
    if 's' in fau:
      self.s[kObs]   = item
    if 'u' in fau:
      if self.store_u:
        self.u[k]    = item
      else:
        k0, k1       = self.split_dims(k)
        self.k_tmp   = k0
        self.tmp[k1] = item

  def __getitem__(self,key):
    k,kObs,fau = self.validate_key(key)

    if len(fau)>1:
      # Check consistency. NB: Somewhat time-consuming.
      for sub in fau[1:]:
        i1 = self[k,kObs,sub]
        i2 = self[k,kObs,fau[0]]
        if np.any(i1!=i2):
          if not (np.all(np.isnan(i1)) and np.all(np.isnan(i2))):
            raise RuntimeError(
              "Requested item corresponding to multiple arrays ('%s'), "%fau +\
              "But the items are not equal.")

    if 'f' in fau:
      return self.f[kObs]
    elif 'a' in fau:
      return self.a[kObs]
    elif 's' in fau:
      return self.s[kObs]
    else:
      if self.store_u:
        return self.u[k]
      else:
        k0, k1 = self.split_dims(k)
        if self.k_tmp != k0:
          msg = "Only item [" + str(self.k_tmp) + "] is available from "+\
          "the universal (.u) series. One possible source of error "+\
          "is that the data has not been computed for entry k="+str(k0)+". "+\
          "Another possibility is that it has been cleared; "+\
          "if so, a fix might be to set store_u=True, "+\
          "or to use analysis (.a), forecast (.f), or smoothed (.s) arrays instead."
          raise KeyError(msg)
        return self.tmp[k1]

  def average(self):
    """
    Avarage series,
    but only if it's univariate (scalar).
    """
    if self.M > 1:
      raise NotImplementedError
    avrg = {}
    t = self.chrono
    for sub in 'afsu':
      if sub=='u':
        inds = t.kk[t.mask_BI]
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



class RollingArray:
  """ND-Array that implements "leftward rolling" along axis 0.
  Used for data that gets plotted in sliding graphs."""
  
  def __init__(self, shape, fillval=nan):
      self.array = np.full(shape, fillval)
      self.k1 = 0      # previous k
      self.nFilled = 0 # 

  def insert(self,k,val):
    dk = k-self.k1

    # Old (more readable?) version:
    # if dk in [0,1]: # case: forecast or analysis update
      # self.array = np.roll(self.array, -1, axis=0)
    # elif dk>1:      # case: user has skipped ahead (w/o liveplotting)
      # self.array = np.roll(self.array, -dk, axis=0)
      # self.array[-dk:] = nan
    # self.array[-1] = val

    dk = max(1,dk)
    self.array = np.roll(self.array, -dk, axis=0)
    self.array[-dk:] = nan
    self.array[-1 :] = val

    self.k1 = k
    self.nFilled = min(len(self), self.nFilled+dk)

  def leftmost(self):
    return self[len(self)-self.nFilled]

  def span(self):
    return (self.leftmost(),  self[-1])

  @property
  def T(self):
    return self.array.T

  def __array__  (self,dtype=None): return self.array
  def __len__    (self):            return len(self.array)
  def __repr__   (self):            return 'RollingArray:\n%s'%str(self.array)
  def __getitem__(self,key):        return self.array[key]
  def __setitem__(self,key,val):
    # Don't implement __setitem__ coz leftmost() is then
    # not generally meaningful (i.e. if an element is set in the middle).
    # Of course self.array can still be messed with.
    raise AttributeError("Values should be set with update()")





