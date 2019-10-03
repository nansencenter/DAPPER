from dapper import *

# TODO: change L to 'nlags', with nlags=L-1, to conform with
# the faster statsmodels.tsa.stattools.acf(xx,True,nlags=L-1,fft=False)
def auto_cov(xx,L=5,zero_mean=False,corr=False):
  """
  Auto covariance function, computed along axis 0.
  L   : max lag (offset) for which to compute acf.
  corr: normalize acf by acf[0] so as to return auto-CORRELATION.
  """
  assert L<=len(xx)

  N = len(xx)
  A = xx if zero_mean else center(xx)[0]
  acovf = zeros((L,)+xx.shape[1:])

  for i in range(L):
    Left  = A[arange(N-i)]
    Right = A[arange(i,N)]
    acovf[i] = (Left*Right).sum(0)/(N-i)

  if corr:
    acovf /= acovf[0]

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
  def round(self,mult=1.0):
      """Round intelligently:

      - conf to 1 sigfig.
      - val:
          - to precision: mult*conf.
          - fallback: rc['sigfig']
      """
      with np.errstate(all='ignore'):
        conf = round2sigfig(self.conf,1) 
        val  = self.val
        if not np.isnan(conf) and conf>0:
            val = round2(val, mult*conf)
        else:
            val = round2sigfig(val,rc['sigfig'])
        return val, conf
  def __str__(self):
    val,conf = self.round()
    return str(val)+' ±'+str(conf)
  def __repr__(self):
    val,conf = self.round()
    return type(self).__name__ + "(val="+str(val)+", conf="+str(conf)+")"

def series_mean_with_conf(xx):
  """Compute the mean of a 1d iterable ``xx``.

  Also provide confidence of mean,
  as estimated from its correlation-corrected variance.
  """
  mu = mean(xx)
  N  = len(xx)
  if (not np.isfinite(mu)) or N<=5:
      vc = val_with_conf(mu, np.nan)
  elif np.allclose(xx,mu):
      vc = val_with_conf(mu, 0)
  else:
      acovf = auto_cov(xx,5)
      var   = acovf[0]
      var  /= N
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
      var *= confidence_correction
      vc = val_with_conf(mu, round2sigfig(sqrt(var)))
  return vc


class FAUSt(NestedPrint):
  """Container for time series of a statistic from filtering.

  Four attributes, each of which is an ndarray:

   - .f for forecast      , (KObs+1,)+item.shape
   - .a for analysis      , (KObs+1,)+item.shape
   - .s for smoothed      , (KObs+1,)+item.shape
   - .u for universial/all, (K   +1,)+item.shape

  Series can also be accessed
  >>> self[kObs,'a']
  >>> self[whatever,kObs,'a']
  >>> # ... and likewise for 'f' and 's'. For 'u', can use:
  >>> self[k,'u']
  >>> self[k,whatever,'u']

  .. note:: If a data series only pertains to the analysis,
            then you should use a plain np.array instead.
  """

  # Printing options (see NestedPrint)
  aliases  = {
      'f':'Forecast (.f)',
      'a':'Analysis (.a)',
      's':'Smoothed (.s)',
      'u':'Universl (.u)'}

  def __init__(self,K,KObs,shape,store_u,**kwargs):
    """Constructor.

     - shape   : shape of an item in the series. 
     - store_u : if False: only the current value is stored.
     - kwargs  : passed on to ndarrays.
    """

    # Convert length (an int) to shape
    if not hasattr(shape, '__len__'):
      if shape==1: shape = ()
      else:        shape = (shape,)

    self.f   = np.full((KObs+1,)+shape, nan, **kwargs)
    self.a   = np.full((KObs+1,)+shape, nan, **kwargs)
    self.s   = np.full((KObs+1,)+shape, nan, **kwargs)
    if store_u:
      self.u = np.full((K   +1,)+shape, nan, **kwargs)
    else:
      self.u = np.full((     1,)+shape, nan, **kwargs)

    self.store_u = store_u
    self.shape   = shape
  
  def _ind(self,key):
    if key[-1]=='u': return key[0] if self.store_u else 0
    else           : return key[-2]

  def __setitem__(self,key,item):
    getattr(self,key[-1])[self._ind(key)] = item

  def __getitem__(self,key):
    return getattr(self,key[-1])[self._ind(key)]


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





