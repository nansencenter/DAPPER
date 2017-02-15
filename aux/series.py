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
  Fit an empirical acf by the acf of an AR1 process.
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
  def __str__(self):
    conf = round2sigfig(self.conf)
    nsig = floor(log10(conf))
    return str(round2(self.val,10**(nsig))) + ' ±' + str(conf)
  def __repr__(self):
    return str(self.__dict__)

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
  if N < 5:
    return val_with_conf(mu, np.nan)
  acovf = auto_cov(xx,5)
  v     = acovf[0]
  v    /= N
  # Estimate (fit) ACF
  # Empirical auto cov function (ACF)
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


class Fseries:
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
  When writing .u,
    - if not store_u: only the current value is stored 
    - if kObs!=None : also .a is written to.
  Data may also be accessed through raw attributes
    - .a, .f, .u.
  """
  def __init__(self,chrono,m,store_u=True,**kwargs):

    self.store_u = store_u
    self.chrono  = chrono

    # Convert int-len to shape-tuple
    self.m = m # store first
    if isinstance(m,int):
      if m==1: m = ()
      else:    m = (m,)

    self.a   = zeros((chrono.KObs+1,)+m, **kwargs)
    self.f   = zeros((chrono.KObs+1,)+m, **kwargs)
    if self.store_u:
      self.u = zeros((chrono.K   +1,)+m, **kwargs)
    else:
      self.tmp   = None
      self.k_tmp = None
  
  def validate_key(self,key):
    try:
      assert isinstance(key,tuple)
      k,kObs,fa = key
      assert fa in ['f','a','u']
      if kObs is not None and k != self.chrono.kkObs[kObs]:
        raise KeyError("kObs indicated, but k!=kkObs[kObs]")
    except (AssertionError,ValueError):
      key = (key,None,'u')
    return key

  def __getitem__(self,key):
    k,kObs,fa = self.validate_key(key)
    if fa == 'f':
      return self.f[kObs]
    elif fa == 'a':
      return self.a[kObs]
    else:
      if self.store_u:
        return self.u[k]
      else:
        if self.k_tmp is not k:
          msg = "Only item [" + str(self.k_tmp) + "] is available from " +\
              "the universal (.u) series (since store_u=False). " +\
              "Maybe use analysis (.a) or forecast (.f) arrays instead?"
          raise KeyError(msg)
        return self.tmp

  def __setitem__(self,key,item):
    k,kObs,fa = self.validate_key(key)
    if fa == 'f':
      self.f[kObs]   = item
    elif fa == 'a':  
      self.a[kObs]   = item
    else:
      if self.store_u:
        self.u[k]    = item
      else:
        self.k_tmp   = k
        self.tmp     = item
      if kObs!=None:
        # Also set .a
        self.a[kObs] = item

  def average(self):
    """
    Avarage series,
    but only if it's univariate (scalar).
    """
    if self.m > 1:
      raise NotImplementedError
    avrg = {}
    t = self.chrono
    for sub in ['a','f','u']:
      if sub=='u':
        inds = t.kk_BI
      else:
        inds = t.maskObs_BI
      if hasattr(self,sub):
        series = getattr(self,sub)[inds]
        avrg[sub] = series_mean_with_conf(series)
    return avrg

  def __repr__(self):
    s = []
    s.append("\nAnalysis (.a):")
    s.append(self.a.__str__())
    s.append("\nForecast (.f):")
    s.append(self.f.__str__())
    s.append("\nAll (.u):")
    if self.store_u:
      s.append(self.u.__str__())
    else:
      s.append("Not stored")
    return '\n'.join(s)




