from common import *

def auto_cov(xx,L=5):
    """Auto covariance function.
    For scalar time series.
    L: lags (offsets) for which to compute acf."""
    assert is1d(xx)
    N = len(xx)
    if N<=L:
      raise ValueError
    mu = np.mean(xx)
    acovf = array([
      sum((xx[:N-i]-mu)*(xx[i:]-mu))/(N-1-i)
      for i in range(L)])
    return acovf

def auto_cov_periodic(xx,L=5):
    """
    Periodic version.
    Assumes that the length of the signal = its period.
    """
    assert is1d(xx)
    N = len(xx)
    if N<=L:
      raise ValueError
    mu = np.mean(xx)
    acovf = array([
      sum(np.roll(xx-mu,i)*(xx-mu))/(N-1)
      for i in range(L)])
    return acovf


#def geometric_mean(xx):
  #return np.exp(mean(log(xx)))
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
  neg_ind = find_1st_ind(array(acf_empir)<=0)
  if neg_ind is not None:
    acf_empir = acf_empir[:neg_ind]
  if len(acf_empir) == 1:
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


# TODO: Use this
class val_with_conf:
  def __init__(self,val,conf):
    self.val  = val
    self.conf = conf
  def __str__(self):
    return str(self.val) + ' ±' + str(round2sigfig(self.conf))
  def __repr__(self):
    return str(self.__dict__)

def series_mean_with_conf(xx):
  """
  Compute series mean.
  Also provide confidence of mean,
  as estimated from its correlation-corrected variance.
  """
  mu    = np.mean(xx)
  N     = len(xx)
  if np.allclose(xx,mu):
    return mu, 0
  if N < 5:
    return mu, np.nan
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
  # c = np.sum([(N-k)*a**k for k in range(1,N)])
  # But this series is analytically tractable:
  c = ( (N-1)*a - N*a**2 + a**(N+1) ) / (1-a)**2
  confidence_correction = 1 + 2/N * c
  v*= confidence_correction
  #sig_fig_std = float('%.1g' % sqrt(decorr_var_of_mu))
  return mu, round2sigfig(sqrt(v))


