# Misc math

from common import *

def ismat(x):
  return (type(x) is np.matrixlib.defmatrix.matrix)

def vec2list2(vec):
  return [[x] for x in vec]

def is1d(a):
  """ Works for list and row/column arrays and matrices"""
  return np.sum(asarray(asarray(a).shape) > 1) <= 1

def tp(a):
  """Tranpose 1d vector"""
  return a[np.newaxis].T

def atmost_2d(func):
  """Decorator to make functions work for 0,1,or 2-dim input.
  Requires that the 1st argument be the one of interest."""
  def wrapr(x,*kargs,**kwargs):
    answer = func(np.atleast_2d(x),*kargs,**kwargs)
    if answer is not None: return answer.squeeze()
  return wrapr
        
def anom(E):
  mu = mean(E,0)
  A  = E - mu
  return A, mu

# Center sample (but maintain its (expected) variance)
def center(E):
  N = E.shape[0]
  A = E - mean(E,0)
  return sqrt(N/(N-1)) * A

def inflate_ens(E,factor):
  A, mu = anom(E)
  return mu + A*factor

def mrdiv(b,A):
  return nla.solve(A.T,b.T).T

def mldiv(A,b):
  return nla.solve(A,b)

def rk4(f, x0, t, dt):
  k1 = dt * f(t      , x0)
  k2 = dt * f(t+dt/2., x0+k1/2.)
  k3 = dt * f(t+dt/2., x0+k2/2.)
  k4 = dt * f(t+dt   , x0+k3)
  return x0 + (k1 + 2.*(k2 + k3) + k4)/6.0

def round2(num,prec=1.0):
  """Round with specific precision.
  Returns int if prec is int."""
  return np.multiply(prec,np.rint(np.divide(num,prec)))

def round2sigfig(x,nfig=1):
  signs = np.sign(x)
  x *= signs
  return signs*round2(x,10**np.round(np.log10(x)-nfig))


def find_1st_ind(xx):
  try:
    return next(k for k in range(len(xx)) if xx[k])
  except StopIteration:
    return None

def equi_spaced_integers(m,p):
  """Provide a range of p equispaced integers between 0 and m-1"""
  return np.round(linspace(floor(m/p/2),ceil(m-m/p/2-1),p)).astype(int)

class Stats:
  """Contains and computes peformance stats."""
  # TODO: Include skew/kurt?
  def __init__(self,params):
    self.params = params
    m    = params.f.m
    K    = params.t.K
    KObs = params.t.KObs
    #
    self.mu   = zeros((K+1,m))
    self.var  = zeros((K+1,m))
    self.err  = zeros((K+1,m))
    self.rmsv = zeros(K+1)
    self.rmse = zeros(K+1)
    self.trHK = zeros(KObs+1)
    self.rh   = zeros((K+1,m))

  def assess(self,E,x,k):
    assert(type(E) is np.ndarray)
    N,m           = E.shape
    self.mu[k,:]  = mean(E,0)
    A             = E - self.mu[k,:]
    self.var[k,:] = np.sum(A**2,0) / (N-1)
    self.err[k,:] = self.mu[k,:] - x[k,:]
    self.rmsv[k]  = sqrt(mean(self.var[k,:]))
    self.rmse[k]  = sqrt(mean(self.err[k,:]**2))
    Ex_sorted     = np.sort(np.vstack((E,x[k,:])),axis=0,kind='heapsort')
    self.rh[k,:]  = [np.where(Ex_sorted[:,i] == x[k,i])[0][0] for i in range(m)]

  def copy_paste(self,s,kObs):
    for key,val in s.items():
      getattr(self,key)[kObs] = val


def auto_cov(xx,L=5):
    """Auto covariance function.
    For scalar time series.
    L: lags (offsets) for which to compute acf."""
    N = len(xx)
    acovf = [np.cov(xx[:N-i], xx[i:])[0,1] for i in range(L)]
    return acovf

def geometric_mean(xx):
  return np.exp(mean(log(xx)))

def mean_ratio(xx):
  return geometric_mean([xx[i]/xx[i-1] for i in range(1,len(xx))])

def fit_acf_by_AR1(acf_empir,L=None):
  """
  Fit an empirical acf by the acf of an AR1 process.
  acf_empir: a-corr-f or a-cov-f.
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
  """ See mods.LA.fundamentals: homogeneous_1D_cov()
  for some math explanation"""
  acovf = auto_cov(xx,10)
  a     = fit_acf_by_AR1(acovf)
  if a == 0:
    return 0
  return 1/log(1/a)

def series_mean_with_conf(xx):
  """
  Compute series mean.
  Also provide confidence of mean,
  as estimated from its correlation-corrected variance.
  """
  mu = np.mean(xx)
  # Estimate (fit) ACF
  # Empirical auto cov function (ACF)
  acovf = auto_cov(xx,5)
  a = fit_acf_by_AR1(acovf)
  v = acovf[0]
  # If xx[k] where independent of xx[k-1],
  # then std_of_mu is the end of the story.
  # The following corrects for the correlation in the time series.
  #
  # See stats.stackexchange.com/q/90062
  # c = np.sum([(N-k)*a**k for k in range(1,N)])
  # But this series is analytically tractable:
  N = len(xx)
  c = ( (N-1)*a - N*a**2 + a**(N+1) ) / (1-a)**2
  confidence_correction = 1 + 2/N * c
  #
  var_of_mu = v/N
  var_of_mu_decorr = var_of_mu * confidence_correction
  #sig_fig_std = float('%.1g' % sqrt(decorr_var_of_mu))
  return mu, round2sigfig(sqrt(var_of_mu_decorr))



