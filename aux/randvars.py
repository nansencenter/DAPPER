# Random variables

from common import *


class RV(MLR_Print):
  "Class to represent random variables."
  def __init__(self,m,**kwargs):
    """
     - m    <int>     : ndim
     - is0  <bool>    : if True, the random variable is identically 0
     - func <func(N)> : use this sampling function. Example:
                        RV(m=4,func=lambda N: rand((N,4))
     - file <str>     : draw from file. Example:
                        RV(m=4,file='data/tmp.npz')
    The following kwords (versions) are available,
    but should not be used for anything serious (use instead subclasses, like GaussRV).
     - icdf <func(x)> : marginal/independent  "inverse transform" sampling. Example:
                        RV(m=4,icdf = scipy.stats.norm.ppf)
     - cdf <func(x)>  : as icdf, but with approximate icdf, from interpolation. Example:
                        RV(m=4,cdf = scipy.stats.norm.cdf)
     - pdf  <func(x)> : "acceptance-rejection" sampling
                        Not implemented.
    """
    self.m = m
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  def sample(self,N):
    if getattr(self,'is0',False):
      # Identically 0
      E = zeros((N,self.m))
    elif hasattr(self,'func'):
      # Provided by function
      E = self.func(N)
    elif hasattr(self,'file'):
      # Provided by numpy file with sample
      data = np.load(self.file)
      sample = data['sample']
      N0     = len(sample)
      if 'w' in data:
        w = data['w']
      else:
        w = ones(N0)/N0
      idx = np.random.choice(N0,N,replace=True,p=w)
      E   = sample[idx]
    elif hasattr(self,'icdf'):
      # Independent "inverse transform" sampling
      icdf = np.vectorize(self.icdf)
      uu   = rand((N,self.m))
      E    = icdf(uu)
    elif hasattr(self,'cdf'):
      # Like above, but with inv-cdf approximate, from interpolation
      if not hasattr(self,'icdf_interp'):
        # Define inverse-cdf
        from scipy.interpolate import interp1d
        from scipy.optimize import fsolve
        cdf    = self.cdf
        Left,  = fsolve(lambda x: cdf(x) - 1e-9    , 0.1)
        Right, = fsolve(lambda x: cdf(x) - (1-1e-9), 0.1)
        xx     = linspace(Left,Right,1001)
        uu     = np.vectorize(cdf)(xx)
        icdf   = interp1d(uu,xx)
        self.icdf_interp = np.vectorize(icdf)
      uu = rand((N,self.m))
      E  = self.icdf_interp(uu)
    elif hasattr(self,'pdf'):
      # "acceptance-rejection" sampling
      raise NotImplementedError
    else:
      raise KeyError
    assert self.m == E.shape[1]
    return E

class GaussRV(RV,MLR_Print):
  """Multivariate, Gaussian (Normal) random variables."""

  # Used by MLR_Print
  ordr_by_linenum = +1

  def __init__(self,mu=0,C=0,m=None):
    """Init allowing for shortcut notation."""

    if isinstance(mu,CovMat):
      raise TypeError("Got a covariance paramter as mu. " +
      "Use kword syntax (C=...) ?")

    # Set mu
    mu = exactly_1d(mu)
    if len(mu)>1:
      if m is None:
        m = len(mu)
      else:
        assert len(mu) == m
    else:
      if m is not None:
        mu = ones(m)*mu

    # Set C
    if isinstance(C,CovMat):
      if m is None:
        m = C.m
    else:
      if C is 0:
        pass # Assign as pure 0!
      else:
        if np.isscalar(C):
          m = len(mu)
          C = CovMat(C*ones(m),'diag')
        else:
          C = CovMat(C)
          if m is None:
            m = C.m

    # Validation
    if len(mu) not in (1,m):
      raise TypeError("Inconsistent shapes of (m,mu,C)")
    if m is None:
      raise TypeError("Could not deduce the value of m")
    try:
      if m!=C.m:
        raise TypeError("Inconsistent shapes of (m,mu,C)")
    except AttributeError:
      pass
    

    # Assign
    self.m  = m
    self.mu = mu
    self.C  = C

  def sample(self,N):
    if self.C is 0:
      D = zeros((N,self.m))
    else:
      R = self.C.Right
      D = randn((N,len(R))) @ R
    return self.mu + D


# TODO: UniRV
# TODO: ExpoRV # for Nerger
# TODO: TruncGaussRV # for Abhi
