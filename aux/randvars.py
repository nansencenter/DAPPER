# Random variables

from common import *


class RV:
  "Class to represent random variables."
  def __init__(self,m,**kwargs):
    """
     - m    <int>     : ndim
     - is0  <bool>    : if True, the random variable is identically 0
     - func <func(N)> : use this sampling function
     - file <str>     : draw from file
    The following kwords (versions) are available,
    but should not be used for anything serious (use instead subclasses, like GaussRV).
     - icdf <func(x)> : marginal/independent  "inverse transform" sampling
     - cdf <func(x)>  : as icdf, but with inv-cdf approximate, from interpolation 
     - pdf  <func(x)> : "acceptance-rejection" sampling
    """
    # NB: Do not implement "forward to GaussRV constructor"
    # e.g. to handle case of: args=(0,) or (3,).
    # This is better left to <Operator>, which also knows ndims.
    self.m = m
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  def sample(self,N):
    if getattr(self,is0,False):
      # Identically 0
      E = zeros((N,self.m))
    elif hasattr(self,func):
      # Provided by function
      E = self.func(N)
    elif hasattr(self,file):
      # Provided by numpy file with sample
      data = np.load(self.file)
      sample = data['sample']
      if 'w' in data:
        w = data['w']
      else:
        w = ones(N)/N
      idx = np.random.choice(len(sample),N,replace=True,p=w)
      E   = sample[idx]
    elif hasattr(self,icdf):
      # Independent "inverse transform" sampling
      icdf = np.vectorize(icdf)
      uu   = rand((N,m))
      E    = icdf(uu)
    elif hasattr(self,cdf):
      # Like above, but with inv-cdf approximate, from interpolation
      if not hasattr(self,icdf_interp):
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
      uu = rand((N,m))
      E  = self.icdf_interp(uu)
    elif hasattr(self,pdf):
      # "acceptance-rejection" sampling
      raise NotImplementedError
    else:
      raise KeyError
    assert self.m == E.shape[1]
    return E

# TODO: UniRV
# TODO: ExpoRV # use for Nerger
# TODO: TruncGaussRV # for Abhi

class GaussRV(RV,MLR_Print):
  def __init__(self,mu=0,C=0,m=None):
    "Init allowing for shortcut notation."

    # Set mu
    mu = np.atleast_1d(mu)
    assert mu.ndim == 1
    if m is not None:
      if len(mu)==1 and m>1:
        mu = mu * ones(m)
      else:
        assert len(mu) == m

    # Set C
    self.is_random = True
    if (not isinstance(C,CovMat)) and (not isinstance(C,spCovMat)):
      if C is 0:
        self.is_random = False # TODO: Also check for None or zero array?
      if np.isscalar(C):
        C = CovMat(C*np.ones(len(mu)),'diag')
      else:
        C = CovMat(C)

    # Revise mu
    if len(mu) == 1 and C.m > 1:
      mu = mu * ones(C.m)
    else:
      assert len(mu) == C.m

    # Assign
    self.m  = len(mu)
    self.mu = mu
    self.C  = C

  def sample(self,N):
    if not self.is_random:
      D = zeros((N,self.m))
    else:
      if isinstance(self.C,spCovMat):
        D = randn((N,self.C.rk)) @ self.C.cholU
      else:
        D = randn((N,self.m)) @ self.C.cholU
    return self.mu + D


