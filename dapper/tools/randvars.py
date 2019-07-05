# Random variables

from dapper import *


class RV(NestedPrint):
  "Class to represent random variables."

  # Used by NestedPrint
  ordr_by_linenum = +1

  def __init__(self,M,**kwargs):
    """
     - M    <int>     : ndim
     - is0  <bool>    : if True, the random variable is identically 0
     - func <func(N)> : use this sampling function. Example:
                        RV(M=4,func=lambda N: rand((N,4))
     - file <str>     : draw from file. Example:
                        RV(M=4,file=dirs['data']+'/tmp.npz')
    The following kwords (versions) are available,
    but should not be used for anything serious (use instead subclasses, like GaussRV).
     - icdf <func(x)> : marginal/independent  "inverse transform" sampling. Example:
                        RV(M=4,icdf = scipy.stats.norm.ppf)
     - cdf <func(x)>  : as icdf, but with approximate icdf, from interpolation. Example:
                        RV(M=4,cdf = scipy.stats.norm.cdf)
     - pdf  <func(x)> : "acceptance-rejection" sampling
                        Not implemented.
    """
    self.M = M
    for key, value in kwargs.items():
      setattr(self, key, value)

  def sample(self,N):
    if getattr(self,'is0',False):
      # Identically 0
      E = zeros((N,self.M))
    elif hasattr(self,'func'):
      # Provided by function
      E = self.func(N)
    elif hasattr(self,'file'):
      # Provided by numpy file with sample
      data   = np.load(self.file)
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
      uu   = rand((N,self.M))
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
      uu = rand((N,self.M))
      E  = self.icdf_interp(uu)
    elif hasattr(self,'pdf'):
      # "acceptance-rejection" sampling
      raise NotImplementedError
    else:
      raise KeyError
    assert self.M == E.shape[1]
    return E


class RV_with_mean_and_cov(RV):
  """Generic multivariate random variable characterized by two parameters: mean and covariance.

  This class must be subclassed to provide sample(),
  i.e. its main purpose is provide a common convenience constructor.
  """

  def __init__(self,mu=0,C=0,M=None):
    """Init allowing for shortcut notation."""

    if isinstance(mu,CovMat):
      raise TypeError("Got a covariance paramter as mu. " +
      "Use kword syntax (C=...) ?")

    # Set mu
    mu = exactly_1d(mu)
    if len(mu)>1:
      if M is None:
        M = len(mu)
      else:
        assert len(mu) == M
    else:
      if M is not None:
        mu = ones(M)*mu

    # Set C
    if isinstance(C,CovMat):
      if M is None:
        M = C.M
    else:
      if C is 0:
        pass # Assign as pure 0!
      else:
        if np.isscalar(C):
          M = len(mu)
          C = CovMat(C*ones(M),'diag')
        else:
          C = CovMat(C)
          if M is None:
            M = C.M

    # Validation
    if len(mu) not in (1,M):
      raise TypeError("Inconsistent shapes of (M,mu,C)")
    if M is None:
      raise TypeError("Could not deduce the value of M")
    try:
      if M!=C.M:
        raise TypeError("Inconsistent shapes of (M,mu,C)")
    except AttributeError:
      pass
    
    # Assign
    self.M  = M
    self.mu = mu
    self.C  = C

  def sample(self,N):
    """Sample N realizations. Returns N-by-M (ndim) sample matrix.

    Example::

      plt.scatter(*(UniRV(C=randcov(2)).sample(10**4).T))
    """
    if self.C is 0:
      D = zeros((N,self.M))
    else:
      D = self._sample(N)
    return self.mu + D

  def _sample(self,N):
    raise NotImplementedError("Must be implemented in subclass")


class GaussRV(RV_with_mean_and_cov):
  """Gaussian (Normal) multivariate random variable."""
  def _sample(self,N):
    R = self.C.Right
    D = randn((N,len(R))) @ R
    return D

class LaplaceRV(RV_with_mean_and_cov):
  """
  Laplace (double exponential) multivariate random variable.
  This is an elliptical generalization. Ref:
  Eltoft (2006) "On the Multivariate Laplace Distribution".
  """
  def _sample(self,N):
    R = self.C.Right
    z = np.random.exponential(1,N)
    D = randn((N,len(R)))
    D = z[:,None]*D
    return D @ R / sqrt(2)

class LaplaceParallelRV(RV_with_mean_and_cov):
  """
  A NON-elliptical multivariate generalization of
  the Laplace (double exponential) random variable.
  """
  def _sample(self,N):
    #R = self.C.Right   # contour: sheared rectangle
    R = self.C.sym_sqrt # contour: rotated rectangle
    D = np.random.laplace(0,1,(N,len(R)))
    return D @ R / sqrt(2)


class StudRV(RV_with_mean_and_cov):
  """
  Student-t multivariate random variable.
  Assumes the covariance exists,
  which requires degreee-of-freedom (dof) > 1+ndim.
  Also requires that dof be integer,
  since chi2 is sampled via Gaussians.
  """
  def __init__(self,dof,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.dof = dof
  def _sample(self,N):
    R = self.C.Right
    nu= self.dof
    r = nu/np.sum(randn((N,nu))**2,axis=1) # InvChi2
    D = sqrt(r)[:,None]*randn((N,len(R)))
    return D @ R * sqrt((nu-2)/nu)

class UniRV(RV_with_mean_and_cov):
  """
  Uniform multivariate random variable.
  with an elliptic-shape support.
  Ref: Voelker et al. (2017) "Efficiently sampling
  vectors and coordinates from the n-sphere and n-ball"
  """
  def _sample(self,N):
    R = self.C.Right
    D = randn((N,len(R)))
    r = rand(N)**(1/len(R)) / np.sqrt(np.sum(D**2,axis=1))
    D = r[:,None]*D
    return D @ R * 2

class UniParallelRV(RV_with_mean_and_cov):
  """
  Uniform multivariate random variable,
  with a parallelogram-shaped support, as determined by the cholesky factor
  applied to the (corners of) the hypercube.
  """
  def _sample(self,N):
    R = self.C.Right
    D = rand((N,len(R)))-0.5
    return D @ R * sqrt(12)

