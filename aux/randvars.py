# Random variables

from common import *


class RV:
  def __init__(self,func=None,m=None):
    #self.pdf = None
    #self.cdf = None
    #self.sample_file = None
    self.func = func
    self.m = m
    self.is0 = False # TODO: vs is_random in GaussRV
    
  def sample(self,N):
    if self.is0:
      E = zeros(self.m, N)
    elif self.func:
      return self.func(N)
    
    #elif self.cdf: # TODO:
      ## multivariate CDF sampling
    #elif self.pdf:
      ## A-R sampling
    # sample_file ==> pdf, cdf (empirical)


# TODO: UniRV

class GaussRV(RV,MLR_Print):
  def __init__(self,mu=0,C=0,m=None):
    # Set mu
    assert is1d(mu)
    self.mu = asarray(mu).squeeze()
    if m is not None:
      if self.mu.size == 1 and m > 1:
        self.mu = self.mu * ones(m)
      else:
        assert self.mu.size == m
    # Set C
    self.is_random = True
    if isinstance(C,CovMat) or isinstance(C,spCovMat):
      self.C = C
    else:
      if C is 0:
        self.is_random = False
      if np.isscalar(C):
        self.C = CovMat(C * np.ones_like(self.mu),'diag')
      else:
        self.C = CovMat(C)
    # Revise mu
    if self.mu.size == 1 and self.C.m > 1:
      self.mu = self.mu * ones(self.C.m)
    else:
      assert self.mu.size == self.C.m
    # Set length
    self.m = self.mu.size

  def sample(self,N):
    if not self.is_random:
      D = zeros((N,self.m))
    else:
      if isinstance(self.C,spCovMat):
        D = randn((N,self.C.rk)) @ self.C.cholU
      else:
        D = randn((N,self.m)) @ self.C.cholU
    return self.mu + D

  @property
  def is_deterministic(self):
    return not self.is_random

  #def __str__(self):
    #s = []
    #printable = ['mu','C']
    #for k in printable:
      #s.append('{}:\n'.format(k) + str(getattr(self,k)))
    #return '\n'.join(s)

  #def __repr__(self):
      #return self.__str__()



