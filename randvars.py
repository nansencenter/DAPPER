from common import *

# From stackoverflow.com/q/3012421
class lazy_property(object):
    '''
    Lazy evaluation of property.
    Should represent non-mutable data,
    as it replaces itself.
    '''
    def __init__(self,fget):
      self.fget = fget
      self.func_name = fget.__name__

    def __get__(self,obj,cls):
      if obj is None:
        return None
      value = self.fget(obj)
      setattr(obj,self.func_name,value)
      return value


class CovMat:
  '''
  A positive-semi-def matrix class.
  '''
  def __init__(self,data,kind='C'):
    if kind in ('C12','sqrtm','ssqrt'):
      C = data.dot(data.T)
      kind = 'C'
    elif is1d(data) and data.size > 1:
      kind = 'diag'
    if kind is 'C':
      C    = data
      m    = data.shape[0]
      d,U  = eigh(data)
      d    = np.maximum(d,0)
      rk   = (d>0).sum()
    elif kind is 'diag':
      data  = asarray(data)
      assert is1d(data)
      #d,U = eigh(diag(data))
      C     = diag(data)
      m     = len(data)
      rk    = (data>0).sum()
      sInds = np.argsort(data)
      d,U   = zeros(m), zeros((m,m))
      for i in range(m):
        U[sInds[i],i] = 1
        d[i] = data[sInds[i]]
    else: raise ValueError

    self.C  = C
    self.U  = U
    self.d  = d
    self.m  = m
    self.rk = rk

  def transform_by(self,f,decomp='full'):
    if decomp is 'full':
      U = self.U
      d = self.d
    else:
      d = self.d[  -self.rk:]
      U = self.U[:,-self.rk:]
    return (U * f(d)) @ U.T

  @lazy_property
  def ssqrt(self):
    return self.transform_by(np.sqrt,'econ')

  @lazy_property
  def inv(self):
    return self.transform_by(lambda x: 1/x)

  @property
  def cholL(self):
    #return sla.cholesky(self.C,lower=True)
    return self.ssqrt

  @property
  def cholU(self):
    #return sla.cholesky(self.C,lower=False)
    return self.ssqrt

  def __str__(self):
    return str(self.C)
  def __repr__(self):
      return self.__str__()
  

class GaussRV:
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
    if isinstance(C,CovMat):
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
      D = randn((N,self.m)) @ self.C.cholU
    return self.mu + D

  def __str__(self):
    s = []
    printable = ['mu','C']
    for k in printable:
      s.append('{}:\n'.format(k) + str(getattr(self,k)))
    return '\n'.join(s)

  def __repr__(self):
      return self.__str__()

    

class RV:
  def __init__(self,data):
    self.pdf = None
    self.cdf = None
    self.sampling_func = None
    self.sample_file = None
  #...
  def sample(self,N):
    if self.is0:
      E = zeros(self.m, N)
    elif self.sampling_func:
      return self.sampling_func(N)
    elif self.cdf:
      pass
      # multivariate CDF sampling
    elif self.pdf:
      pass
      # A-R sampling






def funm_psd(a, func, check_finite=False):
  """Adapted from sla.funm doc.
  Matrix function evaluation for pos-sem-def mat."""
  w, v = eigh(a, check_finite=check_finite)
  w = np.maximum(w, 0)
  w = func(w)
  return (v * w) @ v.T

def sqrtm_psd(A):
  return funm_psd(A, np.sqrt)

