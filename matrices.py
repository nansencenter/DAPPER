from common import *

# Also recommend the package "rogues",
# which replicates Matlab's matrix gallery.

try:
  from magic_square import magic
except ImportError:
  pass


def randcov(m):
  """(Makeshift) random cov mat
  (which is missing from rogues)"""
  N = ceil(2+m**1.2)
  E = randn((N,m))
  return E.T @ E
def randcorr(m):
  """(Makeshift) random corr mat
  (which is missing from rogues)"""
  Cov  = randcov(m)
  Dm12 = diag(diag(Cov)**(-0.5))
  return Dm12@Cov@Dm12


def genOG(m):
  """Generate random orthonormal matrix."""
  Q,R = nla.qr(randn((m,m)))
  for i in range(m):
    if R[i,i] < 0:
      Q[:,i] = -Q[:,i]
  return Q

def genOG_1(N):
  """Random orthonormal mean-preserving matrix.
  Source: ienks code of Sakov/Bocquet."""
  e = ones((N,1))
  V = nla.svd(e)[0] # Basis whose first vector is e
  Q = genOG(N-1)     # Orthogonal mat
  return V @ sla.block_diag(1,Q) @ V.T




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
  




def funm_psd(a, func, check_finite=False):
  """Adapted from sla.funm doc.
  Matrix function evaluation for pos-sem-def mat."""
  w, v = eigh(a, check_finite=check_finite)
  w = np.maximum(w, 0)
  w = func(w)
  return (v * w) @ v.T

def sqrtm_psd(A):
  return funm_psd(A, np.sqrt)




