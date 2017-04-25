from common import *

# Useful matrix to toy with.
try:
  from Misc.magic_square import magic
except ImportError:
  pass

# I also recommend the package "rogues",
# which replicates Matlab's matrix gallery.

def randcov(m):
  """(Makeshift) random cov mat
  (which is missing from rogues)"""
  N = int(ceil(2+m**1.2))
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

def genOG_modified(m,opts=(0,1.0)):
  """
  genOG with modifications.
  Caution: although 'degree' ∈ (0,1) for all versions,
           they're not supposed going to be strictly equivalent.
  Testing: scripts/sqrt_rotations.py
  """

  # Parse opts
  if not opts:
    # Shot-circuit in case of False or 0
    return eye(m)
  elif isinstance(opts,bool):
    return genOG(m)
  elif isinstance(opts,float):
    ver    = 1
    degree = opts
  else:
    ver    = opts[0]
    degree = opts[1]

  if ver==1:
    # Only rotate "once in a while"
    dc = 1/degree # = "while"
    # Retrieve/store persistent variable
    counter = getattr(genOG_modified,"counter",0) + 1
    setattr(genOG_modified,"counter",counter)
    # Compute rot or skip
    if np.mod(counter, dc) < 1:
      Q = genOG(m)
    else:
      Q = eye(m)
  elif ver==2:
    # Background knowledge
    # stackoverflow.com/questions/38426349
    # https://en.wikipedia.org/wiki/Orthogonal_matrix
    Q   = genOG(m)
    s,U = sla.eig(Q)
    s2  = exp(1j*np.angle(s)*degree) # reduce angles
    Q   = mrdiv(U * s2, U)
    Q   = Q.real
  elif ver==3:
    # Reduce Given's rotations in QR algo
    raise NotImplementedError
  elif ver==4:
    # Introduce correlation between columns of randn((m,m))
    raise NotImplementedError
  else:
    raise KeyError
  return Q


# This is actually very cheap compared to genOG,
# so caching doesn't help much.
@functools.lru_cache(maxsize=1)
def basis_beginning_with_ones(ndim):
  """Basis whose first vector is ones(ndim)."""
  e = ones((ndim,1))
  return nla.svd(e)[0]

def genOG_1(N,opts=()):
  """
  Random orthonormal mean-preserving matrix.
  Source: ienks code of Sakov/Bocquet.
  """
  V = basis_beginning_with_ones(N)
  if opts==():
    Q = genOG(N-1)
  else:
    Q = genOG_modified(N-1,opts)
  return V @ sla.block_diag(1,Q) @ V.T




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
      #d    = np.maximum(d,0)
      d    = np.where(d<1e-10,0,d)
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
    else: raise TypeError

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
    return self.transform_by(sqrt,'econ')

  @lazy_property
  def inv(self):
    return self.transform_by(lambda x: 1/x)

  @lazy_property
  def m12(self):
    return self.transform_by(lambda x: 1/sqrt(x),'econ')

  @property
  def cholL(self):
    # L = sla.cholesky(self.C,lower=True)
    # C = L @ L.T
    return self.ssqrt

  @property
  def cholU(self):
    # U = sla.cholesky(self.C,lower=False)
    # C = U.T @ U
    return self.ssqrt

  def __str__(self):
    return str(self.C)
  def __repr__(self):
      return self.__str__()

from scipy import sparse as sprs
class spCovMat():
  '''
  Sparse version of CovMat.
  Careful: if is_sprs it will yield sparse **matrices**,
           which interpret '*' as dot().
  '''
  def __init__(self,C=None,C12=None,ssqrtm=None,E=None,A=None,diagnl=None,trunc=0.99):

    inpt = [C,C12,ssqrtm,E,A,diagnl]
    has_no_None = False
    for arg in inpt:
      if arg is not None:
        if has_no_None:
          raise ValueError('Too many inputs')
        else:
          has_no_None = True
    assert 0 < trunc <= 1

    is_sprs  = False
    if E is not None:
      mu       = mean(E,0)
      A        = E - mu
    if A is not None:
      N        = A.shape[0]
      C12      = A.T / sqrt(N-1)
    if ssqrtm is not None:
      C12      = ssqrtm
    if C12 is not None:
      m        = C12.shape[0]
      U,d12,VT = svd(C12,full_matrices=False)
      d        = d12**2
    elif C is not None:
      C        = np.atleast_2d(C)
      m        = C.shape[0]
      assert     C.shape[1] == m
      d,U      = eigh(C)
      d        = np.flipud(d)
      U        = np.fliplr(U)
    elif diagnl is not None:
      is_sprs  = True
      d        = np.atleast_1d(diagnl)
      assert     is1d(d)
      m        = len(d)
      sInds    = arange(m) if np.all(d == d[0]) else np.argsort(d)[::-1] 
      d        = d[sInds]
      U        = sprs.csr_matrix((ones(m),(sInds,arange(m))))
    else: raise TypeError('Input missing')

    assert np.all(np.isreal(d)) and np.all(d>=0)
    assert np.all(np.isreal(U))

    rk = (d > 1e-13*mean(d)).sum()
    if trunc < 1:
      rk = 1 + find_1st_ind(np.cumsum(d)>=trunc*d.sum())

    self._U      = U[:,:rk]
    self._d      = d[  :rk]
    self.m       = m
    self.rk      = rk
    self.is_sprs = is_sprs

  def transform_by(self,fun):
    d  = self._d
    U  = self._U
    if self.is_sprs:
      return (U @ sprs.diags(fun(d))) @ U.T
    else:
      return (U * fun(d)) @ U.T

  @lazy_property
  def diagonal(self):
    d = zeros(self.m)
    for i in range(self.m):
      if self.is_sprs:
        d[i] = Uii_2 = self._U[i,:].power(2) * self._d
      else:
        d[i] = Uii_2 = self._U[i,:]**2       @ self._d
    return d

  @lazy_property
  def C(self):
    return self.transform_by(lambda x: x)

  @lazy_property
  def inv(self):
    return self.transform_by(lambda x: 1/x)

  @lazy_property
  def ssqrt(self):
    return self.transform_by(sqrt)

  @lazy_property
  def m12(self):
    return self.transform_by(lambda x: 1/sqrt(x))

  @property
  def cholL(self):
    # with sla.cholesky(self.C,lower=True) as L:
    #   C = L @ L.T
    d  = self._d
    U  = self._U
    if self.is_sprs:
      return U @ sprs.diags(sqrt(d))
    else:
      return U * sqrt(d)

  @property
  def cholU(self):
    # with sla.cholesky(self.C,lower=False) as U:
    #   C = U.T @ U
    return self.cholL.T

  def __repr__(self):
    repr_dict = { k: vars(self)[k] for k in ['m','rk','is_sprs'] }
    from pprint import pformat
    s = "<" + type(self).__name__ + ">\n" + pformat(repr_dict, width=1)
    s += '\ndiagonal: ' + str(self.diagonal)
    return s


# Rationale: __matmul__ works fine for myCovMat @ ndarray
# but not ndarray @ myCovMat.
def cvMult(A,B):
  is_transposed = False
  if type(B) is spCovMat and type(A) is np.ndarray:
    A, B = B, A.T
    # don't need to tranpose B, coz it's sym.
    is_transposed = True
  if type(A) is spCovMat:
    assert type(B) is np.ndarray
    U = A._U
    d = A._d
    if A.is_sprs:
      result =  U @ ( sprs.diags(d) @ (U.T @ B) )
    else:
      result = (U * d) @ (U.T @ B)
  else:
    raise TypeError
  if is_transposed:
    result = result.T
  return result





def funm_psd(a, fun, check_finite=False):
  """
  Matrix function evaluation for pos-sem-def mat.
  Adapted from sla.funm() doc.
  e.g.
  def sqrtm_psd(A):
    return funm_psd(A, sqrt)
  """
  w, v = eigh(a, check_finite=check_finite)
  w = np.maximum(w, 0)
  w = fun(w)
  return (v * w) @ v.T


from scipy.linalg.lapack import get_lapack_funcs
def chol_trunc(C):
  """
  Truncate when cholesky() finds a 'leading negative minor'.
  NB: returns a non-square (rank-by-ndim) matrix.
  Example:
    C = E@E.T
    # sla.cholesky(C) yields error coz of numerical error
    U = chol_trunc(C)
    D = C - U.T@U
  D should be close to machine-precision zero,
  especially away from the lower-right-hand corner.
  """
  potrf, = get_lapack_funcs(('potrf',), (C,))
  U, info = potrf(C, lower=False, overwrite_a=False, clean=True)
  if info!=0:
    U = U[:info]
  return U
