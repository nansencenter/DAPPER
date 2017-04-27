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


# TODO: THIS FAILS sometimes. E.g. Q from mods.LA.raanes2015
from scipy.linalg.lapack import get_lapack_funcs
def chol_trunc(C):
  """
  Return U such that C - U.T@U is close to machine-precision zero.
  U is truncated when cholesky() finds a 'leading negative minor'.
  Thus, U is rectangular, with height ∈ [rank, m].
  Example:
    E = randn((20,5))
    C = E@E.T
    # sla.cholesky(C) yields error coz of numerical error
    U = chol_trunc(C)
  """
  potrf, = get_lapack_funcs(('potrf',), (C,))
  U, info = potrf(C, lower=False, overwrite_a=False, clean=True)
  if info!=0:
    U = U[:info]
  return U



class CovMat():
  """
  Covariance matrix class.

  Convenience constructor.

  Convenience transformations with memoization. E.g. shortcut to
    if not hasattr(noise.C,'ssqrt'):
      noise.C.ssqrt = funm_psd(noise.C
  """
  # It's mainly about supporting PSD functionality quickly,
  # not about keeping the memory usage low.
  # That would require a much more detailed implementation,
  # possibly using sparse matrices for the eigenvector matrix.
  # But then you would have the problem that their use involves
  # the **matrix class**, which treats * as dot.
  # Of course, one could try to overload +-@/,
  # but unfortunately there's no right/post-application version of @ and /,
  # which makes this less interesting.

  def __init__(self,data,kind='full_or_diag'):
    """
    The covariance (say P) can be input (specified in the following ways):
    kind    : data
    ----------------------
    'full'  : full m-by-m array (P)
    'diag'  : diagonal of P (assumed diagonal)
    'E'     : ensemble (m-by-N) with sample cov P
    'A'     : as 'E', but pre-centred
    'Right' : any R such that P = R.T@R (e.g. weighted versions of 'A')
    'Left'  : any L such that P = L@L.T
    """

    if kind=='E':
      mu          = mean(data,0)
      data        = data - mu
      kind        = 'A'
    if kind=='A':
      N           = len(data)
      data        = data / sqrt(N-1)
      kind        = 'Right'
    if kind=='Left':
      data        = data.T
      kind        = 'Right'
    if kind=='Right':
      # If a cholesky factor has been input, we will not
      # automatically go for the EVD, seeing as e.g. the
      # diagonal can be computed without it.
      R           = exactly_2d(data)
      self._R     = R
      self._m     = R.shape[1]
    else:
      if kind=='full_or_diag':
        data = np.atleast_1d(data)
        if data.ndim==1 and len(data) > 1: kind = 'diag'
        else:                              kind = 'full'
      if kind=='full':
        # If full has been imput, then we have memory for an EVD,
        # which will probably be put to use in the DA.
        C           = exactly_2d(data)
        self._C     = C
        m           = len(C)
        d,V         = eigh(C)
        d           = CovMat._clip(d)
        rk          = (d>0).sum()
        d           =  d  [-rk:][::-1]
        V           = (V.T[-rk:][::-1]).T
        self._assign_EVD(m,rk,d,V)
      elif kind=='diag':
        # With diagonal input, it would be great to use a sparse
        # (or non-existant) representation of V,
        # but that would require so much other adaption of other code.
        d           = exactly_1d(data)
        self._dg    = d
        m           = len(d)
        if np.all(d==d[0]):
          V   = eye(m)
          rk  = m
        else:
          d   = CovMat._clip(d)
          rk  = (d>0).sum()
          idx = np.argsort(d)[::-1]
          d   = d[idx][:rk]
          nn0 = idx<rk
          V   = zeros((m,rk))
          V[nn0, idx[nn0]] = 1
        self._assign_EVD(m,rk,d,V)
      else:
        raise KeyError

  @staticmethod
  def _clip(d):
    return np.where(d<1e-8*d.max(),0,d)

  def _assign_EVD(self,m,rk,d,V):
      self._m   = m
      self._d   = d
      self._U   = V
      self._rk  = rk

  def _do_EVD(self):
    if not self.has_done_EVD:
      V,s,UT = svd0(self._R)
      m      = UT.shape[1]
      d      = s**2
      d      = CovMat._clip(d)
      rk     = (d>0).sum()
      d      = d [:rk]
      V      = UT[:rk].T
      self._assign_EVD(m,rk,d,V)

  @staticmethod
  def process_diag(d):
    return m,rk,d

  @property
  def m(self):
    """ndims"""
    return self._m
  @property
  def has_done_EVD(self):
    """Whether or not eigenvalue decomposition has been done for matrix."""
    return all([key in vars(self) for key in ['_U','_d','_rk']])
  @property
  def ews(self):
    """Eigenvalues. Only outputs the positive values (i.e. len(ews)==rk)."""
    self._do_EVD()
    return self._d
  @property
  def V(self):
    """Eigenvectors, output corresponding to ews."""
    self._do_EVD()
    return self._U
  @property
  def rk(self):
    """Rank, i.e. the number of positive eigenvalues."""
    self._do_EVD()
    return self._rk

  @lazy_property
  def Left(self):
    """L such that C = L@L.T. Note that L is typically rectangular, but not triangular,
    and that its width is somewhere betwen the rank and m."""
    if hasattr(self,'_R'):
      return self._R.T
    else:
      return self.V * sqrt(self.ews)

  @lazy_property
  def Right(self):
    """R such that C = R.T@R. Note that R is typically rectangular, but not triangular,
    and that its height is somewhere betwen the rank and m."""
    if hasattr(self,'_R'):
      return self._R
    else:
      return self.Left.T
  
  @property
  def full(self):
    "Full covariance matrix"
    if hasattr(self,'_C'):
      return self._C
    else:
      C = self.Left @ self.Left.T
    self._C = C
    return C

  @property
  def diag(self):
    "Diagonal of covariance matrix"
    if hasattr(self,'_dg'):
      dg = self._dg
    elif hasattr(self,'_C'):
      dg = diag(self._C)
    else:
      dg = (self.Left**2).sum(axis=1)
    self._dg = dg
    return dg

  def transform_by(self,fun):
    "Generalize scalar function to covariance matrix via Taylor expansion."
    return (self.V * fun(self.ews)) @ self.V.T
  
  @lazy_property
  def sym_sqrt(self):
    "S such that C = S@S, where S is square."
    return self.transform_by(sqrt)

  @lazy_property
  def sym_sqrt_inv(self):
    "S such that C^{-1} = S@S, where S is square."
    return self.transform_by(lambda x: 1/sqrt(x))

  @lazy_property
  def pinv(self):
    "Pseudo-inverse. Also consider using truncated inverse (tinv)."
    return self.transform_by(lambda x: 1/x)

  @lazy_property
  def inv(self):
    if self.m != self.rk:
      raise RuntimeError("Matrix is rank deficient, "+
          "and cannot be inverted. Use .tinv() instead?")
    return self.pinv

  def __repr__(self):
    s  = "\nm: " + str(self.m)
    s += "\nrk: "
    if self.has_done_EVD:
      s += str(self.rk)
    else:
      s += "<=" + str(self.Right.shape[0])
    s += "\nfull:"
    if hasattr(self,'_C') or np.get_printoptions()['threshold'] > self.m**2:
      # We can afford to compute full matrix
      t = "\n" + str(self.full)
    else:
      # Only compute corners of full matrix
      K  = np.get_printoptions()['edgeitems']
      s += " (only computing corners)"
      if hasattr(self,'_R'):
        U = self.Left[:K ,:] # Upper
        L = self.Left[-K:,:] # Lower
      else:
        U = self.V[:K ,:] * sqrt(self.ews)
        L = self.V[-K:,:] * sqrt(self.ews)

      # Corners
      NW = U@U.T
      NE = U@L.T
      SW = L@U.T
      SE = L@L.T
      
      # Concatenate corners. Fill "cross" between them with nan's
      N  = np.hstack([NW,nan*ones((K,1)),NE])
      S  = np.hstack([SW,nan*ones((K,1)),SE])
      All= np.vstack([N ,nan*ones(2*K+1),S])

      with printoptions(threshold=0):
        t = "\n" + str(All)
    s += t.replace("\n","\n  ")

    with printoptions(threshold=0):
      s += "\ndiag:\n" + "   " + str(self.diag)

    s = repr_type_and_name(self) + s.replace("\n","\n   ")
    return s






