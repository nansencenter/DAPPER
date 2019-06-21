from dapper import *

# Test matrices
def randcov(M):
  """(Makeshift) random cov mat."""
  N = int(ceil(2+M**1.2))
  E = randn((N,M))
  return E.T @ E
def randcorr(M):
  """(Makeshift) random corr mat."""
  Cov  = randcov(M)
  Dm12 = diag(diag(Cov)**(-0.5))
  return Dm12@Cov@Dm12


def genOG(M):
  """Generate random orthonormal matrix."""
  # TODO: This (using Householder) is (slightly?) wrong, 
  # as per section 4 of mezzadri2006generate.
  Q,R = nla.qr(randn((M,M)))
  for i in range(M):
    if R[i,i] < 0:
      Q[:,i] = -Q[:,i]
  return Q

def genOG_modified(M,opts=(0,1.0)):
  """genOG with modifications.

  Caution: although 'degree' ∈ (0,1) for all versions,
           they're not supposed going to be strictly equivalent.

  Testing: scripts/sqrt_rotations.py
  """

  # Parse opts
  if not opts:
    # Shot-circuit in case of False or 0
    return eye(M)
  elif isinstance(opts,bool) or opts is 1:
    return genOG(M)
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
      Q = genOG(M)
    else:
      Q = eye(M)
  elif ver==2:
    # Decompose and reduce angle of (complex) diagonal. Background:
    # stackoverflow.com/questions/38426349
    # https://en.wikipedia.org/wiki/Orthogonal_matrix
    Q   = genOG(M)
    s,U = sla.eig(Q)
    s2  = exp(1j*np.angle(s)*degree) # reduce angles
    Q   = mrdiv(U * s2, U)
    Q   = Q.real
  elif ver==3:
    # Reduce Given's rotations in QR algo
    raise NotImplementedError
  elif ver==4:
    # Introduce correlation between columns of randn((M,M))
    raise NotImplementedError
  elif ver==5:
     # stats.stackexchange.com/q/25552
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
  """Matrix function evaluation for pos-sem-def mat.

  Adapted from sla.funm() doc.

  Example::

    def sqrtm_psd(A):
        return funm_psd(A, sqrt)
  """
  w, v = eigh(a, check_finite=check_finite)
  w = np.maximum(w, 0)
  w = fun(w)
  return (v * w) @ v.T


def chol_reduce(Right):
  """Return rnk-by-ndim R such that Right.T@Right - R.T@R ≈ 0.

  Example::

    A = mean0(randn((20,5)),axis=1)
    C = A.T @ A
    # sla.cholesky(C) throws error
    R = chol_reduce(A)
    R.shape[1] == 4
  """
  _,sig,UT = sla.svd(Right,full_matrices=False)
  R = sig[:,None]*UT

  # The below is DEPRECATED, coz it fails e.g. with Q from dapper.mods.LA.raanes2015.
  #from scipy.linalg.lapack import get_lapack_funcs
  #potrf, = get_lapack_funcs(('potrf',), (C,))
  #R, info = potrf(C, lower=False, overwrite_a=False, clean=True)
  #if info!=0:
    #R = R[:info]
  # Explanation: R is truncated when cholesky() finds a 'leading negative minor'.
  # Thus, R is rectangular, with height ∈ [rank, M].

  return R



class CovMat():
  """Covariance matrix class.

  Main tasks:
    - Unifying the covariance representations:
      full, diagonal, reduced-rank sqrt.
    - Convenience constructor and printing.
    - Convenience transformations with memoization.
      E.g. replaces:
      >if not hasattr(noise.C,'sym_sqrt'):
      >  S = funm_psd(noise.C, sqrt)
      >  noise.C.sym_sqrt = S
      This (hiding it internally) becomes particularly useful
      if the covariance matrix changes with time (but repeat).
  """

  ##################################
  # Init
  ##################################
  def __init__(self,data,kind='full_or_diag',trunc=1.0):
    """The covariance (say P) can be input (specified in the following ways):

    kind    : data
    ----------------------
    'full'  : full M-by-M array (P)
    'diag'  : diagonal of P (assumed diagonal)
    'E'     : ensemble (N-by-M) with sample cov P
    'A'     : as 'E', but pre-centred by mean(E,axis=0)
    'Right' : any R such that P = R.T@R (e.g. weighted form of 'A')
    'Left'  : any L such that P = L@L.T
    """
    
    # Cascade if's down to 'Right'
    if kind=='E':
      mu      = mean(data,0)
      data    = data - mu
      kind    = 'A'
    if kind=='A':
      N       = len(data)
      data    = data / sqrt(N-1)
      kind    = 'Right'
    if kind=='Left':
      data    = data.T
      kind    = 'Right'
    if kind=='Right':
      # If a cholesky factor has been input, we will not
      # automatically go for the EVD, seeing as e.g. the
      # diagonal can be computed without it.
      R       = exactly_2d(data)
      self._R = R
      self._m = R.shape[1]
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
        M           = len(C)
        d,V         = eigh(C)
        d           = CovMat._clip(d)
        rk          = (d>0).sum()
        d           =  d  [-rk:][::-1]
        V           = (V.T[-rk:][::-1]).T
        self._assign_EVD(M,rk,d,V)
      elif kind=='diag':
        # With diagonal input, it would be great to use a sparse
        # (or non-existant) representation of V,
        # but that would require so much other adaption of other code.
        d         = exactly_1d(data)
        self.diag = d
        M         = len(d)
        if np.all(d==d[0]):
          V   = eye(M)
          rk  = M
        else:
          d   = CovMat._clip(d)
          rk  = (d>0).sum()
          idx = np.argsort(d)[::-1]
          d   = d[idx][:rk]
          nn0 = idx<rk
          V   = zeros((M,rk))
          V[nn0, idx[nn0]] = 1
        self._assign_EVD(M,rk,d,V)
      else:
        raise KeyError

    self._kind  = kind
    self._trunc = trunc


  ##################################
  # Protected
  ##################################
  @property
  def M(self):
    """ndims"""
    return self._m

  @property
  def kind(self):
    """Form in which matrix was specified."""
    return self._kind

  @property
  def trunc(self):
    """Truncation threshold."""
    return self._trunc

  ##################################
  # "Non-EVD" stuff
  ##################################
  @property
  def full(self):
    "Full covariance matrix"
    if hasattr(self,'_C'):
      return self._C
    else:
      C = self.Left @ self.Left.T
    self._C = C
    return C

  @lazy_property
  def diag(self):
    "Diagonal of covariance matrix"
    if hasattr(self,'_C'):
      return diag(self._C)
    else:
      return (self.Left**2).sum(axis=1)

  @property
  def Left(self):
    """L such that C = L@L.T. Note that L is typically rectangular, but not triangular,
    and that its width is somewhere betwen the rank and M."""
    if hasattr(self,'_R'):
      return self._R.T
    else:
      return self.V * sqrt(self.ews)
  @property
  def Right(self):
    """R such that C = R.T@R. Note that R is typically rectangular, but not triangular,
    and that its height is somewhere betwen the rank and M."""
    if hasattr(self,'_R'):
      return self._R
    else:
      return self.Left.T

  ##################################
  # EVD stuff
  ##################################
  def _assign_EVD(self,M,rk,d,V):
      self._m   = M
      self._d   = d
      self._V   = V
      self._rk  = rk

  @staticmethod
  def _clip(d):
    return np.where(d<1e-8*d.max(),0,d)

  def _do_EVD(self):
    if not self.has_done_EVD():
      V,s,UT = svd0(self._R)
      M      = UT.shape[1]
      d      = s**2
      d      = CovMat._clip(d)
      rk     = (d>0).sum()
      d      = d [:rk]
      V      = UT[:rk].T
      self._assign_EVD(M,rk,d,V)

  def has_done_EVD(self):
    """Whether or not eigenvalue decomposition has been done for matrix."""
    return all([key in vars(self) for key in ['_V','_d','_rk']])


  @property
  def ews(self):
    """Eigenvalues. Only outputs the positive values (i.e. len(ews)==rk)."""
    self._do_EVD()
    return self._d
  @property
  def V(self):
    """Eigenvectors, output corresponding to ews."""
    self._do_EVD()
    return self._V
  @property
  def rk(self):
    """Rank, i.e. the number of positive eigenvalues."""
    self._do_EVD()
    return self._rk

  
  ##################################
  # transform_by properties
  ##################################
  def transform_by(self,fun):
    """Generalize scalar functions to covariance matrices
    (via Taylor expansion).
    """

    r = truncate_rank(self.ews,self.trunc,True)
    V = self.V[:,:r]
    w = self.ews[:r]

    return (V * fun(w)) @ V.T
  
  @lazy_property
  def sym_sqrt(self):
    "S such that C = S@S (and i.e. S is square). Uses trunc-level."
    return self.transform_by(sqrt)

  @lazy_property
  def sym_sqrt_inv(self):
    "S such that C^{-1} = S@S (and i.e. S is square). Uses trunc-level."
    return self.transform_by(lambda x: 1/sqrt(x))

  @lazy_property
  def pinv(self):
    "Pseudo-inverse. Uses trunc-level."
    return self.transform_by(lambda x: 1/x)

  @lazy_property
  def inv(self):
    if self.M != self.rk:
      raise RuntimeError("Matrix is rank deficient, "+
          "and cannot be inverted. Use .tinv() instead?")
    # Temporarily remove any truncation
    tmp = self.trunc
    self._trunc = 1.0
    # Compute and restore truncation level
    Inv = self.pinv
    self._trunc = tmp
    return Inv

  ##################################
  # __repr__
  ##################################
  def __repr__(self):
    s  = "\n    M: " + str (self.M)
    s += "\n kind: " + repr(self.kind)
    s += "\ntrunc: " + str (self.trunc)

    # Rank
    s += "\n   rk: "
    if self.has_done_EVD():
      s += str(self.rk)
    else:
      s += "<=" + str(self.Right.shape[0])

    # Full (as affordable)
    s += "\n full:"
    if hasattr(self,'_C') or np.get_printoptions()['threshold'] > self.M**2:
      # We can afford to compute full matrix
      t = "\n" + str(self.full)
    else:
      # Only compute corners of full matrix
      K  = np.get_printoptions()['edgeitems']
      s += " (only computing/printing corners)"
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

    # Indent all of cov array, and add to s
    s += t.replace("\n","\n   ")

    # Add diag. Indent array +1 vs cov array
    with printoptions(threshold=0):
      s += "\n diag:\n   " + " " + str(self.diag)

    s = repr_type_and_name(self) + s.replace("\n","\n  ")
    return s

# TODO? The diagonal representation is NOT memory-efficient.
#
# But there's no simple way of making so, especially since the sparse class
# (which would hold the eigenvectors) is a subclass of the matrix class,
# which interprets * as @, and so, when using this class,
# one would have to be always careful about it
# 
# One could try to overload +-@/ (for CovMat),
# but unfortunately there's no right/post-application version of @ and /
# (indeed, how could there be for binary operators?)
# which makes this less interesting.
# Hopefully this restriction is not an issue,
# as diagonal matrices are mainly used for observation error covariance,
# which are usually not infeasibly large.
#
# Another potential solution is to subclass the sparse matrix,
# and revert its operator definitions to that of ndarray. 
# and use it for the V (eigenvector) matrix that gets output
# by various fields of CovMat.





