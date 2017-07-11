from common import *

# Test matrices
def randcov(m):
  """(Makeshift) random cov mat."""
  N = int(ceil(2+m**1.2))
  E = randn((N,m))
  return E.T @ E
def randcorr(m):
  """(Makeshift) random corr mat."""
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
  elif isinstance(opts,bool) or opts is 1:
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
    # Decompose and reduce angle of (complex) diagonal. Background:
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


def chol_reduce(Right):
  """
  Return rnk-by-ndim R such that Right.T@Right - R.T@R ≈ 0.
  Example:
    A = anom(randn((20,5)),axis=1)
    C = A.T @ A
    # sla.cholesky(C) throws error
    R = chol_reduce(A)
    R.shape[1] == 4
  """
  _,sig,UT = sla.svd(Right,full_matrices=False)
  R = sig[:,None]*UT

  # The below is DEPRECATED, coz it fails e.g. with Q from mods.LA.raanes2015.
  #from scipy.linalg.lapack import get_lapack_funcs
  #potrf, = get_lapack_funcs(('potrf',), (C,))
  #R, info = potrf(C, lower=False, overwrite_a=False, clean=True)
  #if info!=0:
    #R = R[:info]
  # Explanation: R is truncated when cholesky() finds a 'leading negative minor'.
  # Thus, R is rectangular, with height ∈ [rank, m].

  return R


class CovMat(object):
  """
  Covariance matrix class.

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
    """
    The covariance (say P) can be input (specified in the following ways):
    kind    : data
    ----------------------
    'full'  : full m-by-m array (P)
    'diag'  : diagonal of P (assumed diagonal)
    'E'     : ensemble (N-by-m) with sample cov P
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
        d         = exactly_1d(data)
        self.diag = d
        m         = len(d)
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

    self._kind  = kind
    self._trunc = trunc


  ##################################
  # Protected
  ##################################
  @property
  def m(self):
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

  """
  #Hazardous
  @trunc.setter
  def trunc(self,value):
    print('set')
    return print(CovMat(data=self.full,trunc=value))
  """
  
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
    and that its width is somewhere betwen the rank and m."""
    if hasattr(self,'_R'):
      return self._R.T
    else:
      return self.V * sqrt(self.ews)
  @property
  def Right(self):
    """R such that C = R.T@R. Note that R is typically rectangular, but not triangular,
    and that its height is somewhere betwen the rank and m."""
    if hasattr(self,'_R'):
      return self._R
    else:
      return self.Left.T


  ##################################
  # EVD stuff
  ##################################
  def _assign_EVD(self,m,rk,d,V):
      self._m   = m
      self._d   = d
      self._V   = V
      self._rk  = rk

  @staticmethod
  def _clip(d):
    return np.where(d<1e-8*d.max(),0,d)

  def _do_EVD(self):
    if not self.has_done_EVD():
      V,s,UT = svd0(self._R)
      m      = UT.shape[1]
      d      = s**2
      d      = CovMat._clip(d)
      rk     = (d>0).sum()
      d      = d [:rk]
      V      = UT[:rk].T
      self._assign_EVD(m,rk,d,V)

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


  def transform_by(self,fun=lambda x:x):
    """
    Generalize scalar functions to covariance matrices
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
    if self.m != self.rk:
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
    s  = "\n    m: " + str (self.m)
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

    # Indent all of cov array, and add to s
    s += t.replace("\n","\n   ")

    # Add diag. Indent array +1 vs cov array
    with printoptions(threshold=0):
      s += "\n diag:\n   " + " " + str(self.diag)

    s = repr_type_and_name(self) + s.replace("\n","\n  ")
    return s



"""
class VarMat(CovMat):

  def __init__(self,data):

    stds=std(data,axis=0)

    new_dat=diag(stds)

    CovMat.__init__(self,data=new_dat,kind='full_or_diag',trunc=1.0)
"""

class CorrMat(CovMat):

  def __init__(self,data,thin=1.0,Rinfl=1.0):

    #Filter elements of the matrix
    data=np.where(data>1.0e-6,data,0)
    CovMat.__init__(self,data=data,kind='full_or_diag',trunc=1.0)
    
    if type(thin)==int or float(thin)<0.999:
      self._thinning(threshold=thin)

    self._args=[t for t in locals().items() if t[0] not in ['self','data']]
    self._thin=thin
    self._Rinfl=Rinfl

    global sizer
    def sizer(f):
      try:
        last_bm_size=Benchmark.instances[-1].setup.h.m
        return functools.partial(f,size=last_bm_size)
      except (IndexError,NameError):
        return f

  def experiment(self,X,Y,setup,config):

    setup.h.noise.C=self

    #timepoint reference
    start_time=time.time()
    #assimilate
    stats=config.assimilate(setup,X,Y)
    #Close timewindow
    delta_t=time.time()-start_time

    #get rmse and spread through time, once BurnIn is done
    rmse_s=stats.rmse.a
    spread_s=mean(stats.mad.a,axis=1)
    umsft_s= mean( abs(stats.umisf.a) ,0)
    umsprd_s=mean(     stats.svals.a  ,0)

    #Get the average rmse and spread
    rmse=mean(rmse_s[setup.t.maskObs_BI])
    spread=mean(spread_s[setup.t.maskObs_BI])
    stats=hstack([rmse_s.reshape((len(rmse_s),-1)),spread_s.reshape((len(spread_s)),-1)])
    ErrComp=hstack([umsft_s.reshape((len(umsft_s),-1)),umsprd_s.reshape(len(umsprd_s),-1)])

    output=[('RMSE',rmse),('Spread',spread),('CPU_Time',delta_t),('Stats_TS',stats),('ErrComp',ErrComp)]+self._args
    return output    

  def __str__(self):
    #self._args+=[()]
    a=[i[0] for i in self._args]
    b=[str(i[1]) for i in self._args]
    for i in range(len(a)):
      a[i]=str(a[i])+' '*(max(len(a[i]),len(b[i]))-len(a[i]))
      b[i]=str(b[i])+' '*(max(len(a[i]),len(b[i]))-len(b[i]))
    cols=' | '.join(a)
    inter='-'*len(cols)
    row=' | '.join(b)
    return cols+'\n'+inter+'\n'+row
    
    #return str(self._args)+'---'+str(self._values)
    return str(self._args)

  def __mul__(self,arr):
    data=self.full@arr
    n=self.__class__.__bases__[0](data=data)
    n._args+=[('kind',dict(self._args)['kind'])]
    return n

  ##################################
  ######## Thinning method #########
  ##################################

  """
  This thinning method has been implemented as suggested by Stewart '08 and is therefore rank conservative (keeps PD a PD matrix)
  Was a pain to:
    -make it user friendly (turn it into a easy-to-use way for the beginner user)
    -ensure this rank conservation

  Is broadcasting a row to muliply it with a matrix faster than turning it into a diagonal matrix and usign the built-in matmul ??

  """


  #Hidden method ! As it works in-place, it should never be accessed in the __main__.
  #it messes up the arguments of the matrix if used so.
  def _thinning(self,threshold=1.0):

    r = truncate_rank(self.ews,threshold,True)

    #This operation is in line with Stewart08 (double checked):
    V=self.V[:,:r]
    w=self.ews[:r]
    #recomputing alpha here is very close to be a doublon with the computations done in truncate_rank but not exactly:
    #if the threshold of truncate_rank is 0.8, it does not mean that the span is reduced to exactly 80%: it means than we chose
    #as much eigenvectors as possible without going further than 80% of the spectrum representation, because we have to select
    #a round number of eigenvectors (obviously).
    
    #alpha is used to re-adjust the span of the whole correlation (therefore )
    alpha=self.ews[r:].sum()/(self.m-r+1)

    w-=alpha

    #now adjust the substracted span to the final result by scaling a identity matrix
    data=(V * w) @ V.T + alpha*eye(self.m)
    
    #It now re-creates (initializes) a CorrMat object (with thin=1.0 (as default) to avoid endless looping)
    #In the end, I still call __init__ twice and it is therefore syntactic sugar between this and calling mother.__init__
    #twice in the child.__init__ but I find it more readable this way.
    #It is overall the price to pay to re-use Patrick's transform_by method
    CorrMat.__init__(self,data=data)


  ################################################
  #Protect the arguments used to build the matrix
  #If needed other arguments, build another matrix
  ################################################

  @property
  def arguments(self):
    return self._args

  #Kinda useless because thin is already accessible and protected in the arguments list above. 
  #Still, it remains more understandable for the reader this way, as it emphasizes that thin
  #is used when building the matrix, it therefore does not make any sense to modify it once the matrix built.
  @property
  def thin(self):
    return self._thin

  @property
  def Rinfl(self):
    return self._Rinfl

  """
  This method was supposed to avoir the need to precise the size of the added matrix to any benchmark but I failed facing several issues:
  Can not access the size of the last benchmark in the body of a function as the globals variable will not be reloaded at each call but only at the
  declaration of the function.
  This was anyway highly uneffecient and heavy/dirty.

  I therefore have to stick to the old version.
  
  @staticmethod
  def sizer(f):
    try:
      return functools.partial(f,size=last_bm_size)
    except (IndexError,NameError):
      return f
  """

class MARKOV(CorrMat):
  def __init__(self,size,deltax=1,Lr=1,thin=0.999999):
    A=zeros((size,size))
    for (i,j) in product(range(size),repeat=2):
      A[i,j]=exp(-abs(i-j)*deltax/Lr)
    CorrMat.__init__(self,data=A,thin=thin)
    del(i,j,A,size)
    l=[t for t in locals().items() if t[0] not in ['self']]
    l.append(('kind','MARKOV'))
    self._args=l



class SOAR(CorrMat):
  
  def __init__(self,size,deltax=1,Lr=1,thin=0.999999):

    A=zeros((size,size))
    for (i,j) in product(range(size),repeat=2):
      A[i,j]=exp(-abs(i-j)*deltax/Lr)*(1+abs(i-j)*deltax/Lr)
    CorrMat.__init__(self,data=A,thin=thin)
    del(i,j,A,size)
    l=[t for t in locals().items() if t[0] not in ['self']]
    l.append(('kind','SOAR'))
    self._args=l

class MultiDiag(CorrMat):

  def __init__(self,size,diags=1,decay=0,thin=0.999999,Rinfl=1.0):
    
    if diags%2==0:
      raise ValueError('Even number of diagonals lead to non-symmetrical correlation matrix')
    
    decay=max(decay,1)

    A=eye(size)

    for k in range(1,diags//2+1):
      A+=(eye(size,k=k)+eye(size,k=-k))/decay**k

    A*= Rinfl if diags==1 else 1.0

    CorrMat.__init__(self,data=A,thin=thin)

    del(size,A)
    if diags>1:
      del(k)
    l=[t for t in locals().items() if t[0] not in ['self']]
    l.append(('kind','MultiDiag'))
    self._args=l

class Custom(CorrMat):

  def __init__(self,size,thin=0.999999,f=lambda x,y:(x==y)*1):

    A=zeros((size,size))
    for (i,j) in product(range(size),repeat=2):
      A[i,j]=f(i,j)
    CorrMat.__init__(self,data=A,thin=thin)


    l=[('kind','Custom'),('thinning',thin)]
    self._args=l

class BlockDiag(CorrMat):

  def __init__(self,*args,size=None,submat=None,thin=0.999999):
    #if size%submat.m>0:
    #  raise ValueError('Uncorrect size of the submatrices')
    l=[('kind','BlockDiag'),('thin',thin)]
    if submat==None:
      try:
        types=[a.__class__.__bases__[0] for a in args]
        if any([not str(t)=="<class 'aux.matrices.CorrMat'>" for t in types]):
          raise ValueError('Uncorrect argument(s), must be CorrMat -or derived type')
        else:
          fulls=tuple([m.full for m in args])
          A=block_diag(*fulls)
          l+=[('submat',[str(a.__class__).replace("'",'').replace('>','').split('.')[-1] for a in args])]
      except:
        IndexError('Uncorrect argument(s), must be CorrMat -or derived type')

    else:

      B=submat.full
      q=B.shape[0]
      #t=submatkind
      n=size/q
      assert (n%1==0), 'bad definition -size issue'
      #S=ObsErrMat(kind=t,size=q).matrix
      A=zeros((size,size))
      #Build the matrix
      for k in range(int(n)):
        for (i,j) in product(range(q),repeat=2):
          A[k*q+i,k*q+j]=B[i,j]
      l+=[('submat',re.sub('[^A-Za-z]',' ',str(submat.__class__)).split()[-1]),('Subdiv',int(n))]

    CorrMat.__init__(self,data=A,thin=thin)

    self._args=l
    
class Sparse(CorrMat):

  def __init__(self,size,deltax=1,Lr=1,thin=0.999999):
    
    A=zeros((size,size))
    for (i,j) in product(range(size),repeat=2):
      A[i,j]=exp((abs(abs(i-j)-(size-1)/2)-(size-1)/2)*deltax/Lr)
    CorrMat.__init__(self,data=A,thin=thin)
    del(i,j,A)
    l=[t for t in locals().items() if t[0] not in ['self','size']]
    l.append(('kind','Sparse'))
    self._args=l


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