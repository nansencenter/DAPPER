# Misc math

from common import *

def is1d(a):
  """ Works for list and row/column arrays and matrices"""
  return sum(asarray(asarray(a).shape) > 1) <= 1

def tp(a):
  """Tranpose 1d vector"""
  return a[np.newaxis].T

# TODO: review useage. Replace by ens_compatible()?
def atmost_2d(func):
  """
  Decorator to make functions that work on 2-dim input
  work for (all of) 0,1, or 2-dim input.
  Requires that the func's 1st argument be the one in question.
  Does not always work (e.g. recursion). Use with caution.
  """
  def wrapr(x,*kargs,**kwargs):
    answer = func(np.atleast_2d(x),*kargs,**kwargs)
    if answer is not None: return answer.squeeze()
  return wrapr

def ens_compatible(func):
  """Tranpose before and after."""
  def wrapr(x,*kargs,**kwargs):
    return func(x.T,*kargs,**kwargs).T
  return wrapr

        
def anom(E,axis=0):
  mu = mean(E,axis=axis, keepdims=True)
  A  = E - mu
  return A, mu.squeeze()

# Center sample (but maintain its (expected) variance)
def center(E,rescale=True):
  N = E.shape[0]
  A = E - mean(E,0)
  if rescale:
    A *= sqrt(N/(N-1))
  return A

def inflate_ens(E,factor):
  A, mu = anom(E)
  return mu + A*factor

def mrdiv(b,A):
  return nla.solve(A.T,b.T).T

def mldiv(A,b):
  return nla.solve(A,b)



def rk4(f, x0, t, dt):
  """Runge-Kutta 4-th order (approximate ODE solver)."""
  k1 = dt * f(t      , x0)
  k2 = dt * f(t+dt/2., x0+k1/2.)
  k3 = dt * f(t+dt/2., x0+k2/2.)
  k4 = dt * f(t+dt   , x0+k3)
  return x0 + (k1 + 2.*(k2 + k3) + k4)/6.0

def make_recursive(func):
  """
  Return a version of func() whose 2nd argument (k)
  is the number of times to times apply func on its output.
  Example:
    def step(x,t,dt): ...
    step_k = make_recursive(step)
    x[k]   = step_k(x0,k,t=NaN,dt)[-1]
  """
  def fun_k(x0,k,*args,**kwargs):
    xx    = zeros((k+1,len(x0)))
    xx[0] = x0
    for i in range(k):
      xx[i+1] = func(xx[i],*args,**kwargs)
    return xx
  return fun_k

def integrate_TLM(M,dt,method='approx'):
  """
  Returns the resolvent, i.e. (equivalently)
   - the Jacobian of the step func.
   - the integral of dU/dt = M@U, with U0=eye.
  Note that M (the TLM) is assumed constant.

  method:
   - 'analytic': exact (assuming TLM is constant).
   - 'approx'  : derived from the forward-euler scheme.
   - 'rk4'     : higher-precision approx.
  NB: 'analytic' typically requries higher inflation in the ExtKF.
  """
  if method == 'analytic':
    Lambda,V  = np.linalg.eig(M)
    resolvent = (V * exp(dt*Lambda)) @ np.linalg.inv(V)
    resolvent = np.real_if_close(resolvent, tol=10**5)
  else:
    I = eye(M.shape[0])
    if method == 'rk4':
      resolvent = rk4(lambda t,U: M@U, I, np.nan, dt)
    elif method.lower().startswith('approx'):
      resolvent = I + dt*M
    else:
      raise ValueError
  return resolvent
    



def round2(num,prec=1.0):
  """Round with specific precision.
  Returns int if prec is int."""
  return np.multiply(prec,np.rint(np.divide(num,prec)))

def round2sigfig(x,nfig=1):
  if x == 0:
    return x
  signs = np.sign(x)
  x *= signs
  return signs*round2(x,10**floor(log10(x)-nfig+1))

def validate_int(x):
  x_int = int(x)
  assert np.isclose(x,x_int)
  return x_int

def find_1st_ind(xx):
  try:
    return next(k for k in range(len(xx)) if xx[k])
  except StopIteration:
    return None

def equi_spaced_integers(m,p):
  """Provide a range of p equispaced integers between 0 and m-1"""
  return np.round(linspace(floor(m/p/2),ceil(m-m/p/2-1),p)).astype(int)

def circulant_ACF(C,do_abs=False):
  """
  Compute the ACF of C,
  assuming it is the cov/corr matrix
  of a 1D periodic domain.
  """
  m    = len(C)
  #cols = np.flipud(sla.circulant(arange(m)[::-1]))
  cols = sla.circulant(arange(m))
  ACF  = zeros(m)
  for i in range(m):
    row = C[i,cols[i]]
    if do_abs:
      row = abs(row)
    ACF += row
    # Note: this actually also accesses masked values in C.
  return ACF/m




def pad0(ss,N):
  out = zeros(N)
  out[:len(ss)] = ss
  return out

def svd0(A):
  """
  Compute the 
   - full    svd if nrows > ncols
   - reduced svd otherwise.
  This is the reverse of Matlab's svd(A,0),
  in keeping with DAPPER convention of transposing ensemble matrices.
  It also contrasts with scipy.linalg's svd and Matlab's svd(A,'econ'),
  both of which always compute the reduced svd.
  """
  m,n = A.shape
  if m>n:
    return sla.svd(A, full_matrices=True)
  else:
    return sla.svd(A, full_matrices=False)


def tsvd(A, threshold=0.99999, avoid_pathological=True):
  """
  Truncated svd.
  Also automates flag: full_matrices.
  threshold: if
   - float, < 1.0 then "rank" = lowest number such that the
                                "energy" retained >= threshold
   - int,  >= 1   then "rank" = threshold
  avoid_pathological: avoid truncating (e.g.) the identity matrix.
                      NB: only applies for float threshold.
  """

  m,n = A.shape
  full_matrices = False

  # Assume number of components requested
  if isinstance(threshold,int):
    assert threshold >= 1
    r = threshold
    assert r <= max((m,n))
    if r > min((m,n)):
      full_matrices = True
    avoid_pathological = False

  # SVD
  U,s,VT = sla.svd(A, full_matrices)

  # Assume proportion requested
  if isinstance(threshold,float):
    assert threshold <= 1.0
    if threshold < 1.0:
      r = sum(np.cumsum(s)/sum(s) < threshold)
      r += 1 # Hence the strict inequality above
      if avoid_pathological:
        # If not avoid_pathological, then the last 4 diag. entries of
        # reconst( *tsvd(eye(400),0.99) )
        # will be zero. This is probably not intended.
        r += sum(np.isclose(s[r-1], s[r:]))
    else:
      r = len(s)

  # Truncate
  U  = U [:,:r]
  VT = VT[  :r]
  s  = s [  :r]
  return U,s,VT
  
def reconst(U,s,VT):
  """
  Reconstruct matrix from svd. Supports truncated svd's.
  A == reconst(*tsvd(A,1.0)).
  Also see: sla.diagsvd().
  """
  return (U * s) @ VT

def tinv(A,*kargs,**kwargs):
  """
  Inverse based on truncated svd.
  Also see sla.pinv2().
  """
  U,s,VT = tsvd(A,*kargs,**kwargs)
  return (VT.T * s**(-1.0)) @ U.T

