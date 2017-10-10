# Misc math

from common import *


########################
# Array manip
########################

def is1d(a):
  """ Works for list and row/column arrays and matrices"""
  return np.sum(asarray(asarray(a).shape) > 1) <= 1

# stackoverflow.com/q/37726830
def is_int(a):
  return np.issubdtype(type(a), np.integer)

def tp(a):
  """Tranpose 1d vector"""
  return a[np.newaxis].T

def exactly_1d(a):
  a = np.atleast_1d(a)
  assert a.ndim==1
  return a
def exactly_2d(a):
  a = np.atleast_2d(a)
  assert a.ndim==2
  return a

def ccat(*args,axis=0):
  return np.concatenate(args,axis=axis)

def roll_n_sub(arr,item,i_repl=0):
  """
  Example:
  In:  roll_n_sub(arange(4),99,0)
  Out: array([99,  0,  1,  2])
  In:  roll_n_sub(arange(4),99,-1)
  Out: array([ 1,  2,  3, 99])
  """
  shift       = i_repl if i_repl<0 else (i_repl+1)
  arr         = np.roll(arr,shift,axis=0)
  arr[i_repl] = item
  return arr
        

########################
# Ensemble matrix manip
########################

def ens_compatible(func):
  """Tranpose before and after."""
  @functools.wraps(func)
  def wrapr(x,*kargs,**kwargs):
    return func(x.T,*kargs,**kwargs).T
  return wrapr

def anom(E,axis=0):
  mu = mean(E,axis=axis, keepdims=True)
  A  = E - mu
  return A, mu.squeeze()

def center(E,rescale=True):
  """
  Center sample,
  but rescale to maintain its (expected) variance.

  Note: similarly, one could correct a sample's 2nd moment,
        (on the diagonal, or other some other subset),
        however this is typically not worth it.
  """
  N = E.shape[0]
  A = E - mean(E,0)
  if rescale:
    A *= sqrt(N/(N-1))
  return A

def inflate_ens(E,factor):
  A, mu = anom(E)
  return mu + A*factor

def weight_degeneracy(w,prec=1e-10):
  return (1-w.max()) < prec

def unbias_var(w=None,N_eff=None,avoid_pathological=False):
  """
  Compute unbias-ing factor for variance estimation.
  wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
  """
  if N_eff is None:
    N_eff = 1/(w@w)
  if avoid_pathological and weight_degeneracy(w):
    ub = 1 # Don't do in case of weights collapse
  else:
    ub = 1/(1 - 1/N_eff) # =N/(N-1) if w==ones(N)/N.
  return ub


########################
# Time stepping (integration)
########################

def rk4(f, x, t, dt, order=4):
  """Runge-Kutta N-th order (explicit, non-adaptive) numerical ODE solvers.""" 
  if order >=1: k1 = dt * f(t     , x)
  if order >=2: k2 = dt * f(t+dt/2, x+k1/2)
  if order ==3: k3 = dt * f(t+dt  , x+k2*2-k1)
  if order ==4:
                k3 = dt * f(t+dt/2, x+k2/2)
                k4 = dt * f(t+dt  , x+k3)
  if    order ==1: return x + k1
  elif  order ==2: return x + k2
  elif  order ==3: return x + (k1 + 4*k2 + k3)/6
  elif  order ==4: return x + (k1 + 2*(k2 + k3) + k4)/6
  else: raise NotImplementedError


def with_rk4(dxdt,autonom=False,order=4):
  """Wrap dxdt in rk4"""
  integrator       = functools.partial(rk4,order=order)
  if autonom: step = lambda x0,t0,dt: integrator(lambda t,x: dxdt(x),x0,np.nan,dt)
  else:       step = lambda x0,t0,dt: integrator(            dxdt   ,x0,t0    ,dt)
  name = "rk"+str(order)+" integration of "+repr(dxdt)+" from "+dxdt.__module__
  step = NamedFunc(step,name)
  return step

def make_recursive(func,with_prog=False):
  """
  Return a version of func() whose 2nd argument (k)
  is the number of times to times apply func on its output.
  Example:
    def step(x,t,dt): ...
    step_k = make_recursive(step)
    x[k]   = step_k(x0,k,t=NaN,dt)[-1]
  """
  def fun_k(x0,k,*args,**kwargs):
    xx    = zeros((k+1,)+x0.shape)
    xx[0] = x0
    rg    = range(k)
    rg    = progbar(rg,'Recurs.') if with_prog else rg
    for i in rg:
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
    

########################
# Rounding
########################

def round2(num,prec=1.0):
  """Round with specific precision.
  Returns int if prec is int."""
  return np.multiply(prec,np.rint(np.divide(num,prec)))

def round2sigfig(x,nfig=1):
  if np.all(array(x) == 0):
    return x
  signs = np.sign(x)
  x *= signs
  return signs*round2(x,10**floor(log10(x)-nfig+1))

def round2nice(xx):
  "Rounds (ordered) array to nice numbers"
  r1 = round2sigfig(xx,nfig=1)
  r2 = round2sigfig(xx,nfig=2)
  # Assign r2 to duplicate entries in r1:
  dup = np.isclose(0,np.diff(r1))
  r1[1:-1][dup[:-1]] = r2[1:-1][dup[:-1]]
  if dup[-1]:
    r1[-2] = r2[-2]
  return r1

def validate_int(x):
  x_int = int(x)
  assert np.isclose(x,x_int)
  return x_int


########################
# Misc
########################

def find_1st_ind(xx):
  try:
    return next(k for k in range(len(xx)) if xx[k])
  except StopIteration:
    return None

def LogSp(start,stop,num=50,**kwargs):
  """Log space defined through non-log numbers"""
  assert 'base' not in kwargs, "The base is irrelevant."
  return np.logspace(log10(start),log10(stop),num=num,base=10)

def CurvedSpace(start,end,curve,N):
  "Monotonic series (space). Set 'curve' param between 0,1."  
  x0 = 1/curve - 1
  span  = end - start
  return start + span*( LogSp(x0,1+x0,N) - x0 )

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


########################
# Linear Algebra
########################

def mrdiv(b,A):
  return nla.solve(A.T,b.T).T

def mldiv(A,b):
  return nla.solve(A,b)


def truncate_rank(s,threshold,avoid_pathological):
  "Find r such that s[:r] contains the threshold proportion of s."
  assert isinstance(threshold,float)
  if threshold == 1.0:
    r = len(s)
  elif threshold < 1.0:
    r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
    r += 1 # Hence the strict inequality above
    if avoid_pathological:
      # If not avoid_pathological, then the last 4 diag. entries of
      # reconst( *tsvd(eye(400),0.99) )
      # will be zero. This is probably not intended.
      r += np.sum(np.isclose(s[r-1], s[r:]))
  else:
    raise ValueError
  return r

def tsvd(A, threshold=0.99999, avoid_pathological=True):
  """
  Truncated svd.
  Also automates 'full_matrices' flag.
  threshold: if
   - float, < 1.0 then "rank" = lowest number such that the
                                "energy" retained >= threshold
   - int,  >= 1   then "rank" = threshold
  avoid_pathological: avoid truncating (e.g.) the identity matrix.
                      NB: only applies for float threshold.
  """

  m,n = A.shape
  full_matrices = False

  if is_int(threshold):
    # Assume specific number is requested
    r = threshold
    assert 1 <= r <= max(m,n)
    if r > min(m,n):
      full_matrices = True

  # SVD
  U,s,VT = sla.svd(A, full_matrices)

  if isinstance(threshold,float):
    # Assume proportion is requested
    r = truncate_rank(s,threshold,avoid_pathological)

  # Truncate
  U  = U [:,:r]
  VT = VT[  :r]
  s  = s [  :r]
  return U,s,VT

def svd0(A):
  """
  Compute the 
   - full    svd if nrows > ncols
   - reduced svd otherwise.
  As in Matlab: svd(A,0),
  except that the input and output are transposed, in keeping with DAPPER convention.
  It contrasts with scipy.linalg's svd(full_matrice=False) and Matlab's svd(A,'econ'),
  both of which always compute the reduced svd.
  For reduction down to rank, see tsvd() instead.
  """
  m,n = A.shape
  if m>n: return sla.svd(A, full_matrices=True)
  else:   return sla.svd(A, full_matrices=False)

def pad0(ss,N):
  out = zeros(N)
  out[:len(ss)] = ss
  return out
  
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


########################
# Setup facilation
########################

def Id_op():
  return NamedFunc(lambda *args: args[0], "Id operator")
def Id_mat(m):
  I = np.eye(m)
  return NamedFunc(lambda x,t: I, "Id("+str(m)+") matrix")

def linear_model_setup(M):
  "M is normalized wrt step length dt."
  M = np.asarray(M) # sparse or matrix classes not supported
  m = len(M)
  @ens_compatible
  def model(x,t,dt): return dt*(M@x)
  def jacob(x,t,dt): return dt*M
  f = {
      'm'    : m,
      'model': model,
      'jacob': jacob,
      }
  return f



def equi_spaced_integers(m,p):
  """Provide a range of p equispaced integers between 0 and m-1"""
  return np.round(linspace(floor(m/p/2),ceil(m-m/p/2-1),p)).astype(int)

def direct_obs_matrix(m,jj):
  """Matrix that "picks" state elements jj out of range(m)"""
  p = len(jj)
  H = zeros((p,m))
  H[range(p),jj] = 1
  return H

def partial_direct_obs_setup(m,jj):
  p  = len(jj)
  H  = direct_obs_matrix(m,jj)
  @ens_compatible
  def model(x,t): return x[jj]
  def jacob(x,t): return H
  def plot(y):    return plt.plot(jj,y,'g*',ms=8)[0]
  h = {
      'm'    : p,
      'model': model,
      'jacob': jacob,
      'plot' : plot,
      }
  return h


