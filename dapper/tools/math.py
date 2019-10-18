# Misc math

from dapper import *


########################
# Array manip
########################

def is1d(a):
    """ Works for list and row/column arrays and matrices"""
    return np.sum(asarray(asarray(a).shape) > 1) <= 1

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
    args = [np.atleast_1d(x) for x in args]
    return np.concatenate(args,axis=axis)

def roll_n_sub(arr,item,i_repl=0):
    """
    Example::

      >>> roll_n_sub(arange(4),99,0)
      array([99,  0,  1,  2])
      >>> roll_n_sub(arange(4),99,-1)
      array([ 1,  2,  3, 99])
    """
    shift       = i_repl if i_repl<0 else (i_repl+1)
    arr         = np.roll(arr,shift,axis=0)
    arr[i_repl] = item
    return arr


########################
# Ensemble matrix manip
########################

def ens_compatible(func):
    """Tranpose before and after.

    Helpful to make functions compatible with both 1d and 2d ndarrays.

    An older version also used np.atleast_2d and squeeze(),
    but that is more messy than necessary.

    Note: this is not the_way™ -- other tricks are sometimes more practical.
    See for example core.py:dxdt() of LorenzUV, Lorenz96, LotkaVolterra.
    """
    @functools.wraps(func)
    def wrapr(x,*args,**kwargs):
        return func(x.T,*args,**kwargs).T
    return wrapr

def center(E,axis=0,rescale=False):
    """Center ensemble.

    Makes use of np features: keepdims and broadcasting.

    - rescale: Inflate to compensate for reduction in the expected variance."""
    x = mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N  = E.shape[axis]
        X *= sqrt(N/(N-1))

    x = x.squeeze()

    return X, x

def mean0(E,axis=0,rescale=True):
    "Same as: center(E,rescale=True)[0]"
    return center(E,axis=axis,rescale=rescale)[0]

def inflate_ens(E,factor):
    if factor==1: return E
    X, x = center(E)
    return x + X*factor

def weight_degeneracy(w,prec=1e-10):
    return (1-w.max()) < prec

def unbias_var(w=None,N_eff=None,avoid_pathological=False):
    """Compute unbias-ing factor for variance estimation.

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


from IPython.lib.pretty import pretty as pretty_repr
def with_rk4(dxdt,autonom=False,order=4):
    """Wrap dxdt in rk4"""
    integrator       = functools.partial(rk4,order=order)
    if autonom: step = lambda x0,t0,dt: integrator(lambda t,x: dxdt(x),x0,np.nan,dt)
    else:       step = lambda x0,t0,dt: integrator(            dxdt   ,x0,t0    ,dt)
    name = "rk"+str(order)+" integration of "+pretty_repr(dxdt)
    step = NamedFunc(step,name)
    return step

def with_recursion(func,prog=False):
    """Make function recursive in its 1st arg.

    Return a version of func() whose 2nd argument (k)
    specifies the number of times to times apply func on its output.

    Example::

      def step(x,t,dt): ...
      step_k = with_recursion(step)
      x[k]   = step_k(x0,k,t=NaN,dt)[-1]
    """
    def fun_k(x0,k,*args,**kwargs):
        xx    = zeros((k+1,)+x0.shape)
        xx[0] = x0
        rg    = range(k)

        if isinstance(prog,str): rg = progbar(rg,prog)
        elif prog:               rg = progbar(rg,'Recurs.')

        for i in rg:
            xx[i+1] = func(xx[i],*args,**kwargs)
        return xx

    return fun_k

def integrate_TLM(TLM,dt,method='approx'):
    """Compute the resolvent.

    The resolvent may also be called

     - the Jacobian of the step func.
     - the integral of dU/dt = TLM@U, with U0=eye.

    .. note:: the TLM is assumed constant.

    method:

     - 'analytic': exact (assuming TLM is constant).
     - 'approx'  : derived from the forward-euler scheme.
     - 'rk4'     : higher-precision approx (still assumes TLM constant).

    .. caution:: 'analytic' typically requries higher inflation in the ExtKF.

    .. seealso:: FD_Jac.
    """
    if method == 'analytic':
        Lambda,V  = np.linalg.eig(TLM)
        resolvent = (V * exp(dt*Lambda)) @ np.linalg.inv(V)
        resolvent = np.real_if_close(resolvent, tol=10**5)
    else:
        I = eye(TLM.shape[0])
        if method == 'rk4':
            resolvent = rk4(lambda t,U: TLM@U, I, np.nan, dt)
        elif method.lower().startswith('approx'):
            resolvent = I + dt*TLM
        else:
            raise ValueError
    return resolvent

def FD_Jac(ens_compatible_function,eps=1e-7):
    """Finite-diff approx. for functions compatible with 1D and 2D input.

    Example:
    >>> dstep_dx = FD_Jac(step)
    """
    def Jacf(x,*args,**kwargs):
        def f(xx): return ens_compatible_function(xx, *args, **kwargs)
        E  = x + eps*eye(len(x)) # row-oriented ensemble
        FT = (f(E) - f(x))/eps   # => correct broadcasting
        return FT.T              # => Jac[i,j] = df_i/dx_j
    return Jacf

# Transpose explanation:
# - Let F[i,j] = df_i/dx_j be the Jacobian matrix such that
#               f(A)-f(x) ≈ F @ (A-x) 
#   for a matrix A whose columns are realizations. Then
#                       F ≈ [f(A)-f(x)] @ inv(A-x)   [eq1]
# - But, to facilitate broadcasting,
#   DAPPER works with row-oriented (i.e. "ens_compatible" functions),
#   meaning that f should be called as f(A').
#   Transposing [eq1] yields:
#        F' = inv(A-x)'  @ [f(A)  - f(x)]'
#           = inv(A'-x') @ [f(A') - f(x')]
#           =      1/eps * [f(A') - f(x')]
#           =              [f(A') - f(x')] / eps     [eq2]
# => Need to compute [eq2] and then transpose.



########################
# Rounding
########################

def round2(num,prec=1.0):
    """Round with specific precision.

    Returns int if prec is int.

    Also see: builtin round with decimals kwarg.
    """
    return np.multiply(prec,np.rint(np.divide(num,prec)))

def round2sigfig(x,nfig=1):
    """Round to ``nfig`` number of *significant* figures
    (i.e. not the same as builtin round)."""
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

def is_whole(x):
    return np.isclose(x,round(x))

# stackoverflow.com/q/37726830
def is_int(a):
    return np.issubdtype(type(a), np.integer)

def validate_int(x):
    assert is_whole(x)
    return round(x) # convert to int



########################
# Misc
########################

def equi_spaced_integers(Nx,Ny):
    """Provide a range of Ny equispaced integers between 0 and Nx-1"""
    return np.round(linspace(floor(Nx/Ny/2),ceil(Nx-Nx/Ny/2-1),Ny)).astype(int)

def linspace_int(Nx,Ny,periodic=True):
    """Provide a range of Ny equispaced integers between 0 and Nx-1"""
    if periodic: jj = linspace(0, Nx, Ny+1)[:-1]
    else:        jj = linspace(0, Nx-1, Ny)
    jj = jj.astype(int)
    return jj

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
    M    = len(C)
    #cols = np.flipud(sla.circulant(arange(M)[::-1]))
    cols = sla.circulant(arange(M))
    ACF  = zeros(M)
    for i in range(M):
        row = C[i,cols[i]]
        if do_abs:
            row = abs(row)
        ACF += row
        # Note: this actually also accesses masked values in C.
    return ACF/M


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
    """Truncated svd.

    Also automates 'full_matrices' flag.

    - threshold:

      - if float, < 1.0 then "rank" = lowest number
        such that the "energy" retained >= threshold
      - if int,  >= 1   then "rank" = threshold

    - avoid_pathological: avoid truncating (e.g.) the identity matrix.
      NB: only applies for float threshold.
    """
    M,N = A.shape
    full_matrices = False

    if is_int(threshold):
        # Assume specific number is requested
        r = threshold
        assert 1 <= r <= max(M,N)
        if r > min(M,N):
            full_matrices = True
            r = min(M,N)

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
    """Similar to Matlab's svd(A,0).

    Compute the 

     - full    svd if nrows > ncols
     - reduced svd otherwise.

    As in Matlab: svd(A,0),
    except that the input and output are transposed, in keeping with DAPPER convention.
    It contrasts with scipy.linalg's svd(full_matrice=False) and Matlab's svd(A,'econ'),
    both of which always compute the reduced svd.

    .. seealso:: tsvd() for rank (and threshold) truncation.
    """
    M,N = A.shape
    if M>N: return sla.svd(A, full_matrices=True)
    else:   return sla.svd(A, full_matrices=False)

def pad0(ss,N):
    out = zeros(N)
    out[:len(ss)] = ss
    return out

def reconst(U,s,VT):
    """Reconstruct matrix from svd. Supports truncated svd's.

    Example::

      A == reconst(*tsvd(A,1.0)).

    .. seealso:: sla.diagsvd().
    """
    return (U * s) @ VT

def tinv(A,*kargs,**kwargs):
    """
    Inverse based on truncated svd.
    Also see sla.pinv2().
    """
    U,s,VT = tsvd(A,*kargs,**kwargs)
    return (VT.T * s**(-1.0)) @ U.T

def trank(A,*kargs,**kwargs):
    """Rank following truncation"""
    return len(tsvd(A,*kargs,**kwargs)[1])


########################
# HMM setup shortcuts 
########################

def Id_op():
    return NamedFunc(lambda *args: args[0], "Id operator")

def Id_mat(M):
    I = np.eye(M)
    return NamedFunc(lambda x,t: I, "Id("+str(M)+") matrix")

def linear_model_setup(ModelMatrix,dt0):
    r"""Make a dictionary the Dyn/Obs field of HMM representing a linear model.

    .. math::

      x(t+dt) = \texttt{ModelMatrix}^{dt/dt0} x(t),

    i.e. 

    .. math::

      \frac{dx}{dt} = \frac{\log(\texttt{ModelMatrix})}{dt0} x(t).

    In typical use, ``dt0==dt`` (where ``dt`` is defined by the chronology).
    Anyways, ``dt`` must be an integer multiple of ``dt0``.
    """


    Mat = np.asarray(ModelMatrix) # does not support sparse and matrix-class

    # Compute and cache ModelMatrix^(dt/dt0).
    @functools.lru_cache(maxsize=1)
    def MatPow(dt):
        assert is_whole(dt/dt0), "Mat. exponentiation unique only for integer powers."
        return nla.matrix_power(Mat, int(round(dt/dt0)))

    @ens_compatible
    def model(x,t,dt): return MatPow(dt) @ x
    def linear(x,t,dt): return MatPow(dt)

    Dyn = {
        'M'    : len(Mat),
        'model': model,
        'linear': linear,
    }
    return Dyn


def direct_obs_matrix(Nx,obs_inds):
    """Matrix that "picks" state elements obs_inds out of range(Nx)"""
    Ny = len(obs_inds)
    H = zeros((Ny,Nx))
    H[range(Ny),obs_inds] = 1

    # One-liner:
    # H = array([[i==j for i in range(M)] for j in jj],float)

    return H


def partial_Id_Obs(Nx,obs_inds):
    Ny = len(obs_inds)
    H = direct_obs_matrix(Nx,obs_inds)
    @ens_compatible
    def model(x,t): return x[obs_inds]
    def linear(x,t): return H
    Obs = {
        'M'    : Ny,
        'model': model,
        'linear': linear,
    }
    return Obs

def Id_Obs(Nx):
    return partial_Id_Obs(Nx,np.arange(Nx))



