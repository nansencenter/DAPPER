"""Math tools: integrators, linear algebra, and convenience funcs."""
# pylint: disable=unused-argument

import functools

import numpy as np
import scipy.linalg as sla
from IPython.lib.pretty import pretty as pretty_repr

import dapper.tools.utils as utils


########################
# Array manip
########################
def is1d(a):
    """Works for list and row/column arrays and matrices"""
    return np.sum(np.asarray(np.asarray(a).shape) > 1) <= 1


def exactly_1d(a):
    "Ensure a is 1d."
    a = np.atleast_1d(a)
    assert a.ndim == 1
    return a


def exactly_2d(a):
    "Ensure a is 2d."
    a = np.atleast_2d(a)
    assert a.ndim == 2
    return a


def ccat(*args, axis=0):
    "Convenience wrapper around np.concatenate."
    args = [np.atleast_1d(x) for x in args]
    return np.concatenate(args, axis=axis)


########################
# Ensemble matrix manip
########################
def ens_compatible(func):
    """Tranpose before and after.

    Helpful to make functions compatible with both 1d and 2d ndarrays.

    An older version also used np.atleast_2d and squeeze(),
    but that is more messy than necessary.

    Note: this is not the_way™ -- other tricks are sometimes more practical.
    See for example dxdt() in __init__.py for LorenzUV, Lorenz96, LotkaVolterra.
    """
    @functools.wraps(func)
    def wrapr(x, *args, **kwargs):
        return np.asarray(func(x.T, *args, **kwargs)).T
    return wrapr


def center(E, axis=0, rescale=False):
    """Center ensemble.

    Makes use of np features: keepdims and broadcasting.

    - rescale: Inflate to compensate for reduction in the expected variance."""
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    "Same as: center(E,rescale=True)[0]"
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    "Inflate the ensemble (center, inflate, re-combine)."
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor


def weight_degeneracy(w, prec=1e-10):
    "Are weights degenerate?"
    return (1-w.max()) < prec


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """Compute unbias-ing factor for variance estimation.

    wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    """
    if N_eff is None:
        N_eff = 1/(w@w)
    if avoid_pathological and weight_degeneracy(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1/(1 - 1/N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub


########################
# Time stepping (integration)
########################

# fmt: off
def rk4(f, x, t, dt, order=4):
    """Runge-Kutta N-th order (explicit, non-adaptive) numerical ODE solvers."""
    # pylint: disable=R1705, W0311
    if order >=1: k1 = dt * f(t     , x)                        # noqa
    if order >=2: k2 = dt * f(t+dt/2, x+k1/2)                   # noqa
    if order ==3: k3 = dt * f(t+dt  , x+k2*2-k1)                # noqa
    if order ==4:                                               # noqa
                  k3 = dt * f(t+dt/2, x+k2/2)                   # noqa
                  k4 = dt * f(t+dt  , x+k3)                     # noqa
    if    order ==1: return x + k1                              # noqa
    elif  order ==2: return x + k2                              # noqa
    elif  order ==3: return x + (k1 + 4*k2 + k3)/6              # noqa
    elif  order ==4: return x + (k1 + 2*(k2 + k3) + k4)/6       # noqa
    else: raise NotImplementedError                             # noqa
# fmt: on


def with_rk4(dxdt, autonom=False, order=4):
    """Wrap dxdt in rk4"""
    integrator = functools.partial(rk4, order=order)
    if autonom:
        def step(x0, t0, dt):
            return integrator(lambda t, x: dxdt(x), x0, np.nan, dt)
    else:
        def step(x0, t0, dt):
            return integrator(dxdt, x0, t0, dt)
    name = "rk"+str(order)+" integration of "+pretty_repr(dxdt)
    step = utils.NamedFunc(step, name)
    return step


def with_recursion(func, prog=False):
    """Make function recursive in its 1st arg.

    Return a version of func() whose 2nd argument (k)
    specifies the number of times to times apply func on its output.

    Example::

      def step(x,t,dt): ...
      step_k = with_recursion(step)
      x[k]   = step_k(x0,k,t=np.nan,dt)[-1]
    """
    def fun_k(x0, k, *args, **kwargs):
        xx = np.zeros((k+1,)+x0.shape)
        xx[0] = x0

        rg = range(k)
        if isinstance(prog, str):
            rg = utils.progbar(rg, prog)
        elif prog:
            rg = utils.progbar(rg, 'Recurs.')

        for i in rg:
            xx[i+1] = func(xx[i], *args, **kwargs)

        return xx

    return fun_k


def integrate_TLM(TLM, dt, method='approx'):
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
        Lambda, V = sla.eig(TLM)
        resolvent = (V * np.exp(dt*Lambda)) @ np.linalg.inv(V)
        resolvent = np.real_if_close(resolvent, tol=10**5)
    else:
        Id = np.eye(TLM.shape[0])
        if method == 'rk4':
            resolvent = rk4(lambda t, U: TLM@U, Id, np.nan, dt)
        elif method.lower().startswith('approx'):
            resolvent = Id + dt*TLM
        else:
            raise ValueError
    return resolvent


def FD_Jac(ens_compatible_function, eps=1e-7):
    """Finite-diff approx. for functions compatible with 1D and 2D input.

    Example:
    >>> dstep_dx = FD_Jac(step)
    """
    def Jacf(x, *args, **kwargs):
        def f(xx): return ens_compatible_function(xx, *args, **kwargs)
        E = x + eps*np.eye(len(x))  # row-oriented ensemble
        FT = (f(E) - f(x))/eps      # => correct broadcasting
        return FT.T                 # => Jac[i,j] = df_i/dx_j
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

@np.vectorize
def _round2prec(num, prec):
    """Don't use! Instead, use round2(). The issue is that:
    >>> _round2prec(0.7,.1) != 0.7
    """
    return prec * round(num / prec)


@np.vectorize
def ndecimal(x):
    """Convert precision to num. of decimals. Example:
    >>> ndecimal(10)    # --> -1
    >>> ndecimal(1)     # --> 0
    >>> ndecimal(0.1)   # --> 1
    >>> ndecimal(0.01)  # --> 2
    >>> ndecimal(0.02)  # --> 2
    >>> ndecimal(0.099) # --> 2 # yes, this is what we want
    """
    if x == 0 or not np.isfinite(x):
        # "Behaviour not defined" => should not be relied upon.
        return 1
    return -int(np.floor(np.log10(np.abs(x))))


@np.vectorize
def round2(num, param=1):
    """Round num as specified by ``param``. Always returns floats.

    If ``param`` is int: round to ``param`` num. of *significant* digits.
    Otherwise          : round to ``param`` precison (must be float).
    By contrast, ``round`` (builtin and np) takes the num. of *decimals*.

    Examples:
    >>> xx = curvedspace(1e-3,1e2,15,.5)
    >>> with np.printoptions(precision=100):
    >>>     spell_out(_round2prec (xx, 1e-2))
    >>>     spell_out(round2      (xx, 1e-2))
    >>>     spell_out(round2      (xx,    1))
    """
    if is_int(param):
        # round2sigfig
        nfig = param-1
        n = nfig + ndecimal(num)
    else:
        # round2prec
        prec = param
        n = ndecimal(prec)
        num = _round2prec(num, prec)

    return np.round(num, n)  # n specified => float (always)


# https://stackoverflow.com/q/37726830
def is_int(a):
    return np.issubdtype(type(a), np.integer)


def is_whole(x):
    return np.isclose(x, round(x))


def validate_int(x):
    assert is_whole(x)
    return round(x)  # convert to int


def isNone(x):
    """x==None that also works for x being an np.ndarray.

    Since python3.8 ``x is None`` throws warning.

    Ref: np.isscalar docstring."""
    return np.ndim(x) == 0 and x == None


########################
# Misc
########################
def linspace_int(Nx, Ny, periodic=True):
    """Provide a range of Ny equispaced integers between 0 and Nx-1"""
    if periodic:
        jj = np.linspace(0, Nx, Ny+1)[:-1]
    else:
        jj = np.linspace(0, Nx-1, Ny)
    jj = jj.astype(int)
    return jj


def curvedspace(start, end, N, curvature=1):
    """A length (func. of curvature) of logspace, normlzd to [start,end]

    - curvature== 0: ==> linspace (start,end,N)
    - curvature==+1: ==> geomspace(start,end,N)
    - curvature==-1: ==> idem, but reflected about y=x

    Example:
    >>> ax.plot(np.geomspace(1e-1, 10, 201) ,label="geomspace")
    >>> ax.plot(np.linspace (1e-1, 10, 201) ,label="linspace")
    >>> ax.plot( curvedspace(1e-1, 10, 201, 0.5),'y--')

    Also see:
    - linspace_int
    - np.logspace
    - np.geomspace
    """
    if -1e-12 < curvature < 1e-12:
        # Define curvature-->0, which is troublesome
        # for linear normalization transformation.
        space01 = np.linspace(0, 1, N)
    else:
        curvature = (end/start)**curvature
        space01 = np.geomspace(1, curvature, N) - 1
        space01 /= space01[-1]

    return start + (end-start)*space01


def circulant_ACF(C, do_abs=False):
    """Compute the ACF of C.

    This assumes it is the cov/corr matrix of a 1D periodic domain."""
    M = len(C)
    # cols = np.flipud(sla.circulant(np.arange(M)[::-1]))
    cols = sla.circulant(np.arange(M))
    ACF = np.zeros(M)
    for i in range(M):
        row = C[i, cols[i]]
        if do_abs:
            row = abs(row)
        ACF += row
        # Note: this actually also accesses masked values in C.
    return ACF/M


########################
# Linear Algebra
########################
def mrdiv(b, A):
    "b/A"
    return sla.solve(A.T, b.T).T


def mldiv(A, b):
    "A \\ b"
    return sla.solve(A, b)


def truncate_rank(s, threshold, avoid_pathological):
    "Find r such that s[:r] contains the threshold proportion of s."
    assert isinstance(threshold, float)
    if threshold == 1.0:
        r = len(s)
    elif threshold < 1.0:
        r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
        r += 1  # Hence the strict inequality above
        if avoid_pathological:
            # If not avoid_pathological, then the last 4 diag. entries of
            # svdi( *tsvd(np.eye(400),0.99) )
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
    M, N = A.shape
    full_matrices = False

    if is_int(threshold):
        # Assume specific number is requested
        r = threshold
        assert 1 <= r <= max(M, N)
        if r > min(M, N):
            full_matrices = True
            r = min(M, N)

    U, s, VT = sla.svd(A, full_matrices)

    if isinstance(threshold, float):
        # Assume proportion is requested
        r = truncate_rank(s, threshold, avoid_pathological)

    # Truncate
    U = U[:, :r]
    VT = VT[:r]
    s = s[:r]
    return U, s, VT


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
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    "Pad ss with zeros so that len(ss)==N."
    out = np.zeros(N)
    out[:len(ss)] = ss
    return out


def svdi(U, s, VT):
    """Reconstruct matrix from (t)svd.

    Example::

    >>> A == svdi(*tsvd(A,1.0)).

    .. seealso:: sla.diagsvd().
    """
    return (U[:, :len(s)] * s) @ VT


def tinv(A, *kargs, **kwargs):
    """
    Inverse based on truncated svd.
    Also see sla.pinv2().
    """
    U, s, VT = tsvd(A, *kargs, **kwargs)
    return (VT.T * s**(-1.0)) @ U.T


def trank(A, *kargs, **kwargs):
    """Rank following truncation"""
    return len(tsvd(A, *kargs, **kwargs)[1])


########################
# HMM setup shortcuts
########################

def Id_op():
    "Id operator."
    return utils.NamedFunc(lambda *args: args[0], "Id operator")


def Id_mat(M):
    "Id matrix."
    Id = np.eye(M)
    return utils.NamedFunc(lambda x, t: Id, "Id("+str(M)+") matrix")


def linear_model_setup(ModelMatrix, dt0):
    r"""Make a dictionary the Dyn/Obs field of HMM representing a linear model.

    .. math::

      x(t+dt) = \texttt{ModelMatrix}^{dt/dt0} x(t),

    i.e.

    .. math::

      \frac{dx}{dt} = \frac{\log(\texttt{ModelMatrix})}{dt0} x(t).

    In typical use, ``dt0==dt`` (where ``dt`` is defined by the chronology).
    Anyways, ``dt`` must be an integer multiple of ``dt0``.
    """

    Mat = np.asarray(ModelMatrix)  # does not support sparse and matrix-class

    # Compute and cache ModelMatrix^(dt/dt0).
    @functools.lru_cache(maxsize=1)
    def MatPow(dt):
        assert is_whole(dt/dt0), "Mat. exponentiation unique only for integer powers."
        return sla.fractional_matrix_power(Mat, int(round(dt/dt0)))

    @ens_compatible
    def model(x, t, dt): return MatPow(dt) @ x
    def linear(x, t, dt): return MatPow(dt)

    Dyn = {
        'M': len(Mat),
        'model': model,
        'linear': linear,
    }
    return Dyn


def direct_obs_matrix(Nx, obs_inds):
    """Matrix that "picks" state elements obs_inds out of range(Nx)"""
    Ny = len(obs_inds)
    H = np.zeros((Ny, Nx))
    H[range(Ny), obs_inds] = 1

    # One-liner:
    # H = np.array([[i==j for i in range(M)] for j in jj],float)

    return H


def partial_Id_Obs(Nx, obs_inds):
    "Id observations of a subset of obs. indices."
    Ny = len(obs_inds)
    H = direct_obs_matrix(Nx, obs_inds)
    @ens_compatible
    def model(x, t): return x[obs_inds]
    def linear(x, t): return H
    Obs = {
        'M': Ny,
        'model': model,
        'linear': linear,
    }
    return Obs


def Id_Obs(Nx):
    "Id obs of entire state."
    return partial_Id_Obs(Nx, np.arange(Nx))
