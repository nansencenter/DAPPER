"""
Math tools: integrators, linear algebra, and convenience funcs.

Note
----
The following functions are not auto-documented by pdoc:
`log10int`, `round2`, `round2sigfig`
"""
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
    """
    Check if input is a 1D array.

    Parameters
    ----------
    a : `ndarray`
        Any array to check if it is one-dimensional

    Returns
    -------
    logical
        True if it is one dimensional otherwise False
    """

    # return np.asarray(a).ndim == 1
    return np.sum(np.asarray(np.asarray(a).shape) > 1) <= 1


def exactly_1d(a):
    """
    Turn one-dimensional list/array into array and check 
    the dimension of input array, if not 1D raise
    an error.

    Parameters
    ----------
    a : `sequence`
        Only 1-D sequences are expected

    Returns
    -------
    a : `ndarray`
        1-D array
    """ 

    a = np.atleast_1d(a)
    assert a.ndim == 1, "input array should be exactly one dimensional"
    return a


def exactly_2d(a):
    """
    Turn two-dimensional list/array into array and check 
    the dimension of input array, if not 2D raise
    an error.

    Parameters
    ----------
    a : `sequence`
        Only 2-D sequences are expected

    Returns
    -------
    a : `ndarray`
        2-D array
    """
    a = np.atleast_2d(a)
    assert a.ndim == 2, 'Input array must be 2-D'
    return a


def ccat(*args, axis=0):
    """
    Concatenate multiple arrays/lists into one array
    for given axis. By default it concatenate the 
    first dimension.

    Parameters
    ----------
    *args : `sequence`
        sequences to be concatenated.

    axis : `int`, `optional`
        The dimension to be concatenated. Default: 0
        
    Returns
    -------
    ndarray
        Array concatenated in the given axis
    """

    args = [np.atleast_1d(x) for x in args]
    return np.concatenate(args, axis=axis)


########################
# Ensemble matrix manip
########################
def ens_compatible(func):
    """
    Function decorator used to tranpose the input and returned array.

    Helpful to make functions compatible with both 1d and 2d ndarrays.

    An older version also used np.atleast_2d and squeeze(), but that is more
    messy than necessary.

    Note
    ----
    This is not the_way™ -- other tricks are sometimes more practical.
    See for example dxdt() in __init__.py for LorenzUV, Lorenz96, LotkaVolterra.
    """
    @functools.wraps(func)
    def wrapr(x, *args, **kwargs):
        return np.asarray(func(x.T, *args, **kwargs)).T
    return wrapr


def center(E, axis=0, rescale=False):
    """
    Center ensemble.

    Makes use of np features: keepdims and broadcasting.
     
    Parameters
    ----------
    E : ndarray
        Ensemble which going to be inflated

    axis : int, optional
        The axis to be centered. Default: 0

    rescale: logical, optional
        If True, inflate to compensate for reduction in the expected variance.
        The inflation factor is \(\sqrt{\\frac{N}{N - 1}}\) 
        where N is the ensemble size. Default: False

    Returns
    -------
    X : ndarray
        Ensemble anomaly

    x : ndarray
        Mean of the ensemble
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    """
    Same as: center(E,rescale=True)[0]

    Parameters
    ----------
    E : ndarray
        Ensemble which going to be inflated

    axis : int, optional
        The axis to be centered. Default: 0

    rescale: logical, optional
        If True, inflate to compensate for reduction in the expected variance.
        The inflation factor is \(\sqrt{\\frac{N}{N - 1}}\). Act as a way for 
        unbiased variance estimation?
        where N is the ensemble size. Default: True


    Returns
    -------
    ndarray
        Ensemble mean
    """
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """
    Inflate the ensemble E by an multiplicative factor

    Parameters
    ----------
    E : ndarray
        Ensemble which going to be inflated

    factor: `float`
        Inflation factor


    Returns
    -------
    ndarray
        Inflated ensemble
    """
    if np.isclose(factor, 1.):
        return E
    X, x = center(E)
    return x + X*factor


def is_degenerate_weight(w, prec=1e-10):
    """
    Check if the input w (weight) is degenerate
    If it is degenerate, the maximum weight
    should be nearly one since sum(w) = 1

    Parameters
    ----------
    w : ndarray
        Weight of something. Sum(w) = 1

    prec: float, optional
        Tolerance of the distance between w and one. Default:1e-10

    Returns
    -------
    logical
        If weight is degenerate True, else False
    """
    return np.isclose(1, w.max(), rtol = prec, atol = prec)


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """
    Compute unbias-ing factor for variance estimation.

    Parameters
    ----------
    w : ndarray, optional
        Weight of something. Sum(w) = 1. 
        Only one of w and N_eff can be None. Default: None

    N_eff: float, optional
        Tolerance of the distance between w and one. 
        Only one of w and N_eff can be None.  Default: None

    avoid_pathological: logical, optional
        Avoid weight collapse. Default: False

    Returns
    -------
    ub : float
        factor used to unbiasing variance

    See Also
    --------
    `https://wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights`
    """
    if N_eff is None:
        N_eff = 1/(w@w)
    if avoid_pathological and is_degenerate_weight(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1/(1 - 1/N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub


########################
# Time stepping (integration)
########################

# fmt: off
def rk4(f, x, t, dt, order=4):
    """
    Runge-Kutta N-th order (explicit, non-adaptive, up to 4th order) numerical ODE solvers.

    Parameters
    ----------
    f : function
        The forcing of the ODE must a function of the form f(t, x)

    x : ndarray or float
        State vector of the forcing term

    t : float
        Starting time of the integration

    dt : float
        Time step interval of the ODE solver

    order : integer, optional
        The order of RK method. Default: 4

    Returns
    -------
    ndarray
        State vector at the new time step t+dt

    """
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

def RK4_adj(f, f_adj, x0, t, dt):
    """
    Runge-Kutta 4-th order (explicit, non-adaptive) numerical solver
    for adjoint equations.

    Parameters
    ----------
    f : function
        The forcing of the ODE must a function of the form f(t, x)

    f_adjoint : function
        The product of M.T@dx where M.T is the transpose of the 
        Jacobian of the f(t, x) and x is a sequence of x and dx

    x0 : sequence
        State vector of the forcing term as the first element 
        dx as the second element

    t : float
        Starting time of the integration

    dt : float
        Time step interval of the ODE solver

    Returns
    -------
    x : ndarray
        State vector at the new time step t+dt

    """    
    x, dx = x0[0], x0[1]
    k1 = dt*f(t, x)
    k2 = dt*f(t+dt/2, x+k1/2)
    k3 = dt*f(t+dt/2, x+k2/2)

    k1 = dt*f_adj(t, (x + k3, dx))
    k2 = dt*f_adj(t, (x + 0.5*k2, dx + 0.5*k1))

    k3 = dt*f_adj(t, (x + 0.5*k1, dx + 0.5*k2))

    k4 = dt*f_adj(t, (x, dx + k3))

    return dx + (k1 + 2.*(k2 + k3) + k4)/6.

def RK4_linear(f, f_tangent, x, M, t, dt):
    """
    Runge-Kutta 4-th order (explicit, non-adaptive) numerical solver
    for tangent linear matrix.

    Parameters
    ----------
    f : function
        The forcing of the ODE must a function of the form f(t, x)

    f_tangent : function
        The function for the Jacobian of forcing terms, which returns
        a matrix M.

    x : ndarray or float
        State vector of the forcing term

    M : ndarray
        Usually an identity matrix for the RK4 integration of the 
        tangent linear matrix

    t : float
        Starting time of the integration

    dt : float
        Time step interval of the ODE solver

    Returns
    -------
    x : ndarray
        State vector at the new time step t+dt

    """    
    k1_f = dt*f(t, x)
    k2_f = dt*f(t+dt/2, x+k1_f/2)
    k3_f = dt*f(t+dt/2, x+k2_f/2)

    k1 = dt*f_tangent(t, x, M)
    k2 = dt*f_tangent(t+dt/2, x+k1_f/2., M+k1/2.)
    k3 = dt*f_tangent(t+dt/2, x+k2_f/2., M+k2/2.)
    k4 = dt*f_tangent(t+dt/2, x+k3_f, M+k3)
    return M + (k1 + 2*(k2 + k3) + k4)/6

def with_rk4(dxdt, autonom=False, order=4):
    """
    Helper method for RK4

    Parameters
    ----------
    dxdt : function
        The forcing of the ODE

    autonom : logical, optional
        If the function is autonomous (only dependent on x and not t),
        set it to be true. Default: False

    order : int, optional
        The order of RK method. Default: 4

    Returns
    -------
    step : function
        A function used to do numerical integration with RK-order

    """
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
    """
    Make function recursive in its 1st arg.


    Parameters
    ----------
    func : func
        Run the input function recursively.

    prog : logical or str
        Determine the mode of progress bar.    

    Returns
    -------
    fun_k : func
        A function that returns the sequence generated by recursively
        run func.

        Stepping of dynamical system

        Parameters
        ----------
        x0 : `float`
            initial condition or first entry of the recursive function

        k : int
            number of recursive time

        *args : `any`
            args for the input func

        **kwargs : `any`
            keyword arguments for the input func

        Returns
        -------
        xx : ndarray
            Trajectory of model evolution

    Examples
    -------

    >>> def step(x,t,dt): 
    ...     ...
    >>> step_k = with_recursion(step)
    >>> x[k]   = step_k(x0,k,t=NaN,dt)[-1]
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

    See Also
    --------
    `FD_Jac`
    """
    if method == 'analytic':
        Lambda, V = sla.eig(TLM)
        resolvent = (V * np.exp(dt*Lambda)) @ np.linalg.inv(V)
        resolvent = np.real_if_close(resolvent, tol=10**5)
    else:
        Id = np.eye(TLM.shape[-1])
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
    """
    Don't use (directly)! Suffers from numerical precision.

    This function is left here just for reference. Use `round2` instead.

    The issue is that:
    >>> _round2prec(0.7,.1)
    0.7000000000000001
    """
    return prec * round(num / prec)


@np.vectorize
def log10int(x):
    """
    Compute decimal order, rounded down.

    Conversion to `int` means that we cannot return nan's or +/- infinity,
    even though this could be meaningful. Instead, we return integers of magnitude
    a little less than IEEE floating point max/min-ima instead.
    This avoids a lot of clauses in the parent/callers to this function.

    Examples
    --------
    >>> log10int([1e-1, 1e-2, 1, 3, 10, np.inf, -np.inf, np.nan])
    array([  -1,   -2,    0,    0,    1,  300, -300, -300])    
    """

    # Extreme cases -- https://stackoverflow.com/q/65248379
    if np.isnan(x):
        y = -300
    elif x < 1e-300:
        y = -300
    elif x > 1e+300:
        y = +300
    # Normal case
    else:
        y = int(np.floor(np.log10(np.abs(x))))
    return y


@np.vectorize
def round2(x, prec=1.0):
    """Round to a nice precision, namely

    $$ 10^{\\text{floor}(-\\log_{10}|\\text{prec}|)} $$

    Parameters
    ----------
    x : Value to be rounded.
    prec : Precision, before prettify.

    Returns
    -------
    Rounded value (always a float).

    See also
    --------
    round2sigfig

    Examples
    --------
    >>> round2(1.65, 0.543)
    1.6
    >>> round2(1.66, 0.543)
    1.7
    >>> round2(1.65, 1.234)
    2.0
    """
    if np.isnan(prec):
        return x
    ndecimal = -log10int(prec)
    return np.round(x, ndecimal)


@np.vectorize
def round2sigfig(x, sigfig=1):
    """Round to significant figures.

    Parameters
    ----------
    x : Value to be rounded.
    sigfig : Number of significant figures to include.

    Returns
    -------
    rounded value (always a float).

    See also
    --------
    round2
    np.round, which rounds to a given number of *decimals*.

    Examples
    --------
    >>> round2sigfig(1234.5678, 1)
    1000
    >>> round2sigfig(1234.5678, 4)
    1234
    >>> round2sigfig(1234.5678, 6)
    1234.57
    """
    ndecimal = sigfig - log10int(x) - 1
    return np.round(x, ndecimal)

def is_int(a):
    """
    Check if the input is any type of integer
    
    Parameters
    ----------
    a : any
        Values to be checked

    Returns
    -------
    l : logical
        True if it is int, otherwise False

    See Also
    --------
    'https://stackoverflow.com/q/37726830'
    """
    return np.issubdtype(type(a), np.integer)


def is_whole(x):
    """
    Check if rounded x is close to the original value
    
    Parameters
    ----------
    x : float or ndarray
        Values to be checked

    Returns
    -------
    l : logical
        True if rounded x is close to x, otherwise False
    """
    return np.isclose(x, round(x))


# def validate_int(x):
#     """
#     Get rounded input if it does not affect the accuracy of the value
#     If rounded value affects the accuracy of x, raise Error

#     Parameters
#     ----------
#     x : float or ndarray
#         Values to be rounded

#     Returns
#     -------
#      : float or ndarray
#         Rounded value
#     """
#     assert is_whole(x)
#     return round(x)  # convert to int


# def is_None(x):
#     """x==None that also works for x being an np.ndarray.

#     Since python3.8 ``x is None`` throws warning.

#     Ref: np.isscalar docstring."""
#     return np.ndim(x) == 0 and x == None


########################
# Misc
########################
def linspace_int(Nx, Ny, periodic=True):
    """
    Provide a range of Ny equispaced integers between 0 and Nx-1
    
    Parameters
    ----------
    Nx : int
        Range of integers
    Ny : int
        Number of integers
    periodic : logical, optional
        Whether the vector is periodic. 
        Determines if the Nx == 0.
        Default: True

    Returns
    -------
     : vector
        Generated vectors.

    Examples
    --------
    >>> linspace_int(10, 10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> linspace_int(10, 4)
    array([0, 2, 5, 7])
    >>> linspace_int(10, 5)
    array([0, 2, 4, 6, 8])
    >>>
    """
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
    """
    Compute the ACF of C.

    This assumes it is the cov/corr matrix of a 1D periodic domain.
    """
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
    """b/A"""
    return sla.solve(A.T, b.T).T


def mldiv(A, b):
    """A \\ b"""
    return sla.solve(A, b)


def truncate_rank(s, threshold, avoid_pathological):
    """Find r such that s[:r] contains the threshold proportion of s."""
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
    r"""
    Make a dictionary the Dyn/Obs field of HMM representing a linear model.

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
    """
    Matrix that "picks" state elements obs_inds out of range(Nx)

    Parameters
    ----------
    Nx : int
        Number of total length of state vector
    obs_inds : ndarray
        The observed indices.

    Returns
    -------
    H : ndarray
        The observation matrix for direct partial observations.

    """
    Ny = len(obs_inds)
    H = np.zeros((Ny, Nx))
    H[range(Ny), obs_inds] = 1

    # One-liner:
    # H = np.array([[i==j for i in range(M)] for j in jj],float)

    return H


def partial_Id_Obs(Nx, obs_inds):
    """
    Observation operator for partial variable observations.
    It is not a function of time.

    Parameters
    ----------
    Nx : int
        Number of total length of state vector
    obs_inds : ndarray
        The observed indices.

    Returns
    -------
    Obs : dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """
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
    """
    Observation operator for all variable observations.
    It is not a function of time.

    Parameters
    ----------
    Nx : int
        Number of total length of state vector

    Returns
    -------
    Obs : dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """    
    return partial_Id_Obs(Nx, np.arange(Nx))
