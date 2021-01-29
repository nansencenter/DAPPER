"""Time stepping (integration) tools."""

import functools

import numpy as np
import scipy.linalg as sla
from IPython.lib.pretty import pretty as pretty_repr

from dapper.tools.progressbar import progbar

from .utils import NamedFunc


# fmt: off
def rk4(f, x, t, dt, order=4):
    """Runge-Kutta (explicit, non-adaptive) numerical ODE solvers.

    Parameters
    ----------
    order: int
        The order of the scheme.
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


def with_rk4(dxdt, autonom=False, order=4):
    """Wrap `dxdt` in `rk4`."""
    integrator = functools.partial(rk4, order=order)
    if autonom:
        def step(x0, t0, dt):
            return integrator(lambda t, x: dxdt(x), x0, np.nan, dt)
    else:
        def step(x0, t0, dt):
            return integrator(dxdt, x0, t0, dt)
    name = "rk"+str(order)+" integration of "+pretty_repr(dxdt)
    step = NamedFunc(step, name)
    return step


def with_recursion(func, prog=False):
    """Make function recursive in its 1st arg.

    Return a version of `func` whose 2nd argument (`k`)
    specifies the number of times to times apply func on its output.

    .. caution:: Only the first argument to `func` will change,
        so, for example, if `func` is `step(x, t, dt)`,
        it will get fed the same `t` and `dt` at each iteration.

    Examples
    --------
    >>> def dxdt(x):
    ...     return -x
    >>> step_1  = with_rk4(dxdt, autonom=True)
    >>> step_k  = with_recursion(step_1)
    >>> x0      = np.arange(3)
    >>> x7      = step_k(x0, 7, t0=np.nan, dt=0.1)[-1]
    >>> x7_true = x0 * np.exp(-0.7)
    >>> np.allclose(x7, x7_true)
    True
    """
    def fun_k(x0, k, *args, **kwargs):
        xx = np.zeros((k+1,)+x0.shape)
        xx[0] = x0

        rg = range(k)
        if isinstance(prog, str):
            rg = progbar(rg, prog)
        elif prog:
            rg = progbar(rg, 'Recurs.')

        for i in rg:
            xx[i+1] = func(xx[i], *args, **kwargs)

        return xx

    return fun_k


def integrate_TLM(TLM, dt, method='approx'):
    r"""Compute the resolvent.

    The resolvent may also be called

    - the Jacobian of the step func.
    - the integral of (with *M* as the TLM):
      $$ \frac{d U}{d t} = M U, \quad U_0 = I .$$

    .. note:: the tangent linear model (TLM)
              is assumed constant (for each `method` below).

    Parameters
    ----------
    method : str
        - `'analytic'`: exact.
        - `'approx'`  : derived from the forward-euler scheme.
        - `'rk4'`     : higher-precision approx.

    .. caution:: 'analytic' typically requries higher inflation in the ExtKF.

    See Also
    --------
    FD_Jac.
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


def FD_Jac(func, eps=1e-7):
    """Finite-diff approx. of Jacobian of `func`.

    The function `func(x)` must be compatible with `x.ndim == 1` and `2`,
    where, in the 2D case, each row is seen as one function input.

    Returns
    -------
    function
        The first input argument is that of which the derivative is taken.


    Examples
    --------
    >>> dstep_dx = FD_Jac(step) # doctest: +SKIP
    """
    def Jac(x, *args, **kwargs):
        def f(y):
            return func(y, *args, **kwargs)
        E = x + eps*np.eye(len(x))  # row-oriented ensemble
        FT = (f(E) - f(x))/eps      # => correct broadcasting
        return FT.T                 # => Jac[i,j] = df_i/dx_j
    return Jac

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
