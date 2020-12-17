"""Math tools: integrators, linear algebra, and convenience funcs."""

import numpy as np
import scipy.linalg as sla


def center(E, axis=0, rescale=False):
    """Center ensemble.

    Makes use of `np` features: keepdims and broadcasting.

    Parameters
    ----------
    rescale: bool
        Whether to inflate to compensate for reduction in the expected variance.

    Returns
    -------
    Centered ensemble, and its mean.
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    """Center, but only return the anomalies (not the mean)."""
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate the ensemble (center, inflate, re-combine)."""
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor


def weight_degeneracy(w, prec=1e-10):
    """Check if the weights are degenerate."""
    return (1-w.max()) < prec


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """Compute unbias-ing factor for variance estimation.

    [Wikipedia](https://wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights)
    """
    if N_eff is None:
        N_eff = 1/(w@w)
    if avoid_pathological and weight_degeneracy(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1/(1 - 1/N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub


def isNone(x):
    """Like `x==None`, but also works for x being an np.ndarray.

    Since python3.8 `x is None` throws warning.

    Ref: `np.isscalar` docstring.
    """
    return np.ndim(x) == 0 and x == None


def curvedspace(start, end, N, curvature=1):
    """Take a segment of logspace, and scale it to [start,end].

    Parameters
    ----------
    curvature: float
        Special cases:

        - 0 produces `linspace(start, end, N)`
        - +1 produces `geomspace(start, end, N)`
        - -1 produces same as `+1`, but reflected about `y=x`

    Examples
    --------
    >>> plt.plot(np.geomspace(1e-1, 10, 201) ,label="geom") # doctest: +SKIP
    >>> plt.plot(np.linspace (1e-1, 10, 201) ,label="lin")  # doctest: +SKIP
    >>> plt.plot(curvedspace (1e-1, 10, 201, 0.5),'y--')    # doctest: +SKIP

    See Also
    --------
    np.logspace, np.geomspace, dapper.mods.utils.linspace_int
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
    """Compute the auto-covariance-function corresponding to `C`.

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
