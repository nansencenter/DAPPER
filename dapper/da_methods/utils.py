"""Utils for working with ensembles and weights."""

import numpy as np


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
