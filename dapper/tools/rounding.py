"""Functions for rounding numbers."""

import functools

import numpy as np


def np_vectorize(f):
    """Like `np.vectorize`, but with some embellishments.

    - Includes `functools.wraps`
    - Applies `.item()` to output if input was a scalar.
    """
    vectorized = np.vectorize(f)

    @functools.wraps(f)
    def new(*args, **kwargs):
        output = vectorized(*args, **kwargs)
        if np.isscalar(args[0]) and not isinstance(args[0], np.ndarray):
            output = output.item()
        return output

    return new


@np_vectorize
def _round2prec(num, prec):
    """Don't use (directly)! Suffers from numerical precision.

    This function is left here just for reference. Use `round2` instead.

    The issue is that:
    >>> _round2prec(0.7,.1)
    0.7000000000000001
    """
    return prec * round(num / prec)


@np_vectorize
def log10int(x):
    """Compute decimal order, rounded down.

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


@np_vectorize
def round2(x, prec=1.0):
    r"""Round to a nice precision.

    Parameters
    ----------
    x : array_like
          Value to be rounded.
    prec: float
        Precision, before prettify, which is given by
        $$ \text{prec} = 10^{\text{floor}(-\log_{10}|\text{prec}|)} $$

    Returns
    -------
    Rounded value (always a float).

    See Also
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


@np_vectorize
def round2sigfig(x, sigfig=1):
    """Round to significant figures.

    Parameters
    ----------
    x
        Value to be rounded.
    sigfig
        Number of significant figures to include.

    Returns
    -------
    rounded value (always a float).

    See Also
    --------
    np.round : rounds to a given number of *decimals*.
    round2 : rounds to a given *precision*.

    Examples
    --------
    >>> round2sigfig(1234.5678, 1)
    1000.0
    >>> round2sigfig(1234.5678, 4)
    1235.0
    >>> round2sigfig(1234.5678, 6)
    1234.57
    """
    ndecimal = sigfig - log10int(x) - 1
    return np.round(x, ndecimal)


def is_whole(x, **kwargs):
    """Check if a number is a whole/natural number to precision given by `np.isclose`.

    For actual type checking, use `isinstance(x, (int, np.integer))`.
    """
    return np.isclose(x, round(x), **kwargs)
