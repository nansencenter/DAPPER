"""Functions for rounding numbers."""

import functools

import numpy as np

from dapper.dpr_config import rc


class UncertainQtty():
    """Data container associating uncertainty (confidence) to a quantity.

    Includes intelligent rounding and printing functionality.

    Usually, the precision parameter will be set to the (potentially estimated)
    standard deviation of an uncertain quantity.
    However, this class in itself does not define the `prec` attribute
    by anything else than what it does: impact the rounding & printing of `val`.

    Examples:
    >>> for c in [.01, .1, .2, .9, 1]:
    ...    print(UncertainQtty(1.2345, c))
    1.23 ±0.01
    1.2 ±0.1
    1.2 ±0.2
    1.2 ±0.9
    1 ±1

    >>> for c in [.01, 1e-10, 1e-17, 0]:
    ...    print(UncertainQtty(1.2, c))
    1.20 ±0.01
    1.2000000000 ±1e-10
    1.19999999999999996 ±1e-17
    1.2000000000 ±0

    Note that in the case of a confidence of exactly 0,
    it defaults to 10 decimal places.
    Meanwhile, a NaN confidence yields printing using `rc.sigfig`:

    >>> print(UncertainQtty(1.234567, np.nan))
    1.235 ±nan

    Also note the effect of large uncertainty:

    >>> for c in [1, 9, 10, 11, 20, 100, np.inf]:
    ...    print(UncertainQtty(12, c))
    12 ±1
    12 ±9
    10 ±10
    10 ±10
    10 ±20
    0 ±100
    0 ±inf
    """

    def __init__(self, val, prec=np.nan):
        self.val = val
        self.prec = prec

    def round(self=1.0):  # noqa
        """Round intelligently.

        - `prec` to 1 sig.fig.
        - `val` to `round2(val, prec)`.
        """
        if np.isnan(self.prec):
            # Fallback to rc.sigfig
            c = self.prec
            v = round2sigfig(self.val, rc.sigfig)
        else:
            # Normal/general case
            c = round2sigfig(self.prec, 1)
            v = round2(self.val, self.prec)
        return v, c

    def __str__(self):
        """Returns 'val ±prec', using `UncertainQtty.round` and some finesse."""
        v, c = self.round()

        if np.isnan(c):
            # Rounding to fallback (rc.sigfig) already took place
            return f"{v} ±{c}"
        elif c == 0:
            # 0 (i.e. not 1e-300) never arises "naturally" => Treat it "magically"
            # by truncating to a default. Also see https://stackoverflow.com/a/25899600
            n = -10
        else:
            # Normal/general case.
            n = log10int(c)

        frmt = "%.f"
        if n < 0:
            # Ensure we get 1.30 ±0.01, NOT 1.3 ±0.01:
            frmt = "%%0.%df" % -n
        elif np.isfinite(c):
            # if c >= 1.0
            c = int(c)
        v = frmt % v

        return f"{v} ±{c}"

    def __repr__(self):
        """Essentially the same as `__str__`."""
        v, c = str(self).split(" ±")
        return self.__class__.__name__ + f"(val={v}, prec={c})"

    @property
    def prec(self):
        return self._prec

    @prec.setter
    def prec(self, prec):
        assert prec >= 0. or np.isnan(prec), "'prec' must be non-negative."
        self._prec = prec


def np_vectorize(f):
    """Like `np.vectorize`, but with some embellishments.

    - Includes `functools.wraps`
    - Applies `.item()` to output if input was a scalar.

    Parameters
    ----------
    f: callable
        Your function.

    Returns
    -------
    vectorized: callable
        Your function, now element-wise applicable to an iterable.
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
    r"""Round x to the decimal order appropriate for the precision.

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
    `round2sigfig`

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
    `round2` : rounds to a given *precision*.

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

    Parameters
    ----------
    x: float or ndarray
        Values to be checked

    Returns
    -------
    l: bool
        True if rounded x is close to x, otherwise False
    """
    return np.isclose(x, round(x), **kwargs)
