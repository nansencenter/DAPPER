"""Time series management and processing."""

from __future__ import annotations

import numpy as np
from numpy import nan
from patlib.std import find_1st_ind

from dapper.tools.repr_util import YamlRepr
from dapper.tools.rounding import UncertainQtty


def auto_cov(xx, nlags=4, zero_mean=False, corr=False):
    """Auto covariance function, computed along axis 0.

    - `nlags`: max lag (offset) for which to compute acf.
    - `corr` : normalize acf by `acf[0]` so as to return auto-CORRELATION.

    With `corr=True`, this is identical to
    `statsmodels.tsa.stattools.acf(xx,True,nlags)`
    """
    assert nlags < len(xx)

    N = len(xx)
    A = xx if zero_mean else (xx - xx.mean(0))
    acovf = np.zeros((nlags + 1,) + xx.shape[1:])

    for i in range(nlags + 1):
        Left = A[np.arange(N - i)]
        Right = A[np.arange(i, N)]
        acovf[i] = (Left * Right).sum(0) / (N - i)

    if corr:
        acovf /= acovf[0]

    return acovf


def fit_acf_by_AR1(acf_empir, nlags=None):
    """Fit an empirical auto cov function (ACF) by that of an AR1 process.

    - `acf_empir`: auto-corr/cov-function.
    - `nlags`: length of ACF to use in AR(1) fitting
    """
    if nlags is None:
        nlags = len(acf_empir)

    # geometric_mean = ss.mstats.gmean
    def geometric_mean(xx):
        return np.exp(np.mean(np.log(xx)))

    def mean_ratio(xx):
        return geometric_mean([xx[i] / xx[i - 1] for i in range(1, len(xx))])

    # Negative correlation => Truncate ACF
    neg_ind = find_1st_ind(np.array(acf_empir) <= 0)
    acf_empir = acf_empir[:neg_ind]

    if len(acf_empir) == 0:
        return 0
    elif len(acf_empir) == 1:
        return 0.01
    else:
        return mean_ratio(acf_empir)


def estimate_corr_length(xx):
    r"""Estimate the correlation length of a time series.

    For explanation, see [`mods.LA.homogeneous_1D_cov`][].
    Also note that, for exponential corr function, as assumed here,

    $$\text{corr}(L) = \exp(-1) \approx 0.368$$
    """
    acovf = auto_cov(xx, min(100, len(xx) - 2))
    a = fit_acf_by_AR1(acovf)
    if a == 0:
        L = 0
    else:
        L = 1 / np.log(1 / a)
    return L


def mean_with_conf(xx):
    """Compute the mean of a 1d iterable `xx`.

    Also provide confidence of mean,
    as estimated from its correlation-corrected variance.
    """
    mu = np.mean(xx)
    N = len(xx)
    # TODO 3: review
    if (not np.isfinite(mu)) or N <= 5:
        uq = UncertainQtty(mu, np.nan)
    elif np.allclose(xx, mu):
        uq = UncertainQtty(mu, 0)
    else:
        acovf = auto_cov(xx)
        var = acovf[0]
        var /= N
        # Estimate (fit) ACF
        a = fit_acf_by_AR1(acovf)
        # If xx[k] where independent of xx[k-1],
        # then std_of_mu is the end of the story.
        # The following corrects for the correlation in the time series.
        #
        # See https://stats.stackexchange.com/q/90062
        # c = sum([(N-k)*a**k for k in range(1,N)])
        # But this series is analytically tractable:
        c = ((N - 1) * a - N * a**2 + a ** (N + 1)) / (1 - a) ** 2
        confidence_correction = 1 + 2 / N * c
        var *= confidence_correction
        uq = UncertainQtty(mu, np.sqrt(var))
    return uq


class StatPrint(YamlRepr):
    """Mixin that pretty-prints stats objects via YAML."""

    _print_excluded: frozenset = frozenset(["HMM", "LP_instance"])
    _print_aliases: dict = {
        "f": "Forecast (.f)",
        "a": "Analysis (.a)",
        "s": "Smoothed (.s)",
        "i": "Integral (.i)",
        "m": "Field mean (.m)",
        "ma": "Field mean-abs (.ma)",
        "rms": "Field root-mean-square (.rms)",
        "gm": "Field geometric-mean (.gm)",
    }

    def _repr_fields(self):
        with np.printoptions(threshold=10, precision=3):
            return {
                self._print_aliases.get(k, k): v
                for k, v in vars(self).items()
                if k not in self._print_excluded and not k.startswith("_")
            }


class DACycleSeries(StatPrint):
    """Container for time series of a statistic across one DA cycle.

    Four attributes, each an ndarray:

    - `.f` for forecast      , shape `(Ko+1,)+item_shape`
    - `.a` for analysis      , shape `(Ko+1,)+item_shape`
    - `.s` for smoothed      , shape `(Ko+1,)+item_shape`
    - `.i` for integrational , shape `(K +1,)+item_shape`

    `.i` covers every model time step, including the ones between observations.
    The name fits three synonyms: **integrational** (steps taken by the model
    integrator), **intermediate** (between analysis times), **intervening**
    (between obs times).

    If `store_i=False`, `.i` has shape `(1,)+item_shape` — only the
    most-recently-written step is kept (ring buffer of size 1).

    Use direct attribute access::

        stat.f[ko] = val
        stat.a[ko] = val
        stat.i[k]  = val   # or stat.i[0] when store_i=False

    !!! note
        If a series only pertains to analysis times, pass `store_f=False`.
    """

    f: np.ndarray
    """Forecast — shape `(Ko+1,)+item_shape`. Absent when `store_f=False`."""
    a: np.ndarray
    """Analysis — shape `(Ko+1,)+item_shape`. Always present."""
    s: np.ndarray
    """Smoothed — shape `(Ko+1,)+item_shape`. Absent when `store_s=False`."""
    i: np.ndarray
    """Integrational — shape `(K+1,)+item_shape`. Always present
    (ring buffer of 1 when `store_i=False`)."""

    # Field-summary children, present on vector stats (populated by Stats.new_series).
    m: DACycleSeries
    """Mean-field scalar time series."""
    ms: DACycleSeries
    """Mean-square scalar time series."""
    rms: DACycleSeries
    """Root-mean-square scalar time series."""
    ma: DACycleSeries
    """Mean-absolute scalar time series."""
    gm: DACycleSeries
    """Geometric-mean scalar time series."""

    def __init__(self, K, Ko, item_shape, store_i, store_s, store_f=True, **kwargs):
        """Construct object.

        - `item_shape` : shape of an item in the series.
        - `store_i`    : if False: only the current value is stored in `.i`.
        - `store_f`    : if False: `.f` is not created (analysis-only stats).
        - `kwargs`     : passed on to ndarrays.
        """
        fill_value = -99 if kwargs.get("dtype", None) is int else nan
        if store_f:
            self.f = np.full((Ko + 1,) + item_shape, fill_value, **kwargs)
        self.a = np.full((Ko + 1,) + item_shape, fill_value, **kwargs)
        if store_s:
            self.s = np.full((Ko + 1,) + item_shape, fill_value, **kwargs)
        if store_i:
            self.i = np.full((K + 1,) + item_shape, fill_value, **kwargs)
        else:
            self.i = np.full((1,) + item_shape, fill_value, **kwargs)

    # Using property => won't appear in vars(self), and read-only.
    item_shape = property(lambda self: self.a.shape[1:])
    store_f = property(lambda self: hasattr(self, "f"))
    store_i = property(lambda self: len(self.i) > 1)

    def __getitem__(self, key):
        """Read via `(k, ko, sub)` tuple — used internally by liveplotting."""
        sub = key[-1]
        ind = (key[0] if self.store_i else 0) if sub == "i" else key[-2]
        return getattr(self, sub)[ind]


class RollingArray:
    """ND-Array that implements "leftward rolling" along axis 0.

    Used for data that gets plotted in sliding graphs.
    """

    def __init__(self, shape, fillval=nan):
        self.array = np.full(shape, fillval)
        self.k1 = 0  # previous k
        self.nFilled = 0

    def insert(self, k, val):
        dk = k - self.k1

        # Old (more readable?) version:
        # if dk in [0,1]: # case: forecast or analysis update
        # self.array = np.roll(self.array, -1, axis=0)
        # elif dk>1:      # case: user has skipped ahead (w/o liveplotting)
        # self.array = np.roll(self.array, -dk, axis=0)
        # self.array[-dk:] = nan
        # self.array[-1] = val

        dk = max(1, dk)
        # TODO 7: Should have used deque?
        self.array = np.roll(self.array, -dk, axis=0)
        self.array[-dk:] = nan
        self.array[-1:] = val

        self.k1 = k
        self.nFilled = min(len(self), self.nFilled + dk)

    def leftmost(self):
        return self[len(self) - self.nFilled]

    def span(self):
        return (self.leftmost(), self[-1])

    @property
    def T(self):
        return self.array.T

    def __array__(self, _dtype=None):
        return self.array

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"RollingArray:\n{self.array!s}"

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, val):
        # Don't implement __setitem__ coz leftmost() is then
        # not generally meaningful (i.e. if an element is set in the middle).
        # Of course self.array can still be messed with.
        raise AttributeError("Values should be set with update()")
