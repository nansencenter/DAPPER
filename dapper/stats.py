"""Statistics for the assessment of DA methods.

`Stats` is a data container for ([mostly] time series of) statistics.
It comes with a battery of methods to compute the default statistics.

`Avrgs` is a data container *for the same statistics*,
but after they have been averaged in time (after the assimilation has finished).

Instances of these objects are created by `dapper.da_methods.da_method`
(i.e. "`xp`") objects and written to their `.stats` and `.avrgs` attributes.

.. include:: ../docs/stats_etc.md
"""

import warnings

import numpy as np
import scipy.linalg as sla
import struct_tools
from matplotlib import pyplot as plt
from patlib.std import do_once
from tabulate import tabulate

import dapper.tools.liveplotting as liveplotting
import dapper.tools.series as series
from dapper.dpr_config import rc
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar

__pdoc__ = {"align_col": False}


class Stats(series.StatPrint):
    """Contains and computes statistics of the DA methods."""

    def __init__(self, xp, HMM, xx, yy, liveplots=False, store_u=rc.store_u):
        """Init the default statistics."""
        ######################################
        # Preamble
        ######################################
        self.xp        = xp
        self.HMM       = HMM
        self.xx        = xx
        self.yy        = yy
        self.liveplots = liveplots
        self.store_u   = store_u
        self.store_s   = any(key in xp.__dict__ for key in
                             ["Lag", "DeCorr"])  # prms used by smoothers

        # Shapes
        K  = xx.shape[0] - 1
        Nx = xx.shape[1]
        Ko = yy.shape[0] - 1
        Ny = yy.shape[1]
        self.K , self.Nx = K, Nx
        self.Ko, self.Ny = Ko, Ny

        # Methods for summarizing multivariate stats ("fields") as scalars
        # Don't use nanmean here; nan's should get propagated!
        en_mean = lambda x: np.mean(x, axis=0)  # noqa
        self.field_summaries = dict(
            m   = lambda x: en_mean(x),                  # mean-field
            ms  = lambda x: en_mean(x**2),               # root-mean-square
            rms = lambda x: np.sqrt(en_mean(x**2)),      # root-mean-square
            ma  = lambda x: en_mean(np.abs(x)),          # mean-absolute
            gm  = lambda x: np.exp(en_mean(np.log(x))),  # geometric mean
        )
        # Only keep the methods listed in rc
        self.field_summaries = struct_tools.intersect(self.field_summaries,
                                                      rc.field_summaries)

        # Define similar methods, but restricted to sectors
        self.sector_summaries = {}
        def restrict(fun, inds): return (lambda x: fun(x[inds]))
        for suffix, formula in self.field_summaries.items():
            for sector, inds in HMM.sectors.items():
                f = restrict(formula, inds)
                self.sector_summaries['%s.%s' % (suffix, sector)] = f

        ######################################
        # Allocate time series of various stats
        ######################################
        self.new_series('mu'    , Nx, field_mean='sectors')  # Mean
        self.new_series('spread', Nx, field_mean='sectors')  # Std. dev. ("spread")
        self.new_series('err'   , Nx, field_mean='sectors')  # Error (mu - truth)
        self.new_series('gscore', Nx, field_mean='sectors')  # Gaussian (log) score

        # To save memory, we only store these field means:
        self.new_series('mad' , 1)  # Mean abs deviations
        self.new_series('skew', 1)  # Skewness
        self.new_series('kurt', 1)  # Kurtosis

        if hasattr(xp, 'N'):
            N = xp.N
            self.new_series('w', N, field_mean=True)  # Importance weights
            self.new_series('rh', Nx, dtype=int)  # Rank histogram

            self._is_ens = True
            minN = min(Nx, N)
            self.do_spectral  = np.sqrt(Nx*N) <= rc.comps["max_spectral"]
        else:
            self._is_ens = False
            minN = Nx
            self.do_spectral = Nx <= rc.comps["max_spectral"]

        if self.do_spectral:
            # Note: the mean-field and RMS time-series of
            # (i) svals and (ii) umisf should match the corresponding series of
            # (i) spread and (ii) err.
            self.new_series('svals', minN)  # Principal component (SVD) scores
            self.new_series('umisf', minN)  # Error in component directions

        ######################################
        # Allocate a few series for outside use
        ######################################
        self.new_series('trHK' , 1, Ko+1)
        self.new_series('infl' , 1, Ko+1)
        self.new_series('iters', 1, Ko+1)

        # Weight-related
        self.new_series('N_eff' , 1, Ko+1)
        self.new_series('wroot' , 1, Ko+1)
        self.new_series('resmpl', 1, Ko+1)

    def new_series(self, name, shape, length='FAUSt', field_mean=False, **kws):
        """Create (and register) a statistics time series, initialized with `nan`s.

        If `length` is an integer, a `DataSeries` (a trivial subclass of
        `numpy.ndarray`) is made. By default, though, a `series.FAUSt` is created.

        NB: The `sliding_diagnostics` liveplotting relies on detecting `nan`'s
            to avoid plotting stats that are not being used.
            Thus, you cannot use `dtype=bool` or `int` for stats that get plotted.
        """
        # Convert int shape to tuple
        if not hasattr(shape, '__len__'):
            if shape == 1:
                shape = ()
            else:
                shape = (shape,)

        def make_series(parent, name, shape):
            if length == 'FAUSt':
                total_shape = self.K, self.Ko, shape
                store_opts = self.store_u, self.store_s
                tseries = series.FAUSt(*total_shape, *store_opts, **kws)
            else:
                total_shape = (length,)+shape
                tseries = series.DataSeries(total_shape, *kws)
            register_stat(parent, name, tseries)

        # Principal series
        make_series(self, name, shape)

        # Summary (scalar) series:
        if shape != ():
            if field_mean:
                for suffix in self.field_summaries:
                    make_series(getattr(self, name), suffix, ())
            # Make a nested level for sectors
            if field_mean == 'sectors':
                for ss in self.sector_summaries:
                    suffix, sector = ss.split('.')
                    make_series(struct_tools.deep_getattr(
                        self, f"{name}.{suffix}"), sector, ())

    @property
    def data_series(self):
        return [k for k in vars(self)
                if isinstance(getattr(self, k), series.DataSeries)]

    def assess(self, k, ko=None, faus=None,
               E=None, w=None, mu=None, Cov=None):
        """Common interface for both `Stats.assess_ens` and `Stats.assess_ext`.

        The `_ens` assessment function gets called if `E is not None`,
        and `_ext` if `mu is not None`.

        faus: One or more of `['f',' a', 'u', 's']`, indicating
              that the result should be stored in (respectively)
              the forecast/analysis/universal attribute.
              Default: `'u' if ko is None else 'au' ('a' and 'u')`.
        """
        # Initial consistency checks.
        if k == 0:
            if ko is not None:
                raise KeyError("DAPPER convention: no obs at t=0. Helps avoid bugs.")
            if self._is_ens == True:
                if E is None:
                    raise TypeError("Expected ensemble input but E is None")
                if mu is not None:
                    raise TypeError("Expected ensemble input but mu/Cov is not None")
            else:
                if E is not None:
                    raise TypeError("Expected mu/Cov input but E is not None")
                if mu is None:
                    raise TypeError("Expected mu/Cov input but mu is None")

        # Default. Don't add more defaults. It just gets confusing.
        if faus is None:
            faus = 'u' if ko is None else 'au'

        # TODO 4: for faus="au" (e.g.) we don't need to re-**compute** stats,
        #         merely re-write them?
        for sub in faus:

            # Skip assessment if ('u' and stats not stored or plotted)
            if k != 0 and ko == None:
                if not (self.store_u or self.LP_instance.any_figs):
                    continue

            # Silence repeat warnings caused by zero variance
            with np.errstate(divide='call', invalid='call'):
                np.seterrcall(warn_zero_variance)

                # Assess
                stats_now = Avrgs()
                if self._is_ens:
                    self.assess_ens(stats_now, self.xx[k], E, w)
                else:
                    self.assess_ext(stats_now, self.xx[k], mu, Cov)
                self.derivative_stats(stats_now)
                self.summarize_marginals(stats_now)

            self.write(stats_now, k, ko, sub)

            # LivePlot -- Both init and update must come after the assessment.
            try:
                self.LP_instance.update((k, ko, sub), E, Cov)
            except AttributeError:
                self.LP_instance = liveplotting.LivePlot(
                    self, self.liveplots, (k, ko, sub), E, Cov)

    def write(self, stat_dict, k, ko, sub):
        """Write `stat_dict` to series at `(k, ko, sub)`."""
        for name, val in stat_dict.items():
            stat = struct_tools.deep_getattr(self, name)
            isFaust = isinstance(stat, series.FAUSt)
            stat[(k, ko, sub) if isFaust else ko] = val

    def summarize_marginals(self, now):
        """Compute Mean-field and RMS values."""
        formulae = {**self.field_summaries, **self.sector_summaries}

        with np.errstate(divide='ignore', invalid='ignore'):
            for stat in list(now):
                field = now[stat]
                for suffix, formula in formulae.items():
                    statpath = stat+'.'+suffix
                    if struct_tools.deep_hasattr(self, statpath):
                        now[statpath] = formula(field)

    def derivative_stats(self, now):
        """Stats that derive from others, and are not specific for `_ens` or `_ext`)."""
        try:
            now.gscore = 2*np.log(now.spread) + (now.err/now.spread)**2
        except AttributeError:
            # happens in case rc.comps['error_only']
            pass

    def assess_ens(self, now, x, E, w):
        """Ensemble and Particle filter (weighted/importance) assessment."""
        N, Nx = E.shape

        # weights
        if w is None:
            w = np.ones(N)/N  # All equal. Also, rm attr from stats:
            if hasattr(self, 'w'):
                delattr(self, 'w')
            # Use non-weight formula (since w=None) for mu computations.
            # The savings are noticeable when rc.comps['error_only'] is noticeable.
            now.mu = E.mean(0)
        else:
            now.w = w
            if abs(w.sum()-1) > 1e-5:
                raise RuntimeError("Weights did not sum to one.")
            now.mu = w @ E

        # Crash checks
        if not np.all(np.isfinite(E)):
            raise RuntimeError("Ensemble not finite.")
        if not np.all(np.isreal(E)):
            raise RuntimeError("Ensemble not Real.")

        # Compute errors
        now.err = now.mu - x
        if rc.comps['error_only']:
            return

        A = E - now.mu
        # While A**2 is approx as fast as A*A,
        # A**3 is 10x slower than A**2 (or A**2.0).
        # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
        # But, to save memory, only use A_pow.
        A_pow = A**2

        # Compute variances
        var  = w @ A_pow
        ub   = unbias_var(w, avoid_pathological=True)
        var *= ub

        # Compute standard deviation ("Spread")
        s = np.sqrt(var)  # NB: biased (even though var is unbiased)
        now.spread = s

        # For simplicity, use naive (biased) formulae, derived
        # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
        # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
        A_pow *= A
        now.skew = np.nanmean(w @ A_pow / (s*s*s))
        A_pow *= A
        now.kurt = np.nanmean(w @ A_pow / var**2 - 3)

        now.mad  = np.nanmean(w @ abs(A))

        if self.do_spectral:
            if N <= Nx:
                _, s, UT  = sla.svd((np.sqrt(w)*A.T).T, full_matrices=False)
                s        *= np.sqrt(ub)  # Makes s^2 unbiased
                now.svals = s
                now.umisf = UT @ now.err
            else:
                P         = (A.T * w) @ A
                s2, U     = sla.eigh(P)
                s2       *= ub
                now.svals = np.sqrt(s2.clip(0))[::-1]
                now.umisf = U.T[::-1] @ now.err

            # For each state dim [i], compute rank of truth (x) among the ensemble (E)
            E_x = np.sort(np.vstack((E, x)), axis=0, kind='heapsort')
            now.rh = np.asarray(
                [np.where(E_x[:, i] == x[i])[0][0] for i in range(Nx)])

    def assess_ext(self, now, x, mu, P):
        """Kalman filter (Gaussian) assessment."""
        if not np.all(np.isfinite(mu)):
            raise RuntimeError("Estimates not finite.")
        if not np.all(np.isreal(mu)):
            raise RuntimeError("Estimates not Real.")
        # Don't check the cov (might not be explicitly availble)

        # Compute errors
        now.mu  = mu
        now.err = now.mu - x
        if rc.comps['error_only']:
            return

        # Get diag(P)
        if P is None:
            var = np.zeros_like(mu)
        elif np.isscalar(P):
            var = np.ones_like(mu) * P
        else:
            if isinstance(P, CovMat):
                var = P.diag
                P   = P.full
            else:
                var = np.diag(P)

            if self.do_spectral:
                s2, U     = sla.eigh(P)
                now.svals = np.sqrt(np.maximum(s2, 0.0))[::-1]
                now.umisf = (U.T @ now.err)[::-1]

        # Compute stddev
        now.spread = np.sqrt(var)
        # Here, sqrt(2/pi) is the ratio, of MAD/Spread for Gaussians
        now.mad = np.nanmean(now.spread) * np.sqrt(2/np.pi)

    def average_in_time(self, kk=None, kko=None, free=False):
        """Avarage all univariate (scalar) time series.

        - `kk`    time inds for averaging
        - `kko` time inds for averaging obs
        """
        tseq = self.HMM.tseq
        if kk is None:
            kk     = tseq.mask
        if kko is None:
            kko  = tseq.masko

        def average1(tseries):
            avrgs = Avrgs()

            def average_multivariate(): return avrgs
            # Plain averages of nd-series are rarely interesting.
            # => Shortcircuit => Leave for manual computations

            if isinstance(tseries, series.FAUSt):
                # Average series for each subscript
                if tseries.item_shape != ():
                    return average_multivariate()
                for sub in [ch for ch in 'fas' if hasattr(tseries, ch)]:
                    avrgs[sub] = series.mean_with_conf(tseries[kko, sub])
                if tseries.store_u:
                    avrgs['u'] = series.mean_with_conf(tseries[kk, 'u'])

            elif isinstance(tseries, series.DataSeries):
                if tseries.array.shape[1:] != ():
                    return average_multivariate()
                elif len(tseries.array) == self.Ko+1:
                    avrgs = series.mean_with_conf(tseries[kko])
                elif len(tseries.array) == self.K+1:
                    avrgs = series.mean_with_conf(tseries[kk])
                else:
                    raise ValueError

            elif np.isscalar(tseries):
                avrgs = tseries  # Eg. just copy over "duration" from stats

            else:
                raise TypeError(f"Don't know how to average {tseries}")

            return avrgs

        def recurse_average(stat_parent, avrgs_parent):
            for key in getattr(stat_parent, "stat_register", []):
                try:
                    tseries = getattr(stat_parent, key)
                except AttributeError:
                    continue  # Eg assess_ens() deletes .weights if None
                avrgs = average1(tseries)
                recurse_average(tseries, avrgs)
                avrgs_parent[key] = avrgs

        avrgs = Avrgs()
        recurse_average(self, avrgs)
        self.xp.avrgs = avrgs
        if free:
            delattr(self.xp, 'stats')

    def replay(self, figlist="default", speed=np.inf, t1=0, t2=None, **kwargs):
        """Replay LivePlot with what's been stored in 'self'.

        - t1, t2: time window to plot.
        - 'figlist' and 'speed': See LivePlot's doc.

        .. note:: `store_u` (whether to store non-obs-time stats) must
        have been `True` to have smooth graphs as in the actual LivePlot.

        .. note:: Ensembles are generally not stored in the stats
        and so cannot be replayed.
        """
        # Time settings
        tseq = self.HMM.tseq
        if t2 is None:
            t2 = t1 + tseq.Tplot

        # Ens does not get stored in stats, so we cannot replay that.
        # If the LPs are initialized with P0!=None, then they will avoid ens plotting.
        # TODO 4: This system for switching from Ens to stats must be replaced.
        #       It breaks down when M is very large.
        try:
            P0 = np.full_like(self.HMM.X0.C.full, np.nan)
        except AttributeError:  # e.g. if X0 is defined via sampling func
            P0 = np.eye(self.HMM.Nx)

        LP = liveplotting.LivePlot(self, figlist, P=P0, speed=speed,
                                   Tplot=t2-t1, replay=True, **kwargs)
        plt.pause(.01)  # required when speed=inf

        # Remember: must use progbar to unblock read1.
        # Let's also make a proper description.
        desc = self.xp.da_method + " (replay)"

        # Play through assimilation cycles
        for k, ko, t, _dt in progbar(tseq.ticker, desc):
            if t1 <= t <= t2:
                if ko is not None:
                    LP.update((k, ko, 'f'), None, None)
                    LP.update((k, ko, 'a'), None, None)
                LP.update((k, ko, 'u'), None, None)

        # Pause required when speed=inf.
        # On Mac, it was also necessary to do it for each fig.
        if LP.any_figs:
            for _name, updater in LP.figures.items():
                if plt.fignum_exists(_name) and getattr(updater, 'is_active', 1):
                    plt.figure(_name)
                    plt.pause(0.01)


def register_stat(self, name, value):
    """Do `self.name = value` and register `name` as in self's `stat_register`.

    Note: `self` is not always a `Stats` object, but could be a "child" of it.
    """
    setattr(self, name, value)
    if not hasattr(self, "stat_register"):
        self.stat_register = []
    self.stat_register.append(name)


class Avrgs(series.StatPrint, struct_tools.DotDict):
    """A `dict` specialized for the averages of statistics.

    Embellishments:

    - `dapper.tools.StatPrint`
    - `Avrgs.tabulate`
    - `getattr` that supports abbreviations.
    """

    def tabulate(self, statkeys=(), decimals=None):
        columns = tabulate_avrgs([self], statkeys, decimals=decimals)
        return tabulate(columns, headers="keys").replace('␣', ' ')

    abbrevs = {'rmse': 'err.rms', 'rmss': 'spread.rms', 'rmv': 'spread.rms'}

    # Use getattribute coz it gets called before getattr.
    def __getattribute__(self, key):
        """Support deep and abbreviated lookup."""
        # key = abbrevs[key] # Instead of this, also support rmse.a:
        key = '.'.join(Avrgs.abbrevs.get(seg, seg) for seg in key.split('.'))

        if "." in key:
            return struct_tools.deep_getattr(self, key)
        else:
            return super().__getattribute__(key)

# In case of degeneracy, variance might be 0, causing warnings
# in computing skew/kurt/MGLS (which all normalize by variance).
# This should and will yield nan's, but we don't want mere diagnostics
# computations to cause repetitive warnings, so we only warn once.
#
# I would have expected this (more elegant solution?) to work,
# but it just makes it worse.
# with np.errstate(divide='warn',invalid='warn'), warnings.catch_warnings():
# warnings.simplefilter("once",category=RuntimeWarning)
# ...


@do_once
def warn_zero_variance(err, flag):
    msg = "\n".join(["Numerical error in stat comps.",
                     "Probably caused by a sample variance of 0."])
    warnings.warn(msg)


# Why not do all columns at once using the tabulate module? Coz
#  - Want subcolumns, including fancy formatting (e.g. +/-)
#  - Want separation (using '|') of attr and stats
#  - ...
def align_col(col, pad='␣', missingval='', just=">"):
    r"""Align column.

    Treats `int`s and fixed-point `float`/`str` especially, aligning on the point.

    Example:
    >>> xx = [1, 1., 1.234, 12.34, 123.4, "1.2e-3", None, np.nan, "inf", (1, 2)]
    >>> print(*align_col(xx), sep="\n")
    ␣␣1␣␣␣␣
    ␣␣1.0␣␣
    ␣␣1.234
    ␣12.34␣
    123.4␣␣
    ␣1.2e-3
    ␣␣␣␣␣␣␣
    ␣␣␣␣nan
    ␣␣␣␣inf
    ␣(1, 2)
    """
    def split_decimal(x):
        x = str(x)
        try:
            y = float(x)
        except ValueError:
            pass
        else:
            if np.isfinite(y) and ("e" not in x.lower()):
                a, *b = x.split(".")
                if b == []:
                    b = "int"
                else:
                    b = b[0]
                return a, b
        return x, False

    # Find max nInt, nDec
    nInt = nDec = -1
    for x in col:
        ints, decs = split_decimal(x)
        if decs:
            nInt = max(nInt, len(ints))
            if decs != "int":
                nDec = max(nDec, len(decs))

    # Format entries. Floats get aligned on point.
    def frmt(x):
        if x is None:
            return missingval
        ints, decs = split_decimal(x)
        x = f"{ints.rjust(nInt, pad)}"
        if decs == "int":
            if nDec >= 0:
                x += pad + pad*nDec
        elif decs:
            x += "." + f"{decs.ljust(nDec, pad)}"
        else:
            x = ints
        return x

    # Format
    col = [frmt(x) for x in col]
    # Find max width
    Max = max(len(x) for x in col)
    # Right-justify
    shift = str.rjust if just == ">" else str.ljust
    col = [shift(x, Max, pad) for x in col]
    return col


def unpack_uqs(uq_list, decimals=None):
    """Convert list of `uq`s into dict of lists (of equal-length) of attributes.

    The attributes are obtained by `vars(uq)`,
    and may get formatted somehow (e.g. cast to strings) in the output.

    If `uq` is `None`, then `None` is inserted in each list.
    Else, `uq` must be an instance of `dapper.tools.rounding.UncertainQtty`.

    Parameters
    ----------
    uq_list: list
        List of `uq`s.

    decimals: int
        Desired number of decimals.
        Used for (only) the columns "val" and "prec".
        Default: `None`. In this case, the formatting is left to the `uq`s.
    """
    def frmt(uq):
        if not isinstance(uq, series.UncertainQtty):
            # Presumably uq is just a number
            uq = series.UncertainQtty(uq)

        attrs = vars(uq).copy()

        # val/prec: round
        if decimals is None:
            v, p = str(uq).split(" ±")
        else:
            frmt = "%%.%df" % decimals
            v, p = frmt % uq.val, frmt % uq.prec
        attrs["val"], attrs["prec"] = v, p

        # tuned_coord: convert to tuple
        try:
            attrs["tuned_coord"] = tuple(a for a in uq.tuned_coord)
        except AttributeError:
            pass
        return attrs

    cols = {}
    for i, uq in enumerate(uq_list):
        if uq is not None:
            # Format
            attrs = frmt(uq)
            # Insert attrs as a "row" in the `cols`:
            for k in attrs:
                # Init column
                if k not in cols:
                    cols[k] = [None]*len(uq_list)
                # Insert element
                cols[k][i] = attrs[k]

    return cols


def tabulate_avrgs(avrgs_list, statkeys=(), decimals=None):
    """Tabulate avrgs (val±prec)."""
    if not statkeys:
        statkeys = ['rmse.a', 'rmv.a', 'rmse.f']

    columns = {}
    for stat in statkeys:
        column = [getattr(a, stat, None) for a in avrgs_list]
        column = unpack_uqs(column, decimals)
        if not column:
            raise ValueError(f"The stat. key '{stat}' was not"
                             " found among any of the averages.")
        vals  = align_col([stat] + column["val"])
        precs = align_col(['1σ'] + column["prec"], just="<")
        headr = vals[0]+'  '+precs[0]
        mattr = [f"{v} ±{c}" for v, c in zip(vals, precs)][1:]
        columns[headr] = mattr

    return columns


def center(E, axis=0, rescale=False):
    r"""Center ensemble.

    Makes use of `np` features: keepdims and broadcasting.

    Parameters
    ----------
    E: ndarray
        Ensemble which going to be inflated

    axis: int, optional
        The axis to be centered. Default: 0

    rescale: bool, optional
        If True, inflate to compensate for reduction in the expected variance.
        The inflation factor is \(\sqrt{\frac{N}{N - 1}}\)
        where N is the ensemble size. Default: False

    Returns
    -------
    X: ndarray
        Ensemble anomaly

    x: ndarray
        Mean of the ensemble
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze(axis=axis)

    return X, x


def mean0(E, axis=0, rescale=True):
    """Like `center`, but only return the anomalies (not the mean).

    Uses `rescale=True` by default, which is beneficial
    when used to center observation perturbations.
    """
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate the ensemble (center, inflate, re-combine).

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
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor


def weight_degeneracy(w, prec=1e-10):
    """Check if the weights are degenerate.

    If it is degenerate, the maximum weight
    should be nearly one since sum(w) = 1

    Parameters
    ----------
    w: ndarray
        Importance weights. Must sum to 1.

    prec: float, optional
        Tolerance of the distance between w and one. Default:1e-10

    Returns
    -------
    bool
        If weight is degenerate True, else False
    """
    return (1-w.max()) < prec


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """Compute unbias-ing factor for variance estimation.

    Parameters
    ----------
    w: ndarray, optional
        Importance weights. Must sum to 1.
        Only one of `w` and `N_eff` can be `None`. Default: `None`

    N_eff: float, optional
        The "effective" size of the weighted ensemble.
        If not provided, it is computed from the weights.
        The unbiasing factor is $$ N_{eff} / (N_{eff} - 1) $$.

    avoid_pathological: bool, optional
        Avoid weight collapse. Default: `False`

    Returns
    -------
    ub: float
        factor used to unbiasing variance

    Reference
    --------
    [Wikipedia](https://wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights)
    """
    if N_eff is None:
        N_eff = 1/(w@w)

    if avoid_pathological and weight_degeneracy(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1/(1 - 1/N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub
