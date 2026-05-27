"""Statistics for the assessment of DA methods.

[`Stats`][stats.Stats] records per-timestep statistics during assimilation and
exposes them as [`DACycleSeries`][tools.series.DACycleSeries] attributes.
The default statistics come in two shapes and sizes:

- **Vector stats** (`err`, `spread`, `mu`, …) — one value per state dim.
  Each has child scalar **field-summary** (`.rms`, `.m`, `.ma`, …)
- **Scalar stats** (`mad`, `skew`, `kurt`, `trHK`, …) — inherently scalar.

[`Avrgs`][stats.Avrgs] holds the same hierarchy *after time-averaging*.

--8<-- "dapper/stats_etc.md"
"""

import warnings

import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from scipy import special
from tabulate import tabulate

import dapper.tools.liveplotting as liveplotting
import dapper.tools.series as series
from dapper.dpr_config import rc
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar
from dapper.tools.rounding import UncertainQtty
from dapper.tools.struct import DotDict, deep_getattr, deep_hasattr, intersect


class DACycleAvrgs(series.StatPrint, DotDict):
    """Time-averaged scalar stat: one [`tools.rounding.UncertainQtty`][] per subscript.

    Only subscripts for which finite data exist are set as attributes.
    Inherits `DotDict` so the object is iterable and supports `in` tests.
    """

    # Declared for static analysis; populated by Stats.average_in_time().
    f: UncertainQtty
    """Forecast (prior) time average."""
    a: UncertainQtty
    """Analysis (posterior) time average."""
    s: UncertainQtty
    """Smoothed time average (smoothers only)."""
    i: UncertainQtty
    """Integrational (between-obs) time average."""


class FieldAvrgs(series.StatPrint, DotDict):
    """Time-averaged vector stat: one [`stats.DACycleAvrgs`][] per field summary.

    Only the field summaries that were actually computed are set as attributes.
    Inherits `DotDict` so the object is iterable and supports `in` tests.
    """

    # Declared for static analysis; populated by Stats.average_in_time().
    m: DACycleAvrgs
    """Mean-field average."""
    ms: DACycleAvrgs
    """Mean-square average."""
    rms: DACycleAvrgs
    """Root-mean-square average."""
    ma: DACycleAvrgs
    """Mean-absolute average."""
    gm: DACycleAvrgs
    """Geometric-mean average."""


class Stats(series.StatPrint):
    """Records and computes per-timestep statistics for a DA method.

    DA methods may register additional stats via `self.stats.register(name, value)`
    for the purpose of automatic plotting and averaging.
    """

    # Declared for static analysis; created dynamically via new_series() in __init__.
    mu: series.DACycleSeries
    """Mean estimate (ensemble mean, or Kalman/variational estimate)."""
    spread: series.DACycleSeries
    """Ensemble spread (std dev), or Kalman posterior std dev."""
    err: series.DACycleSeries
    """Error of the mean estimate vs. truth (`mu - x`)."""
    gscore: series.DACycleSeries
    """Gaussian (log) score: `2 log(spread) + (err/spread)²`."""
    crps: series.DACycleSeries
    """Continuous ranked probability score."""
    mad: series.DACycleSeries
    """Mean absolute deviation of the ensemble from its mean."""
    skew: series.DACycleSeries
    """Skewness of the ensemble."""
    kurt: series.DACycleSeries
    """Excess kurtosis of the ensemble (0 for Gaussians)."""
    # Ensemble-only (created only when xp has N):
    w: series.DACycleSeries
    """Importance weights (particle filter; absent when weights are uniform)."""
    rh: series.DACycleSeries
    """Rank histogram: rank of the truth among sorted ensemble members."""
    svals: series.DACycleSeries
    """Singular values (principal-component scores) of the ensemble anomalies."""
    umisf: series.DACycleSeries
    """Error projected onto the leading ensemble/covariance directions."""
    # Analysis-only:
    trHK: series.DACycleSeries
    """Trace of the observation-space gain matrix `HK`."""
    infl: series.DACycleSeries
    """Inflation factor applied at this analysis step."""
    iters: series.DACycleSeries
    """Number of iterations (iterative methods only)."""
    N_eff: series.DACycleSeries
    """Effective ensemble size `1 / Σwᵢ²`."""
    wroot: series.DACycleSeries
    """Root-finding output for optimal weight tempering (particle filter)."""
    resmpl: series.DACycleSeries
    """Resampling flag/count at this analysis step."""
    # Added by da_method wrapper after assimilate() returns:
    duration: float
    """Wall-clock time (seconds) for the full `assimilate()` call."""

    def __init__(self, xp, HMM, xx, yy, liveplots=False, store_i=rc.store_i):
        """Init the default statistics."""
        ######################################
        # Preamble
        ######################################
        self.xp = xp
        self.HMM = HMM
        self.xx = xx
        self.yy = yy
        self.liveplots = liveplots
        self.store_i = store_i
        self._stat_names: list[str] = []
        # True for smoothers, which write to the 's' (smoothed) subscript.
        self.store_s = any(hasattr(xp, key) for key in ("Lag", "DeCorr"))

        # Shapes
        K = xx.shape[0] - 1
        Nx = xx.shape[1]
        Ko = yy.shape[0] - 1
        self.K, self.Ko, self.Nx = K, Ko, Nx

        # Methods for summarizing multivariate stats ("fields") as scalars
        # Don't use nanmean here; nan's should get propagated!
        en_mean = lambda x: np.mean(x, axis=0)  # noqa
        self.field_summaries = dict(
            m=lambda x: en_mean(x),  # mean-field
            ms=lambda x: en_mean(x**2),  # root-mean-square
            rms=lambda x: np.sqrt(en_mean(x**2)),  # root-mean-square
            ma=lambda x: en_mean(np.abs(x)),  # mean-absolute
            gm=lambda x: np.exp(en_mean(np.log(x))),  # geometric mean
        )
        # Only keep the methods listed in rc
        self.field_summaries = intersect(self.field_summaries, rc.field_summaries)

        # Define similar methods, but restricted to sectors
        self.sector_summaries = {}

        def restrict(fun, inds):
            return lambda x: fun(x[inds])

        for suffix, formula in self.field_summaries.items():
            for sector, inds in HMM.sectors.items():
                f = restrict(formula, inds)
                self.sector_summaries[f"{suffix}.{sector}"] = f

        ######################################
        # Allocate time series of various stats
        ######################################
        self.new_series("mu", Nx, field_mean="sectors")  # Mean
        self.new_series("spread", Nx, field_mean="sectors")  # Std. dev. ("spread")
        self.new_series("err", Nx, field_mean="sectors")  # Error (mu - truth)
        self.new_series("gscore", Nx, field_mean="sectors")  # Gaussian (log) score
        self.new_series("crps", Nx, field_mean="sectors")  # Cont. ranked prob. score

        # To save memory, we only store these field means:
        self.new_series("mad", 1)  # Mean abs deviations
        self.new_series("skew", 1)  # Skewness
        self.new_series("kurt", 1)  # Kurtosis

        if hasattr(xp, "N"):
            N = xp.N
            self.new_series("w", N, field_mean=True)  # Importance weights
            self.new_series("rh", Nx, dtype=int)  # Rank histogram

            self._is_ens = True
            minN = min(Nx, N)
            self.do_spectral = np.sqrt(Nx * N) <= rc.comps["max_spectral"]
        else:
            self._is_ens = False
            minN = Nx
            self.do_spectral = Nx <= rc.comps["max_spectral"]

        if self.do_spectral:
            # Note: the mean-field and RMS time-series of
            # (i) svals and (ii) umisf should match the corresponding series of
            # (i) spread and (ii) err.
            self.new_series("svals", minN)  # Principal component (SVD) scores
            self.new_series("umisf", minN)  # Error in component directions

        ######################################
        # Allocate a few series for outside use
        ######################################
        self.new_series("trHK", 1, analysis_only=True)
        self.new_series("infl", 1, analysis_only=True)
        self.new_series("iters", 1, analysis_only=True)

        # Weight-related
        self.new_series("N_eff", 1, analysis_only=True)
        self.new_series("wroot", 1, analysis_only=True)
        self.new_series("resmpl", 1, analysis_only=True)

    def new_series(self, name, shape, analysis_only=False, field_mean=False, **kws):
        """Create (and register) a statistics time series, initialized with `nan`s.

        Creates a [tools.series.DACycleSeries][]. If `analysis_only=True`, the
        series has no `.f` sub-array and `.i` is a ring buffer of size 1.

        NB: The `sliding_diagnostics` liveplotting relies on detecting `nan`'s
            to avoid plotting stats that are not being used.
            Thus, you cannot use `dtype=bool` or `int` for stats that get plotted.
        """
        # Convert int shape to tuple
        if not hasattr(shape, "__len__"):
            if shape == 1:
                shape = ()
            else:
                shape = (shape,)

        def make_series(parent, name, shape):
            tseries = series.DACycleSeries(
                self.K,
                self.Ko,
                shape,
                store_i=not analysis_only and self.store_i,
                store_s=not analysis_only and self.store_s,
                store_f=not analysis_only,
                **kws,
            )
            register(parent, name, tseries)

        # Principal series
        make_series(self, name, shape)

        # Summary (scalar) series:
        if shape != ():
            if field_mean:
                for suffix in self.field_summaries:
                    make_series(getattr(self, name), suffix, ())
            # Make a nested level for sectors
            if field_mean == "sectors":
                for ss in self.sector_summaries:
                    suffix, sector = ss.split(".")
                    make_series(deep_getattr(self, f"{name}.{suffix}"), sector, ())

    def assess(
        self,
        k: int,
        ko: int | None = None,
        fais: str | None = None,
        E: np.ndarray | None = None,
        w: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        Cov: np.ndarray | None = None,
    ) -> None:
        """Common interface for both [`Stats`.assess_ens][stats.Stats.assess_ens]
        and [`Stats`.assess_ext][stats.Stats.assess_ext].

        The `_ens` assessment function gets called if `E is not None`,
        and `_ext` if `mu is not None`.

        fais: One or more of `['f', 'a', 'i', 's']`, indicating
              that the result should be stored in (respectively)
              the forecast/analysis/integrational/smoothed attribute.
              Default: `'i' if ko is None else 'ai' ('a' and 'i')`.
        """
        # Initial consistency checks.
        if k == 0:
            if ko is not None:
                raise KeyError("DAPPER convention: no obs at t=0. Helps avoid bugs.")
            if self._is_ens:
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
        if fais is None:
            fais = "i" if ko is None else "ai"

        # TODO 4: for fais="ai" (e.g.) we don't need to re-**compute** stats,
        #         merely re-write them?
        for sub in fais:
            # Skip assessment if ('i' and stats not stored or plotted)
            if k != 0 and ko is None:
                if not (self.store_i or self.LP_instance.any_figs):
                    continue

            # Silence repeat warnings caused by zero variance
            with np.errstate(divide="call", invalid="call"):
                np.seterrcall(warn_zero_variance)

                # Assess
                stats_now = DotDict()
                if self._is_ens:
                    self.assess_ens(stats_now, self.xx[k], E, w)
                else:
                    self.assess_ext(stats_now, self.xx[k], mu, Cov)
                self.derivative_stats(stats_now)
                self.summarize_marginals(stats_now)

            self.write(stats_now, k, ko, sub)

            # LivePlot — both init and update must come after the assessment.
            try:
                self.LP_instance.update((k, ko, sub), E, Cov)
            except AttributeError:
                self.LP_instance = liveplotting.LivePlot(
                    self, self.liveplots, (k, ko, sub), E, Cov
                )

    def write(self, stat_dict, k, ko, sub):
        """Write `stat_dict` to series at `(k, ko, sub)`."""
        for name, val in stat_dict.items():
            stat = deep_getattr(self, name)
            if isinstance(stat, series.DACycleSeries):
                ind = (k if stat.store_i else 0) if sub == "i" else ko
                if hasattr(stat, sub):
                    getattr(stat, sub)[ind] = val
            else:
                stat[ko] = val

    def summarize_marginals(self, now):
        """Compute Mean-field and RMS values."""
        formulae = {**self.field_summaries, **self.sector_summaries}

        with np.errstate(divide="ignore", invalid="ignore"):
            for stat in list(now):
                field = now[stat]
                for suffix, formula in formulae.items():
                    statpath = stat + "." + suffix
                    if deep_hasattr(self, statpath):
                        now[statpath] = formula(field)

    def derivative_stats(self, now):
        """Stats that derive from others, and are not specific for `_ens` or `_ext`)."""
        try:
            now.gscore = 2 * np.log(now.spread) + (now.err / now.spread) ** 2
        except AttributeError:
            # happens in case rc.comps['error_only']
            pass

    def assess_ens(self, now, x, E, w):
        """Ensemble and Particle filter (weighted/importance) assessment."""
        N, Nx = E.shape

        # weights
        if w is None:
            w = np.ones(N) / N  # All equal. Also, rm attr from stats:
            if hasattr(self, "w"):
                delattr(self, "w")
            # Use non-weight formula (since w=None) for mu computations.
            # The savings are noticeable when rc.comps['error_only'] is noticeable.
            now.mu = E.mean(0)
        else:
            now.w = w
            if abs(w.sum() - 1) > 1e-5:
                raise RuntimeError("Weights did not sum to one.")
            now.mu = w @ E

        # Crash checks
        if not np.all(np.isfinite(E)):
            raise RuntimeError("Ensemble not finite.")
        if not np.all(np.isreal(E)):
            raise RuntimeError("Ensemble not Real.")

        # Compute errors
        now.err = now.mu - x
        if rc.comps["error_only"]:
            return

        now.crps = crps_ens(x, E, w)

        A = E - now.mu
        # While A**2 is approx as fast as A*A,
        # A**3 is 10x slower than A**2 (or A**2.0).
        # => Use A2 = A**2, A3 = A*A2, A4=A*A3.
        # But, to save memory, only use A_pow.
        A_pow = A**2

        # Compute variances
        var = w @ A_pow
        ub = unbias_var(w, avoid_pathological=True)
        var *= ub

        # Compute standard deviation ("Spread")
        s = np.sqrt(var)  # NB: biased (even though var is unbiased)
        now.spread = s

        # For simplicity, use naive (biased) formulae, derived
        # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
        # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
        A_pow *= A
        now.skew = np.nanmean(w @ A_pow / (s * s * s))
        A_pow *= A
        now.kurt = np.nanmean(w @ A_pow / var**2 - 3)

        now.mad = np.nanmean(w @ abs(A))

        if self.do_spectral:
            if N <= Nx:
                _, s, UT = sla.svd((np.sqrt(w) * A.T).T, full_matrices=False)
                s *= np.sqrt(ub)  # Makes s^2 unbiased
                now.svals = s
                now.umisf = UT @ now.err
            else:
                P = (A.T * w) @ A
                s2, U = sla.eigh(P)
                s2 *= ub
                now.svals = np.sqrt(s2.clip(0))[::-1]
                now.umisf = U.T[::-1] @ now.err

            # For each state dim [i], compute rank of truth (x) among the ensemble (E)
            E_x = np.sort(np.vstack((E, x)), axis=0, kind="heapsort")
            now.rh = np.asarray([np.where(E_x[:, i] == x[i])[0][0] for i in range(Nx)])

    def assess_ext(self, now, x, mu, P):
        """Kalman filter (Gaussian) assessment."""
        if not np.all(np.isfinite(mu)):
            raise RuntimeError("Estimates not finite.")
        if not np.all(np.isreal(mu)):
            raise RuntimeError("Estimates not Real.")
        # Don't check the cov (might not be explicitly availble)

        # Compute errors
        now.mu = mu
        now.err = now.mu - x
        if rc.comps["error_only"]:
            return

        # Get diag(P)
        if P is None:
            var = np.zeros_like(mu)
        elif np.isscalar(P):
            var = np.ones_like(mu) * P
        else:
            if isinstance(P, CovMat):
                var = P.diag
                P = P.full
            else:
                var = np.diag(P)

            if self.do_spectral:
                s2, U = sla.eigh(P)
                now.svals = np.sqrt(np.maximum(s2, 0.0))[::-1]
                now.umisf = (U.T @ now.err)[::-1]

        now.crps = crps_ext(x, mu, var)

        # Compute stddev
        now.spread = np.sqrt(var)
        # Here, sqrt(2/pi) is the ratio, of MAD/Spread for Gaussians
        now.mad = np.nanmean(now.spread) * np.sqrt(2 / np.pi)

    def average_in_time(
        self,
        kk: np.ndarray | None = None,
        kko: np.ndarray | None = None,
        free: bool = False,
    ) -> None:
        """Average all scalar time series, producing `xp.avrgs`.

        Parameters
        ----------
        kk:
            Model-step indices to include when averaging the `i` (integrational)
            subscript.  Defaults to `tseq.mask` (post-burnin).
        kko:
            Obs-time indices for the `f`/`a`/`s` subscripts.
            Defaults to `tseq.masko`.
        free:
            If `True`, delete `xp.stats` after averaging to free memory.
        """
        tseq = self.HMM.tseq
        kk = kk if kk is not None else tseq.mask
        kko = kko if kko is not None else tseq.masko

        def avg(tseries):
            """Recursively average a DACycleSeries (or copy a scalar)."""
            if np.isscalar(tseries):
                return tseries  # Not a time series (e.g. duration) ⇒ copy directly

            if not isinstance(tseries, series.DACycleSeries):
                raise TypeError(f"Expected DACycleSeries but got {type(tseries)}")

            # Average DACycleSeries.f/a/s/i
            if tseries.item_shape == ():
                # Is a scalar (univariate) time series
                result = DACycleAvrgs()
                for sub in "fasi":
                    arr = getattr(tseries, sub, None)
                    if arr is None:
                        continue
                    if sub == "i" and not tseries.store_i:
                        continue  # ring buffer — not stored over time
                    vals = arr[kk if sub == "i" else kko]
                    if np.any(np.isfinite(vals)):
                        result[sub] = series.mean_with_conf(vals)
            else:
                # Is a vector (multivariate) time series: don't average because usually
                # not very interesting (user can do themselves if stats not "free"d),
                # but init the namespace for the sake of its children (field summaries).
                result: FieldAvrgs = FieldAvrgs()

            # Recurse into children (field summaries and sector sub-stats).
            for name in getattr(tseries, "_stat_names", []):
                child = getattr(tseries, name, None)
                if child is not None:
                    result[name] = avg(child)
            return result

        # Init recursion
        avrgs = Avrgs()
        for name in self._stat_names:
            tseries = getattr(self, name, None)
            if tseries is None:
                continue  # e.g. assess_ens() deletes .w when weights are uniform
            avrgs[name] = avg(tseries)

        self.xp.avrgs = avrgs
        if free:
            delattr(self.xp, "stats")

    def replay(self, figlist="default", speed=np.inf, t1=0, t2=None, **kwargs):
        """Replay LivePlot with what's been stored in 'self'.

        - t1, t2: time window to plot.
        - 'figlist' and 'speed': See LivePlot's doc.

        !!! note
            `store_i` (whether to store non-obs-time stats) must
            have been `True` to have smooth graphs as in the actual LivePlot.

        !!! note
            Ensembles are generally not stored in the stats and so cannot be replayed.
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

        LP = liveplotting.LivePlot(
            self, figlist, P=P0, speed=speed, Tplot=t2 - t1, replay=True, **kwargs
        )

        # Remember: must use progbar to unblock read1.
        # Let's also make a proper description.
        desc = self.xp.da_method + " (replay)"

        # Play through assimilation cycles
        for k, ko, t, _dt in progbar(tseq.ticker, desc):
            if t1 <= t <= t2:
                if ko is not None:
                    LP.update((k, ko, "f"), None, None)
                    LP.update((k, ko, "a"), None, None)
                LP.update((k, ko, "i"), None, None)

        # Pause required when speed=inf.
        # On Mac, it was also necessary to do it for each fig.
        if LP.any_figs:
            for _name, updater in LP.figures.items():
                if plt.fignum_exists(_name) and getattr(updater, "is_active", 1):
                    plt.figure(_name)
                    plt.pause(0.01)


def register(self, name, value):
    """Do `self.name = value` and register `name` in `self._stat_names`.

    Note: `self` is not always a `Stats` object, but could be a "child" of it
    (e.g. a DACycleSeries). Child objects get `_stat_names` lazily.
    """
    setattr(self, name, value)
    if not hasattr(self, "_stat_names"):
        self._stat_names: list[str] = []
    self._stat_names.append(name)


Stats.register = register


class Avrgs(series.StatPrint, DotDict):
    """Time-averaged statistics produced by [`stats.Stats.average_in_time`][].

    Attributes get populated by `Stats.average_in_time()`.

    Vector stats time series (e.g. `err`, `spread`) first get reduced to
    scalar time series ([`stats.FieldAvrgs`][] objects).
    Scalar time series (including those inherently so, e.g. `mad`, `trHK`)
    get reduced to [`stats.DACycleAvrgs`][] objects.

    Supports:

    - Deep attribute access: `avrgs.err.rms.a`
    - Dotted-key `getattr`: `getattr(avrgs, "err.rms.a")`
    - Named abbreviations (as properties): `avrgs.rmse`  →  `avrgs.err.rms`
    - Tabulation: `avrgs.tabulate(["rmse.a", "rmv.a"])`
    - Dict-like iteration and `in` tests (inherited from `DotDict`)
    """

    # Declared for static analysis; populated by Stats.average_in_time().
    err: FieldAvrgs
    """Time-averaged error field (and its scalar summaries)."""
    spread: FieldAvrgs
    """Time-averaged spread field (and its scalar summaries)."""
    mu: FieldAvrgs
    """Time-averaged mean-estimate field (and its scalar summaries)."""
    gscore: FieldAvrgs
    """Time-averaged Gaussian score field (and its scalar summaries)."""
    crps: FieldAvrgs
    """Time-averaged CRPS field (and its scalar summaries)."""
    mad: DACycleAvrgs
    """Time-averaged mean absolute deviation."""
    skew: DACycleAvrgs
    """Time-averaged skewness."""
    kurt: DACycleAvrgs
    """Time-averaged excess kurtosis."""
    w: FieldAvrgs
    """Time-averaged importance weights (and their scalar summaries)."""
    svals: FieldAvrgs
    """Time-averaged singular values (and their scalar summaries)."""
    umisf: FieldAvrgs
    """Time-averaged projected error (and its scalar summaries)."""
    trHK: DACycleAvrgs
    """Time-averaged trace of `HK`."""
    infl: DACycleAvrgs
    """Time-averaged inflation factor."""
    iters: DACycleAvrgs
    """Time-averaged iteration count."""
    N_eff: DACycleAvrgs
    """Time-averaged effective ensemble size."""
    wroot: DACycleAvrgs
    """Time-averaged weight-tempering root."""
    resmpl: DACycleAvrgs
    """Time-averaged resampling count."""
    duration: float
    """Wall-clock time (seconds) for the full `assimilate()` call."""

    def __getattr__(self, key: str):
        """Forward dotted-key lookups such as `getattr(avrgs, "err.rms.a")`.

        Called only when normal lookup fails.  If `key` contains a dot,
        resolve it as a chain of attribute accesses (which lets the
        abbreviation properties — `rmse`, etc. — participate naturally).
        Plain missing keys raise `AttributeError` immediately to avoid
        infinite recursion.
        """
        if "." not in key:
            raise AttributeError(f"'Avrgs' object has no attribute {key!r}")
        try:
            return deep_getattr(self, key)
        except AttributeError:
            raise AttributeError(f"'Avrgs' object has no attribute {key!r}") from None

    rmse = property(lambda self: self.err.rms, doc="Alias for `err.rms`")
    rmss = property(lambda self: self.spread.rms, doc="Alias for `spread.rms`.")
    rmv = property(lambda self: self.spread.rms, doc="Alias for `spread.rms`.")

    def tabulate(self, statkeys=(), decimals=None):
        """Tabulate using [`tabulate_avrgs`][stats.tabulate_avrgs]."""
        columns = tabulate_avrgs([self], statkeys, decimals=decimals)
        return tabulate(columns, headers="keys").replace("␣", " ")


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


# Decorator to ensure a function runs at most once; subsequent calls are no-ops.
def do_once(fun):
    done = False

    def wrapper(*args, **kwargs):
        nonlocal done
        if not done:
            done = True
            return fun(*args, **kwargs)

    return wrapper


@do_once
def warn_zero_variance(err, flag):
    msg = "\n".join(
        ["Numerical error in stat comps.", "Probably caused by a sample variance of 0."]
    )
    warnings.warn(msg, stacklevel=2)


# Why not do all columns at once using the tabulate module? Coz
#  - Want subcolumns, including fancy formatting (e.g. +/-)
#  - Want separation (using '|') of attr and stats
#  - ...
def align_col(col, pad="␣", missingval="", just=">"):
    r"""Align column.

    Treats `int`s and fixed-point `float`/`str` especially, aligning on the point.

    Examples
    --------
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
        if isinstance(x, tuple):
            x = tuple(np2builtin(v) for v in x)
        ints, decs = split_decimal(x)
        x = f"{ints.rjust(nInt, pad)}"
        if decs == "int":
            if nDec >= 0:
                x += pad + pad * nDec
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
    Else, `uq` must be an instance of [`tools.rounding.UncertainQtty`][].

    Parameters
    ----------
    uq_list : list
        List of `uq`s.

    decimals : int
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
            frmt = f"%.{decimals:d}f"
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
                    cols[k] = [None] * len(uq_list)
                # Insert element
                cols[k][i] = attrs[k]

    return cols


def tabulate_avrgs(avrgs_list, statkeys=(), decimals=None):
    """Tabulate avrgs (val±prec)."""
    if not statkeys:
        statkeys = ["rmse.a", "rmv.a", "rmse.f"]

    columns = {}
    for stat in statkeys:
        column = [getattr(a, stat, None) for a in avrgs_list]
        column = unpack_uqs(column, decimals)
        if not column:
            raise ValueError(
                f"The stat. key '{stat}' was not found among any of the averages."
            )
        vals = align_col([stat] + column["val"])
        precs = align_col(["1σ"] + column["prec"], just="<")
        headr = vals[0] + "  " + precs[0]
        mattr = [f"{v} ±{c}" for v, c in zip(vals, precs)][1:]
        columns[headr] = mattr

    return columns


def np2builtin(v):
    "Sometimes necessary since NEP-50"
    return v.item() if isinstance(v, np.generic) else v


def center(E, axis=0, rescale=False):
    r"""Center ensemble.

    Makes use of `np` features: keepdims and broadcasting.

    Parameters
    ----------
    E : ndarray
        Ensemble which going to be inflated

    axis : int, optional
        The axis to be centered. Default: 0

    rescale : bool, optional
        If True, inflate to compensate for reduction in the expected variance.
        The inflation factor is \(\sqrt{\frac{N}{N - 1}}\)
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
        X *= np.sqrt(N / (N - 1))

    x = x.squeeze(axis=axis)

    return X, x


def mean0(E, axis=0, rescale=True):
    """Like [`center`][stats.center], but only return the anomalies (not the mean).

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

    factor : `float`
        Inflation factor

    Returns
    -------
    ndarray
        Inflated ensemble
    """
    if factor == 1:
        return E
    X, x = center(E)
    return x + X * factor


def weight_degeneracy(w, prec=1e-10):
    """Check if the weights are degenerate.

    If it is degenerate, the maximum weight
    should be nearly one since sum(w) = 1

    Parameters
    ----------
    w : ndarray
        Importance weights. Must sum to 1.

    prec : float, optional
        Tolerance of the distance between w and one. Default:1e-10

    Returns
    -------
    bool
        If weight is degenerate True, else False
    """
    return (1 - w.max()) < prec


def unbias_var(w=None, N_eff=None, avoid_pathological=False):
    """Compute unbias-ing factor for variance estimation.

    Parameters
    ----------
    w : ndarray, optional
        Importance weights. Must sum to 1.
        Only one of `w` and `N_eff` can be `None`. Default: `None`

    N_eff : float, optional
        The "effective" size of the weighted ensemble.
        If not provided, it is computed from the weights.
        The unbiasing factor is $$ N_{eff} / (N_{eff} - 1) $$.

    avoid_pathological : bool, optional
        Avoid weight collapse. Default: `False`

    Returns
    -------
    ub : float
        factor used to unbiasing variance

    Reference
    --------
    [Wikipedia](https://wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights)
    """
    if N_eff is None:
        N_eff = 1 / (w @ w)

    if avoid_pathological and weight_degeneracy(w):
        ub = 1  # Don't do in case of weights collapse
    else:
        ub = 1 / (1 - 1 / N_eff)  # =N/(N-1) if w==ones(N)/N.
    return ub


def crps_ens(x, ensemble, weights=None):
    """Compute CRPS for `ensemble` given obs/truth `x`.

    Tested to reproduce values from `properscoring.crps_ensemble()`.

    The 0th axis of `ensemble` is taken to enumerate the members,
    and must have same length as the 1d `weights` (if any)
    If `x.ndim == ensemble.ndim`, the 0th axis of `x` is taken to enumerate multiple x.
    The CRPS is computed independently (and efficiently) for any/all other dimensions.
    Thus `ensemble.shape[1:]` must be matched by `x.shape[1:]` or `x.shape`,
    and the output gets the shape of `x`.

    Examples
    --------

    In 1D:

    >>> ens = np.array([-1.5, -1.0, 1.0, 1.5])
    >>> crps_ens(0, ens)
    array(0.5625)

    >>> crps_ens([0], ens)
    array([0.5625])

    >>> crps_ens([-2, -1.5, 0, 1, 1.5, 2], ens)
    array([1.3125, 0.8125, 0.5625, 0.5625, 0.8125, 1.3125])

    In 2D:

    >>> ens2 = np.vstack([ens, ens]).T
    >>> crps_ens([1, 2], ens2)
    array([0.5625, 1.3125])

    >>> crps_ens([[1, 2]], ens2)
    array([[0.5625, 1.3125]])

    >>> crps_ens([[1, 2], [-2, -1.5]], ens2)
    array([[0.5625, 1.3125],
           [1.3125, 0.8125]])

    Try weighting:

    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng(3000)
    >>> ens = rng.standard_normal(10**3)
    >>> grd = np.linspace(-5, 5, num=len(ens))
    >>> a1 = crps_ens(0, ens)
    >>> a2 = crps_ens(0, ens, weights=norm.pdf(grd))
    >>> np.allclose(a1, a2, atol=1e-2)
    True
    """

    # Setup weights
    N = len(ensemble)
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights)
        assert len(weights) == N and np.all(weights >= 0)
    weights = weights / weights.sum()

    # Setup ndim/shape of x, ensemble
    ensemble = np.asarray(ensemble)
    x = np.asarray(x)
    shp = x.shape
    if ensemble.ndim - x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    assert ensemble.ndim == x.ndim, "Dimensions mismatch"

    # Add "ghost" (weight 0) member(s) to ensemble, which does not change its CDF,
    # but avoids having to compute partial quadrature bins around location(s) of x.
    ens = np.concatenate([ensemble, x])
    w = np.pad(weights, (0, len(x)))

    # Construct empirical CDF
    order = np.argsort(ens, axis=0)  # has (average) complexity O(N log N), per dim
    w = w[order]
    ens = np.take_along_axis(ens, order, axis=0)
    cdf = np.cumsum(w, axis=0)

    # Integrate
    dxs = np.diff(ens, axis=0)
    ens = ens[:-1]  # cdf[i] applies for "bin" from ens[i] to ens[i+1] ⇒ discard [-1]
    cdf = cdf[:-1]

    x = x[:, None]  # expand_dims
    heaviside = np.where(x > ens, 0, 1)
    integrand = (heaviside - cdf) ** 2
    q = np.sum(dxs * integrand, axis=1).reshape(shp)
    return q


def crps_ext(x, mu, var):
    """Adapted from `properscoring.crps_gaussian()`.

    The shapes of `x`, `mu`, and `var` must match, but can be anything,
    and yields output of same shape.

    Ref `http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html`
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    s1 = np.sqrt(np.asarray(var))
    z = (x - mu) / s1  # standardize
    pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-(z * z) / 2)
    cdf = special.ndtr(z)
    pi_inv = 1.0 / np.sqrt(np.pi)
    return s1 * (z * (2 * cdf - 1) + 2 * pdf - pi_inv)
