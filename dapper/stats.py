"""Stats computation for the assessment of DA methods."""

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
from dapper.tools.series import DataSeries, StatPrint


class Stats(StatPrint):
    """Contains and computes statistics of the DA methods.

    Use new_series() to register your own stat time series.
    """

    def __init__(self, xp, HMM, xx, yy, liveplots=False, store_u=rc.store_u):
        """Init the default statistics.

        Note: Python allows dynamically creating attributes, so you can easily
        add custom stat. series to a Stat instance within a particular method,
        for example. Use ``new_series`` to get automatic averaging too.
        """

        ######################################
        # Preamble
        ######################################
        self.xp        = xp
        self.HMM       = HMM
        self.xx        = xx
        self.yy        = yy
        self.liveplots = liveplots
        self.store_u   = store_u
        self.store_s   = hasattr(xp, 'Lag')

        # Shapes
        K    = xx.shape[0]-1
        Nx   = xx.shape[1]
        KObs = yy.shape[0]-1
        Ny   = yy.shape[1]
        self.K,    self.Nx = K, Nx
        self.KObs, self.Ny = KObs, Ny

        # Methods for summarizing multivariate stats ("fields") as scalars
        # Don't use nanmean here; nan's should get propagated!
        self.field_summaries = dict(
            m   = lambda x: np.mean(x),                  # mean-field
            rms = lambda x: np.sqrt(np.mean(x**2)),      # root-mean-square
            ma  = lambda x: np.mean(np.abs(x)),          # mean-absolute
            gm  = lambda x: np.exp(np.mean(np.log(x))),  # geometric mean
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
        self.new_series('mu',     Nx, MS='sec')  # Mean
        self.new_series('std',    Nx, MS='sec')  # Std. dev. ("spread")
        self.new_series('err',    Nx, MS='sec')  # Error (mu - truth)
        self.new_series('gscore', Nx, MS='sec')  # Gaussian (log) score

        # To save memory, we only store these field means:
        self.new_series('mad',  1)  # Mean abs deviations
        self.new_series('skew', 1)  # Skewness
        self.new_series('kurt', 1)  # Kurtosis

        if hasattr(xp, 'N'):
            N            = xp.N
            self.new_series('w', N, MS=True)    # Importance weights
            self.new_series('rh', Nx, dtype=int)  # Rank histogram

            self._is_ens = True
            minN         = min(Nx, N)
            do_spectral  = np.sqrt(Nx*N) <= rc.comp_threshold_b
        else:
            self._is_ens = False
            minN         = Nx
            do_spectral  = Nx <= rc.comp_threshold_b

        if do_spectral:
            # Note: the mean-field and RMS time-series of
            # (i) svals and (ii) umisf should match the corresponding series of
            # (i) std and (ii) err.
            self.new_series('svals', minN)  # Principal component (SVD) scores
            self.new_series('umisf', minN)  # Error in component directions

        ######################################
        # Allocate a few series for outside use
        ######################################
        self.new_series('trHK',  1, KObs+1)
        self.new_series('infl',  1, KObs+1)
        self.new_series('iters', 1, KObs+1)

        # Weight-related
        self.new_series('N_eff',  1, KObs+1)
        self.new_series('wroot',  1, KObs+1)
        self.new_series('resmpl', 1, KObs+1)

    def new_series(self, name, shape, length='FAUSt', MS=False, **kws):
        """Create (and register) a statistics time series.

        Series are initialized with nan's.

        Example: Create ndarray of length KObs+1 for inflation time series:
        >>> self.new_series('infl', 1, KObs+1)

        NB: The ``sliding_diagnostics`` liveplotting relies on detecting ``nan``'s
            to avoid plotting stats that are not being used.
            => Cannot use ``dtype=bool`` or ``int`` for stats that get plotted.
        """

        # Convert int shape to tuple
        if not hasattr(shape, '__len__'):
            if shape == 1:
                shape = ()
            else:
                shape = (shape,)

        def make_series(parent, name, shape):
            if length == 'FAUSt':
                total_shape = self.K, self.KObs, shape
                store_opts = self.store_u, self.store_s
                tseries = series.FAUSt(*total_shape, *store_opts, **kws)
            else:
                total_shape = (length,)+shape
                tseries = DataSeries(total_shape, *kws)
            register_stat(parent, name, tseries)

        # Principal series
        make_series(self, name, shape)

        # Summary (scalar) series:
        if shape != ():
            if MS:
                for suffix in self.field_summaries:
                    make_series(getattr(self, name), suffix, ())
            # Make a nested level for sectors
            if MS == 'sec':
                for ss in self.sector_summaries:
                    suffix, sector = ss.split('.')
                    make_series(struct_tools.deep_getattr(
                        self, f"{name}.{suffix}"), sector, ())

    @property
    def data_series(self):
        return [k for k in vars(self) if isinstance(getattr(self, k), DataSeries)]

    def assess(self, k, kObs=None, faus=None,
               E=None, w=None, mu=None, Cov=None):
        """Common interface for both assess_ens and _ext.

        The _ens assessment function gets called if E is not None,
        and _ext if mu is not None.

        faus: One or more of ['f',' a', 'u'], indicating
              that the result should be stored in (respectively)
              the forecast/analysis/universal attribute.
              Default: 'u' if kObs is None else 'au' ('a' and 'u').
        """

        # Initial consistency checks.
        if k == 0:
            if kObs is not None:
                raise KeyError("DAPPER convention: no obs at t=0."
                               " Helps avoid bugs.")
            if faus is None:
                faus = 'u'
            if self._is_ens == True:
                if E is None:
                    raise TypeError(
                        "Expected ensemble input but E is None")
                if mu is not None:
                    raise TypeError(
                        "Expected ensemble input but mu/Cov is not None")
            else:
                if E is not None:
                    raise TypeError(
                        "Expected mu/Cov input but E is not None")
                if mu is None:
                    raise TypeError(
                        "Expected mu/Cov input but mu is None")

        # Default. Don't add more defaults. It just gets confusing.
        if faus is None:
            faus = 'u' if kObs is None else 'au'

        # Select assessment call and arguments
        if self._is_ens:
            _assess = self.assess_ens
            _prms   = {'E': E, 'w': w}
        else:
            _assess = self.assess_ext
            _prms   = {'mu': mu, 'P': Cov}

        for sub in faus:

            # Skip assessment if ('u' and stats not stored or plotted)
            if k != 0 and kObs == None:
                if not (self.store_u or self.LP_instance.any_figs):
                    continue

            # Silence repeat warnings caused by zero variance
            with np.errstate(divide='call', invalid='call'):
                np.seterrcall(warn_zero_variance)

                # Assess
                stats_now = Avrgs()
                _assess(stats_now, self.xx[k], **_prms)
                self.derivative_stats(stats_now)
                self.summarize_marginals(stats_now)

            # Write current stats to series
            for name, val in stats_now.items():
                stat = struct_tools.deep_getattr(self, name)
                isFaust = isinstance(stat, series.FAUSt)
                stat[(k, kObs, sub) if isFaust else kObs] = val

            # LivePlot -- Both init and update must come after the assessment.
            try:
                self.LP_instance.update((k, kObs, sub), E, Cov)
            except AttributeError:
                self.LP_instance = liveplotting.LivePlot(
                    self, self.liveplots, (k, kObs, sub), E, Cov)

    def summarize_marginals(self, now):
        "Compute Mean-field and RMS values"
        formulae = {**self.field_summaries, **self.sector_summaries}

        with np.errstate(divide='ignore', invalid='ignore'):
            for stat in list(now):
                field = now[stat]
                for suffix, formula in formulae.items():
                    statpath = stat+'.'+suffix
                    if struct_tools.deep_hasattr(self, statpath):
                        now[statpath] = formula(field)

    def derivative_stats(self, now):
        """Stats that derive from others (=> not specific for _ens or _ext)."""
        now.gscore = 2*np.log(now.std) + (now.err/now.std)**2

    def assess_ens(self, now, x, E, w):
        """Ensemble and Particle filter (weighted/importance) assessment."""
        N, Nx = E.shape

        if w is None:
            w = np.ones(N)/N  # All equal. Also, rm attr from stats:
            if hasattr(self, 'w'):
                delattr(self, 'w')
        else:
            now.w = w
            if abs(w.sum()-1) > 1e-5:
                raise RuntimeError("Weights did not sum to one.")
        if not np.all(np.isfinite(E)):
            raise RuntimeError("Ensemble not finite.")
        if not np.all(np.isreal(E)):
            raise RuntimeError("Ensemble not Real.")

        now.mu  = w @ E
        now.err = now.mu - x
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
        std = np.sqrt(var)  # NB: biased (even though var is unbiased)
        now.std = std

        # For simplicity, use naive (biased) formulae, derived
        # from "empirical measure". See doc/unbiased_skew_kurt.jpg.
        # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
        A_pow *= A
        now.skew = np.nanmean(w @ A_pow / (std*std*std))
        A_pow *= A
        now.kurt = np.nanmean(w @ A_pow / var**2 - 3)

        now.mad  = np.nanmean(w @ abs(A))

        if hasattr(self, 'svals'):
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

        now.mu  = mu
        now.err = now.mu - x

        var = P.diag if isinstance(P, CovMat) else np.diag(P)
        now.std = np.sqrt(var)

        # Here, sqrt(2/pi) is the ratio, of MAD/STD for Gaussians
        now.mad = np.nanmean(now.std) * np.sqrt(2/np.pi)

        if hasattr(self, 'svals'):
            P         = P.full if isinstance(P, CovMat) else P
            s2, U      = sla.eigh(P)
            now.svals = np.sqrt(np.maximum(s2, 0.0))[::-1]
            now.umisf = (U.T @ now.err)[::-1]

    def average_in_time(self, kk=None, kkObs=None, free=False):
        """Avarage all univariate (scalar) time series.

        - ``kk``    time inds for averaging
        - ``kkObs`` time inds for averaging obs
        """
        chrono = self.HMM.t
        if kk is None:
            kk     = chrono.mask_BI
        if kkObs is None:
            kkObs  = chrono.maskObs_BI

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
                    avrgs[sub] = series.mean_with_conf(tseries[kkObs, sub])
                if tseries.store_u:
                    avrgs['u'] = series.mean_with_conf(tseries[kk, 'u'])

            elif isinstance(tseries, DataSeries):
                if tseries.array.shape[1:] != ():
                    return average_multivariate()
                elif len(tseries.array) == self.KObs+1:
                    avrgs = series.mean_with_conf(tseries[kkObs])
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

        .. note:: ``store_u`` (whether to store non-obs-time stats) must
        have been ``True`` to have smooth graphs as in the actual LivePlot.

        .. note:: Ensembles are generally not stored in the stats
        and so cannot be replayed.
        """

        # Time settings
        chrono = self.HMM.t
        if t2 is None:
            t2 = t1 + chrono.Tplot

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
        for k, kObs, t, dt in progbar(chrono.ticker, desc):
            if t1 <= t <= t2:
                if kObs is not None:
                    LP.update((k, kObs, 'f'), None, None)
                    LP.update((k, kObs, 'a'), None, None)
                LP.update((k, kObs, 'u'), None, None)

        # Pause required when speed=inf.
        # On Mac, it was also necessary to do it for each fig.
        if LP.any_figs:
            for name, (num, updater) in LP.figures.items():
                if plt.fignum_exists(num) and getattr(updater, 'is_active', 1):
                    plt.figure(num)
                    plt.pause(0.01)


def register_stat(self, name, value):
    setattr(self, name, value)
    if not hasattr(self, "stat_register"):
        self.stat_register = []
    self.stat_register.append(name)


class Avrgs(StatPrint, struct_tools.DotDict):
    """A DotDict specialized for stat. averages.

    Embellishments:
    - StatPrint
    - tabulate
    - getattr that supports abbreviations.
    """

    def tabulate(self, statkeys=()):
        columns = tabulate_avrgs([self], statkeys, decimals=None)
        return tabulate(columns, headers="keys").replace('␣', ' ')

    abbrevs = {'rmse': 'err.rms', 'rmss': 'std.rms', 'rmv': 'std.rms'}

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
def align_col(col, header, pad='␣', missingval='', frmt=None):
    """Format a single column, return as list.

    - Use tabulate() to get decimal point alignment.
    - Inf and nan are handled individually so that they don't
      align left of the decimal point (makes too wide columns).
      Custom ``frmt`` also supported.
    - Pad (on the right) each row so that the widths are equal.
    """

    def preprocess(x):
        try:
            # Custom frmt supplied
            if frmt is not None:
                return frmt(x)

            # Standard formatting
            if x is None:
                return missingval
            elif np.isnan(x):
                return "NAX"
            elif x == -np.inf:
                return "-INX"
            elif x == np.inf:
                return "INX"
            return x  # leave formatting to tabulate()

        except TypeError:
            return missingval

    def postprocess(s):
        s = s.replace("NAX", "nan")
        s = s.replace("INX", "inf")
        return s

    # Make text column, aligned
    col = [[preprocess(x)] for x in col]
    col = tabulate(col, [header], 'plain')
    col = col.split("\n")  # NOTE: dont use splitlines (removes empty lines)

    # Undo nan/inf treatment
    col = [postprocess(s) for s in col]

    # Pad on the right, for equal widths
    mxW = max(len(s) for s in col)
    col = [s.ljust(mxW) for s in col]

    # Use pad char. on BOTH left/right, to prevent trunc. by later tabulate().
    col = [s.replace(" ", pad) for s in col]

    return col


def unpack_uqs(uq_list, decimals=None, cols=("val", "conf")):
    """Make array whose (named) cols are `[uq.col for uq in uq_list]`.

    Embellishments:
    - Insert None (in each col) if uq is None.
    - Apply uq.round() when extracting val & conf.
    """

    def unpack1(arr, i, uq):
        if uq is None:
            return
        # val/conf
        if decimals is None:
            v, c = uq.round()
        else:
            v, c = np.round([uq.val, uq.conf], decimals)
        arr["val"][i], arr["conf"][i] = v, c
        # Others
        for col in struct_tools.complement(cols, ["val", "conf"]):
            try:
                arr[col][i] = getattr(uq, col)
            except AttributeError:
                pass

    # np.array with named columns. "O" => allow storing None's.
    dtypes = np.dtype([(c, "O") for c in cols])
    arr = np.full_like(uq_list, dtype=dtypes, fill_value=None)
    for i, uq in enumerate(uq_list):
        unpack1(arr, i, uq)

    return arr


def tabulate_avrgs(avrgs_list, statkeys=(), decimals=None):
    """Tabulate avrgs (val±conf)."""

    if not statkeys:
        statkeys = ['rmse.a', 'rmv.a', 'rmse.f']

    columns = {}
    for stat in statkeys:
        column = unpack_uqs(
            [getattr(a, stat, None) for a in avrgs_list], decimals)
        vals   = align_col(column["val"], stat)
        confs  = align_col(column["conf"], '1σ')
        headr  = vals[0]+'  1σ'
        mattr  = [v + ' ±'+c for v, c in zip(vals, confs)][1:]
        columns[headr] = mattr

    return columns


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
