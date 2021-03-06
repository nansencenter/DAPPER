"""Tools for plotting."""

import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from mpl_tools.fig_layout import freshfig
from numpy import arange, array
from patlib.std import find_1st_ind
from scipy.interpolate import interp1d

import dapper.tools.series as series
from dapper.tools.rounding import round2sigfig


def setup_wrapping(M, periodicity=None):
    """Make state indices representative for periodic system.

    More accurately: Wrap the state indices and create a function that
    does the same for state vectors (or and ensemble thereof).

    - 'periodicity' defines the mode of the wrapping.
      Default: "+1", meaning that the first element is appended after the last.
    """
    if periodicity in (None, True):
        periodicity = "+1"

    if periodicity == "+1":
        ii = arange(M+1)

        def wrap(E):
            return E[..., list(range(M))+[0]]

    elif periodicity == "+/-05":
        ii = np.hstack([-0.5, arange(M), M-0.5])

        def wrap(E):
            midpoint = (E[..., [0]] + E[..., [-1]])/2
            return np.concatenate([midpoint, E, midpoint], axis=-1)

    else:
        ii = arange(M)
        def wrap(x): return x

    return ii, wrap


def amplitude_animation(EE, dt=None, interval=1,
                        periodicity=None, blit=True,
                        fignum=None, repeat=False):
    fig, ax = freshfig(fignum)
    ax.set_xlabel('State index')
    ax.set_ylabel('Amplitue')
    ax.set_ylim(*stretch(*xtrema(EE), 1.1))

    if EE.ndim == 2:
        EE = np.expand_dims(EE, 1)
    K, N, Nx = EE.shape

    ii, wrap = setup_wrapping(Nx, periodicity)

    lines = ax.plot(ii, wrap(EE[0]).T)
    ax.set_xlim(*xtrema(ii))

    if dt is not None:
        times = 'time = %.1f'
        lines += [ax.text(0.05, 0.9, '', transform=ax.transAxes)]

    def anim(k):
        Ek = wrap(EE[k])
        for n in range(N):
            lines[n].set_ydata(Ek[n])
        if len(lines) > N:
            lines[-1].set_text(times % (dt*k))
        return lines

    return FuncAnimation(fig, anim, range(K),
                         interval=interval, blit=blit,
                         repeat=repeat)


def adjust_position(ax, adjust_extent=False, **kwargs):
    """Adjust values (add) to get_position().

    `kwarg` must be one of `x0`, `y0`, `width`, `height`.
    """
    # Load get_position into d
    pos = ax.get_position()
    d = {}
    for key in ['x0', 'y0', 'width', 'height']:
        d[key] = getattr(pos, key)
    # Make adjustments
    for key, item in kwargs.items():
        d[key] += item
        if adjust_extent:
            if key == 'x0':
                d['width']  -= item
            if key == 'y0':
                d['height'] -= item
    # Set
    ax.set_position(d.values())


def xtrema(xx, axis=None):
    a = np.nanmin(xx, axis)
    b = np.nanmax(xx, axis)
    return a, b


def stretch(a, b, factor=1, int_=False):
    """Stretch distance `a-b` by factor.

    Return a, b.
    If int_: `floor(a)` and `ceil(b)`
    """
    c = (a+b)/2
    a = c + factor*(a-c)
    b = c + factor*(b-c)
    if int_:
        a = np.floor(a)
        b = np.ceil(b)
    return a, b


def set_ilim(ax, i, Min=None, Max=None):
    """Set bounds on axis i."""
    if i == 0:
        ax.set_xlim(Min, Max)
    if i == 1:
        ax.set_ylim(Min, Max)
    if i == 2:
        ax.set_zlim(Min, Max)


def estimate_good_plot_length(xx, chrono=None, mult=100):
    """Estimate good length for plotting stuff.

    Tries to estimate the time scale of the system.
    Provide sensible fall-backs (better if chrono is supplied).

    Example
    -------
    >>> K_lag = estimate_good_plot_length(stats.xx, chrono, mult=80) # doctest: +SKIP
    """
    if xx.ndim == 2:
        # If mult-dim, then average over dims (by ravel)....
        # But for inhomogeneous variables, it is important
        # to subtract the mean first!
        xx = xx - np.mean(xx, axis=0)
        xx = xx.ravel(order='F')

    try:
        K = mult * series.estimate_corr_length(xx)
    except ValueError:
        K = 0

    if chrono is not None:
        t = chrono
        K = int(min(max(K, t.dkObs), t.K))
        T = round2sigfig(t.tt[K], 2)  # Could return T; T>tt[-1]
        K = find_1st_ind(t.tt >= T)
        if K:
            return K
        else:
            return t.K
    else:
        K = int(min(max(K, 1), len(xx)))
        T = round2sigfig(K, 2)
        return K


def plot_pause(interval):
    """Similar to `plt.pause`, but doesn't focus window.

    NB: doesn't create windows either.
    For that, use `plt.pause` or `plt.show` instead.
    """
    # plt.pause(0) just seems to freeze execution.
    if interval == 0:
        return

    try:
        # Implement plt.pause() that doesn't focus window, c.f.
        # https://github.com/matplotlib/matplotlib/issues/11131
        # https://stackoverflow.com/q/45729092
        # Only necessary for some platforms (e.g. Windows) and mpl versions.
        # Even then, mere figure creation may steal the focus.
        # This was done deliberately:
        # https://github.com/matplotlib/matplotlib/pull/6384#issue-69259165
        # https://github.com/matplotlib/matplotlib/issues/8246#issuecomment-505460935
        # from matplotlib import _pylab_helpers

        def _plot_pause(interval, focus_figure=True):
            canvas = plt.gcf().canvas
            manager = canvas.manager
            if manager is not None:
                if canvas.figure.stale:
                    canvas.draw_idle()
                if focus_figure:
                    plt.show(block=False)
                # if not is_notebook: # also see below
                if True:
                    canvas.start_event_loop(interval)
            else:
                time.sleep(interval)
        _plot_pause(interval, focus_figure=False)

    except: # noqa
        # Jupyter notebook support: https://stackoverflow.com/q/34486642
        # Note: no longer needed with the above _plot_pause()?
        plt.gcf().canvas.draw()
        time.sleep(0.1)


def plot_hovmoller(xx, chrono=None, **kwargs):
    """Plot Hovmöller diagram."""
    fig, ax = freshfig(26, figsize=(4, 3.5))

    if chrono is not None:
        mask = chrono.tt <= chrono.Tplot*2
        kk   = chrono.kk[mask]
        tt   = chrono.tt[mask]
        ax.set_ylabel('Time (t)')
    else:
        K    = estimate_good_plot_length(xx, mult=20)
        kk   = arange(K)
        tt   = kk
        ax.set_ylabel('Time indices (k)')

    plt.contourf(arange(xx.shape[1]), tt, xx[kk], 25)
    plt.colorbar()
    ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Hovmoller diagram (of 'Truth')")
    ax.set_xlabel('Dimension index (i)')

    plot_pause(0.1)
    plt.tight_layout()


def integer_hist(E, N, centrd=False, weights=None, **kwargs):
    """Histogram for integers."""
    ax = plt.gca()
    rnge = (-0.5, N+0.5) if centrd else (0, N+1)
    ax.hist(E, bins=N+1, range=rnge, density=True, weights=weights, **kwargs)
    ax.set_xlim(rnge)


def not_available_text(ax, txt=None, fs=20):
    if txt is None:
        txt = '[Not available]'
    else:
        txt = '[' + txt + ']'
    ax.text(0.5, 0.5, txt,
            fontsize=fs,
            transform=ax.transAxes,
            va='center', ha='center',
            wrap=True)


def plot_err_components(stats):
    """Plot components of the error.

    .. note::
      it was chosen to plot(ii, mean_in_time(abs(err_i))),
      and thus the corresponding spread measure is MAD.
      If one chose instead: plot(ii, std_in_time(err_i)),
      then the corresponding measure of spread would have been std.
      This choice was made in part because (wrt. subplot 2)
      the singular values (svals) correspond to rotated MADs,
      and because rms(umisf) seems to convoluted for interpretation.
    """
    fig, (ax0, ax1, ax2) = freshfig(25, figsize=(6, 6), nrows=3)

    chrono = stats.HMM.t
    Nx     = stats.xx.shape[1]

    err   = np.mean(np.abs(stats.err  .a), 0)
    sprd  = np.mean(stats.mad  .a, 0)
    umsft = np.mean(np.abs(stats.umisf.a), 0)
    usprd = np.mean(stats.svals.a, 0)

    ax0.plot(arange(Nx), err, 'k', lw=2, label='Error')
    if Nx < 10**3:
        ax0.fill_between(arange(Nx), [0]*len(sprd), sprd, alpha=0.7, label='Spread')
    else:
        ax0.plot(arange(Nx), sprd, alpha=0.7, label='Spread')
    # ax0.set_yscale('log')
    ax0.set_title('Element-wise error comparison')
    ax0.set_xlabel('Dimension index (i)')
    ax0.set_ylabel('Time-average (_a) magnitude')
    ax0.set_xlim(0, Nx-1)
    ax0.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    ax0.legend(loc='upper right')

    ax1.set_xlim(0, Nx-1)
    ax1.set_xlabel('Principal component index')
    ax1.set_ylabel('Time-average (_a) magnitude')
    ax1.set_title('Spectral error comparison')
    has_been_computed = np.any(np.isfinite(umsft))
    if has_been_computed:
        L = len(umsft)
        ax1.plot(arange(L), umsft, 'k', lw=2, label='Error')
        ax1.fill_between(arange(L), [0]*L, usprd, alpha=0.7, label='Spread')
        ax1.set_yscale('log')
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    else:
        not_available_text(ax1)

    rmse = stats.err_rms.a[chrono.maskObs_BI]
    ax2.hist(rmse, bins=30, density=False)
    ax2.set_ylabel('Num. of occurence (_a)')
    ax2.set_xlabel('RMSE')
    ax2.set_title('Histogram of RMSE values')
    ax2.set_xlim(left=0)

    plot_pause(0.1)
    plt.tight_layout()


def plot_rank_histogram(stats):
    chrono = stats.HMM.t

    has_been_computed = \
        hasattr(stats, 'rh') and \
        not all(stats.rh.a[-1] == array(np.nan).astype(int))

    fig, ax = freshfig(24, (6, 3), loc="3313")
    ax.set_title('(Mean of marginal) rank histogram (_a)')
    ax.set_ylabel('Freq. of occurence\n (of truth in interval n)')
    ax.set_xlabel('ensemble member index (n)')

    if has_been_computed:
        ranks = stats.rh.a[chrono.maskObs_BI]
        Nx    = ranks.shape[1]
        N     = stats.xp.N
        if not hasattr(stats, 'w'):
            # Ensemble rank histogram
            integer_hist(ranks.ravel(), N)
        else:
            # Experimental: weighted rank histogram.
            # Weight ranks by inverse of particle weight. Why? Coz, with correct
            # importance weights, the "expected value" histogram is then flat.
            # Potential improvement: interpolate weights between particles.
            w  = stats.w.a[chrono.maskObs_BI]
            K  = len(w)
            w  = np.hstack([w, np.ones((K, 1))/N])  # define weights for rank N+1
            w  = array([w[arange(K), ranks[arange(K), i]] for i in range(Nx)])
            w  = w.T.ravel()
            # Artificial cap. Reduces variance, but introduces bias.
            w  = np.maximum(w, 1/N/100)
            w  = 1/w
            integer_hist(ranks.ravel(), N, weights=w)
    else:
        not_available_text(ax)

    plot_pause(0.1)
    plt.tight_layout()


# TODO: rm
def adjustable_box_or_forced():
    """For set_aspect(), adjustable='box-forced' replaced by 'box' since mpl 2.2.0."""
    from pkg_resources import parse_version as pv
    return 'box-forced' if pv(mpl.__version__) < pv("2.2.0") else 'box'


def axis_scale_by_array(ax, arr, axis='y', nbins=3):
    """Scale axis so that the arr entries appear equidistant.

    The full transformation is piecewise-linear.
    """
    yy = array([y for y in arr if y is not None], dtype=float)  # rm None

    # Make transformation
    xx = arange(len(yy))
    func = interp1d(xx, yy, fill_value="extrapolate")
    invf = interp1d(yy, xx, fill_value="extrapolate")

    # Set transformation
    set_scale = eval(f"ax.set_{axis}scale")
    set_scale('function', functions=(invf, func))

    # Adjust axis ticks
    _axis = getattr(ax, axis+"axis")
    _axis.set_major_locator(ticker.FixedLocator(yy, nbins=nbins))
    _axis.set_minor_locator(ticker.FixedLocator(yy))
    _axis.set_minor_formatter(ticker.NullFormatter())
