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

    Parameters
    ----------
    M: int
        Length of the periodic domain
    periodicity: bool, optional
        The mode of the wrapping.
        "+1": the first element is appended after the last.
        "+/-05": adding the midpoint of the first and last elements.
        Default: "+1"

    Returns
    -------
    ii: ndarray
        indices of periodic domain
    wrap: func
        transform non-periodic data into periodic data
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
    """Animation of line chart.

    Using an ensemble of
    the shape (time, ensemble size, state vector length).

    Parameters
    ----------
    EE: ndarray
        Ensemble arry of the shape (K, N, Nx).
        K is the length of time, N is the ensemble size, and
        Nx is the length of state vector.
    dt: float
        Time interval of each frame.
    interval: float, optional
        Delay between frames in milliseconds. Defaults to 200.
    periodicity: bool, optional
        The mode of the wrapping.
        "+1": the first element is appended after the last.
        "+/-05": adding the midpoint of the first and last elements.
        Default: "+1"
    blit: bool, optional
        Controls whether blitting is used to optimize drawing. Default: True
    fignum: int, optional
        Figure index. Default: None
    repeat: bool, optional
        If True, repeat the animation. Default: False
    """
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
    """Adjust (add) values of plot bounding box using get_position().

    Parameters
    ----------
    ax: matplotlib.axes
    adjust_extent: bool, optional
        If true, do not adjust the coordinate of the bounding box.
        Defaults: False
    kwargs: dict
        the keys must be `x0`, `y0`, `width`, `height`;
        the values are length changes.
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
    """Get minimum and maximum of a sequence.

    Parameters
    ----------
    xx: ndarray
    axis: int, optional
        Specific axis for min and max. Defaults: None

    Returns
    -------
    a: float
        min value
    b: float
        max value
    """
    a = np.nanmin(xx, axis)
    b = np.nanmax(xx, axis)
    return a, b


def stretch(a, b, factor=1, int_=False):
    """Stretch distance `a-b` by factor.

    Parameters
    ----------
    a: float
        Lower bound of domain.
    b: float
        Upper bound of domain.
    factor: float, optional
        Streching factor. Defaults: 1
    int_: bool, optional
        If True, the domain bounds are integer.
        Defaults: False

    Returns
    -------
    a: float
        Lower bound of domain.
    b: float
        Upper bound of domain.
    """
    c = (a+b)/2
    a = c + factor*(a-c)
    b = c + factor*(b-c)
    if int_:
        a = np.floor(a)
        b = np.ceil(b)
    return a, b


def set_ilim(ax, i, Min=None, Max=None):
    """Set bounds on axis i.

    Parameters
    ----------
    ax: matplotlib.axes
    i: int
        1: x-axis; 2: y-axis; 3: z-axis
    Min: float, optional
        Lower bound limit. Defaults: None
    Max: float, optional
        Upper bound limit. Defaults: None
    """
    if i == 0:
        ax.set_xlim(Min, Max)
    if i == 1:
        ax.set_ylim(Min, Max)
    if i == 2:
        ax.set_zlim(Min, Max)


def estimate_good_plot_length(xx, chrono=None, mult=100):
    """Estimate the range of the xx slices for plotting.

    The length is based on the estimated time scale (wavelength)
    of the system.
    Provide sensible fall-backs (better if chrono is supplied).

    Parameters
    ----------
    xx: ndarray
        Plotted array
    chrono: `dapper.tools.chronos.Chronology`, optional
        object with property dkObS. Defaults: None
    mult: int, optional
        Number of waves for plotting. Defaults: 100

    Returns
    -------
    K: int
        length for plotting

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
    """Like `plt.pause`, but better.

    Actually works in Jupyter notebooks, unlike `plt.pause`.

    In regular mpl (interactive) backends: doesn't focus window.
    NB: doesn't create windows either.
    For that, use `plt.pause` or `plt.show` instead.
    """
    # plt.pause(0) just seems to freeze execution.
    if interval == 0:
        return

    if mpl.get_backend() == "nbAgg":  # ie. %matplotlib notebook
        # https://stackoverflow.com/q/34486642
        plt.gcf().canvas.draw()
        time.sleep(interval)

        # About the "small" figures: https://stackoverflow.com/a/66399257
        # It seems to me that it's using the "inline" backend until
        # the liveplotting finishes. Unfortunately the "inline"
        # backend is incompatible with "stop/pause" buttons.

    elif "inline" in mpl.get_backend():  # ie. %matplotlib inline
        # https://stackoverflow.com/a/29675706/38281
        # NB: Not working, but could possibly be made to work,
        # except that it won't support a "pause/stop" button.
        from IPython import display
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(interval)

    else:  # for non-notebook interactive backends

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


def plot_hovmoller(xx, chrono=None, **kwargs):
    """Plot Hovmöller diagram.

    Parameters
    ----------
    xx: ndarray
        Plotted array
    chrono: `dapper.tools.chronos.Chronology`, optional
        object with property dkObS. Defaults: None
    """
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

    plt.pause(0.1)
    plt.tight_layout()


def integer_hist(E, N, centrd=False, weights=None, **kwargs):
    """Histogram for integers.

    Parameters
    ----------
    E: ndarray
        Ensemble array.
    N: int
        Number of histogram bins.
    centrd: bool, optional
        If True, each bin is centered in the midpoint. Default: False
    weights: float, optional
        Weights for histogram. Default: None
    kwargs: dict
        keyword arguments for matplotlib.hist
    """
    ax = plt.gca()
    rnge = (-0.5, N+0.5) if centrd else (0, N+1)
    ax.hist(E, bins=N+1, range=rnge, density=True, weights=weights, **kwargs)
    ax.set_xlim(rnge)


def not_available_text(ax, txt=None, fs=20):
    """Plot given text on the figure

    Parameters
    ----------
    ax: matplotlib.axes
    txt: str, optional
        Printed text. Defaults: '[Not available]'
    fs: float, optional
        Font size. Defaults: 20.
    """
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

    Parameters
    ----------
    stats: `dapper.stats.Stats`

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

    err   = np.mean(np.abs(stats.err.a), 0)
    sprd  = np.mean(stats.std.a, 0)
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

    rmse = stats.err.rms.a[chrono.maskObs_BI]
    ax2.hist(rmse, bins=30, density=False)
    ax2.set_ylabel('Num. of occurence (_a)')
    ax2.set_xlabel('RMSE')
    ax2.set_title('Histogram of RMSE values')
    ax2.set_xlim(left=0)

    plt.pause(0.1)
    plt.tight_layout()


def plot_rank_histogram(stats):
    """Plot rank histogram of ensemble.

    Parameters
    ----------
    stats: `dapper.stats.Stats`
    """
    chrono = stats.HMM.t

    has_been_computed = \
        hasattr(stats, 'rh') and \
        not all(stats.rh.a[-1] == array(np.nan).astype(int))

    fig, ax = freshfig(24, figsize=(6, 3))
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

    plt.pause(0.1)
    plt.tight_layout()


# TODO: rm
def adjustable_box_or_forced():
    """For set_aspect(), adjustable='box-forced' replaced by 'box' since mpl 2.2.0."""
    from pkg_resources import parse_version as pv
    return 'box-forced' if pv(mpl.__version__) < pv("2.2.0") else 'box'


def axis_scale_by_array(ax, arr, axis='y', nbins=3):
    """Scale axis so that the arr entries appear equidistant.

    The full transformation is piecewise-linear.

    Parameters
    ----------
    ax: matplotlib.axes
    arr: ndarray
        Array for plotting
    axis: str, optional
        Scaled axis, which can be 'x', 'y' or 'z'. Defaults: 'y'
    nbins: int, optional
        Number of major ticks. Defaults: 3
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
