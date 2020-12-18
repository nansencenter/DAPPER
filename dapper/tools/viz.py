"""Tools for plotting."""

import inspect
import itertools
import os
import textwrap
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import transforms as mtransforms
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import CheckButtons
from numpy import arange, array
from patlib.std import find_1st_ind
from scipy.interpolate import interp1d
from struct_tools import NicePrint

import dapper.tools.series as series
from dapper.dpr_config import rc
from dapper.tools.rounding import round2sigfig


def setup_wrapping(M, periodicity=None):
    """
    Wrap the state indices and
    create a function that does the same for state vectors (or and ensemble thereof).

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
    """
    Adjust values (add) to get_position().
    kwarg must be one of 'x0','y0','width','height'.
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


def stretch(a, b, factor=1, int=False):
    """
    Stretch distance a-b by factor.
    Return a,b.
    If int: floor(a) and ceil(b)
    """
    c = (a+b)/2
    a = c + factor*(a-c)
    b = c + factor*(b-c)
    if int:
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

# Examples:
# K_lag = estimate_good_plot_length(stats.xx,chrono,mult = 80)


def estimate_good_plot_length(xx, chrono=None, mult=100):
    """
    Estimate good length for plotting stuff
    from the time scale of the system.
    Provide sensible fall-backs (better if chrono is supplied).
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
    """Similar to plt.pause(), but doesn't focus window.

    NB: doesn't create windows either.
    For that, use plt.pause() or plt.show() instead."""

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

        def _plot_pause(interval,  focus_figure=True):
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
    """
    Plot Hovmöller diagram.
    """
    fig, ax = freshfig(26, figsize=(4, 3.5), loc='331-22')

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
    fig, (ax0, ax1, ax2) = freshfig(25, figsize=(6, 6), loc='1313', nrows=3)

    chrono = stats.HMM.t
    Nx     = stats.xx.shape[1]

    err   = np.mean(np.abs(stats.err  .a), 0)
    sprd  = np.mean(stats.mad  .a, 0)
    umsft = np.mean(np.abs(stats.umisf.a), 0)
    usprd = np.mean(stats.svals.a, 0)

    ax0.plot(arange(Nx),               err, 'k', lw=2, label='Error')
    if Nx < 10**3:
        ax0.fill_between(arange(Nx), [0]*len(sprd), sprd, alpha=0.7, label='Spread')
    else:
        ax0.plot(arange(Nx),              sprd, alpha=0.7, label='Spread')
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
        ax1.plot(arange(L),      umsft, 'k', lw=2, label='Error')
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


def adjustable_box_or_forced():
    "For set_aspect(), adjustable='box-forced' replaced by 'box' since mpl 2.2.0."
    from pkg_resources import parse_version as pv
    return 'box-forced' if pv(mpl.__version__) < pv("2.2.0") else 'box'


def freshfig(num=None, figsize=None, *args, **kwargs):
    """Create/clear figure.

    Similar to::

      fig, ax = suplots(*args,**kwargs)

    With the modification that:

    - If the figure does not exist: create it.
      This allows for figure sizing -- even with mpl backend MacOS.
      Also place figure.
    - Otherwise: clear figure.
      Avoids closing/opening so as to keep pos and size.
    """
    exists = plt.fignum_exists(num)

    fig = plt.figure(num=num, figsize=figsize)

    # Deal with warning bug
    # https://github.com/matplotlib/matplotlib/issues/9970
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig.clf()

    loc = kwargs.pop('loc', None)
    if not exists and loc:
        fig_place(loc, fig)

    _, ax = plt.subplots(num=fig.number, *args, **kwargs)
    return fig, ax


def show_figs(fignums=None):
    """Move all fig windows to top"""
    if fignums is None:
        fignums = plt.get_fignums()
    try:
        fignums = list(fignums)
    except: # noqa
        fignums = [fignums]
    for f in fignums:
        plt.figure(f)
        fmw = plt.get_current_fig_manager().window
        fmw.attributes('-topmost', 1)  # Bring to front, but
        fmw.attributes('-topmost', 0)  # don't keep in front


def get_fig(fignum=None):
    "Get/validate fig handle"
    if fignum is None:
        return plt.gcf()
    elif isinstance(fignum, mpl.figure.Figure):
        return fignum
    else:
        return plt.figure(fignum)


def get_fmw(fignum=None):
    fig = get_fig(fignum)
    fmw = fig.canvas.manager.window
    # fmw = plt.get_current_fig_manager().window
    return fmw


def get_screen_size():
    """Get **available** screen size/resolution."""
    if mpl.get_backend().startswith('Qt'):
        # Inspired by spyder/widgets/shortcutssummary.py
        from qtpy.QtWidgets import QDesktopWidget  # noqa
        widget = QDesktopWidget()
        sg = widget.availableGeometry(widget.primaryScreen())
        x0 = sg.x()
        y0 = sg.y()
        w0 = sg.width()
        h0 = sg.height()
    elif mpl.get_backend() == "TkAgg":
        # https://stackoverflow.com/a/42951711/38281
        window = plt.get_current_fig_manager().window
        x0, y0 = 0, 0
        w0, h0 = window.wm_maxsize()
        # h = window.winfo_screenheight()
        # w = window.winfo_screenwidth()
    else:
        # Mac Retina Early 2013
        x0 = 0
        y0 = 23
        w0 = 1280
        h0 = 773
    return x0, y0, w0, h0


def fig_rel_geometry(fignum=None, x=None, y=None, w=None, h=None):
    """Place figure on screen, in coordinates relative (between 0 and 1)."""
    try:
        fmw = get_fmw(fignum)
    except AttributeError:
        return  # do nothing

    x0, y0, w0, h0 = get_screen_size()

    # It seems the window footers are not taken into account
    # by the geometry settings. Correct for this:
    footer = 0.028*(h0+y0)

    # Current values (Qt4Agg only!):
    w = w if w is not None else fmw.width() / w0
    h = h if h is not None else fmw.height()/h0
    x = x if x is not None else fmw.x()     / w0
    y = y if y is not None else fmw.y()     / h0

    x = x0 + x*w0
    y = y0 + y*h0 + footer
    w = w*w0
    h = h*h0 - footer

    try:  # For Qt4Agg/Qt5Agg
        fmw.setGeometry(x, y, w, h)
    except: # noqa # For TkAgg
        geo = f"{int(w)}x{int(h)}+{int(x)}+{int(y)}"
        fmw.geometry(newGeometry=geo)


def fig_place(loc, fignum=None):
    """Place figure on screen.

    - loc: string that defines the figures new geometry, given either as
       * NW, E, ...
       * 4 digits (as str or int) to define grid M,N,i,j.

    Example:
    >>> N = 3
    >>> for i in 1+arange(N):
    >>>   loc = str(N)*2 + str(i)*2
    >>>   fig_place(loc, i)
    """

    # NB: Experimental. Fails on some systems/backends.
    if not rc.place_figs:
        return

    loc = str(loc)
    loc = loc.replace(",", "")
    if not loc[:4].isnumeric():
        if loc.startswith('NW'):
            loc = '2211'
        elif loc.startswith('SW'):
            loc = '2221'
        elif loc.startswith('NE'):
            loc = '2212'
        elif loc.startswith('SE'):
            loc = '2222'
        elif loc.startswith('W'):
            loc = '1211'
        elif loc.startswith('E'):
            loc = '1212'
        elif loc.startswith('S'):
            loc = '2121'
        elif loc.startswith('N'):
            loc = '2111'

    # Split digits
    M, N = int(loc[0]), int(loc[1])
    if loc[3] == '-':
        i1, i2 = int(loc[2]), int(loc[4])
    else:
        i1, i2 = int(loc[2]), int(loc[2])
    if loc[-2] == '-':
        j1, j2 = int(loc[-3]), int(loc[-1])
    else:
        j1, j2 = int(loc[-1]), int(loc[-1])
    # Validate
    assert M >= i2 >= i1 > 0, "The specified col index is invalid."
    assert N >= j2 >= j1 > 0, "The specified row index is invalid."

    # Place
    di = i2-i1+1
    dj = j2-j1+1
    fig_rel_geometry(fignum,  (j1-1)/N,   (i1-1)/M,   dj/N,   di/M)


# https://stackoverflow.com/a/7396313


def autoscale_based_on(ax, line_handles):
    "Autoscale axis based (only) on line_handles."
    ax.dataLim = mtransforms.Bbox.unit()
    for iL, lh in enumerate(line_handles):
        xy = np.vstack(lh.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=(iL == 0))
    ax.autoscale_view()


def toggle_lines(ax=None, autoscl=True, numbering=False,
                 txtwidth=15, txtsize=None, state=None):
    """
    Make checkbuttons to toggle visibility of each line in current plot.
    autoscl  : Rescale axis limits as required by currently visible lines.
    numbering: Add numbering to labels.
    txtwidth : Wrap labels to this length.

    State of checkboxes can be inquired by
    OnOff = [lh.get_visible() for lh in
    ax.findobj(lambda x: isinstance(x,mpl.lines.Line2D))[::2]]
    """

    if ax is None:
        ax = plt.gca()
    if txtsize is None:
        txtsize = mpl.rcParams['font.size']

    # Get lines and their properties
    lines = {'handle': list(ax.get_lines())}
    for prop in ['label', 'color', 'visible']:
        lines[prop] = [plt.getp(x, prop) for x in lines['handle']]

    # Rm those that start with _
    not_ = [not x.startswith('_') for x in lines['label']]
    for prop in lines:
        lines[prop] = list(itertools.compress(lines[prop], not_))
    N = len(lines['handle'])

    # Adjust labels
    if numbering:
        lines['label'] = [str(i)+': '+x for i, x in enumerate(lines['label'])]
    if txtwidth:
        lines['label'] = [textwrap.fill(x, width=txtwidth) for x in lines['label']]

    # Set state. BUGGY? sometimes causes MPL complaints after clicking boxes
    if state is not None:
        state = array(state).astype(bool)
        lines['visible'] = state
        for i, x in enumerate(state):
            lines['handle'][i].set_visible(x)

    # Setup buttons
    # When there's many, the box-sizing is awful, but difficult to fix.
    W       = 0.23 * txtwidth/15 * txtsize/10
    nBreaks = sum(x.count('\n') for x in lines['label'])  # count linebreaks
    H       = min(1, 0.05*(N+nBreaks))
    plt.subplots_adjust(left=W+0.12, right=0.97)
    rax = plt.axes([0.05, 0.5-H/2, W, H])
    check = CheckButtons(rax, lines['label'], lines['visible'])

    # Adjust button style
    for i in range(N):
        check.rectangles[i].set(lw=0, facecolor=lines['color'][i])
        check.labels[i].set(color=lines['color'][i])
        if txtsize:
            check.labels[i].set(size=txtsize)

    # Callback
    def toggle_visible(label):
        ind    = lines['label'].index(label)
        handle = lines['handle'][ind]
        vs     = not lines['visible'][ind]
        handle.set_visible(vs)
        lines['visible'][ind] = vs
        if autoscl:
            autoscale_based_on(ax, list(itertools.compress(
                lines['handle'], lines['visible'])))
        plt.draw()
    check.on_clicked(toggle_visible)

    # Return focus
    plt.sca(ax)

    # Must return (and be received) so as not to expire.
    return check


def toggle_viz(*handles, prompt=False, legend=False, pause=True):
    """Toggle visibility of the graphics with handle handles."""

    are_viz = []
    for h in handles:

        # Core functionality: turn on/off
        is_viz = not h.get_visible()
        h.set_visible(is_viz)
        are_viz += [is_viz]

        # Legend updating. Basic version: works by
        #  - setting line's label (to /'_nolegend_' if off)
        #  - re-calling legend()
        if legend:
            if is_viz:
                try:
                    h.set_label(h.actual_label)
                except AttributeError:
                    pass
            else:
                h.actual_label = h.get_label()
                h.set_label('_nolegend_')
            # Legend refresh
            ax = h.axes
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=UserWarning)
                try:
                    ax.legend()
                except UserWarning:
                    # If all labels are '_nolabel_' then ax.legend() throws warning,
                    # and quits before refreshing.
                    # => Refresh by creating/rm another legend.
                    ax.legend('TMP').remove()

    # Pause at where used (typically sequentially in script)
    if prompt:
        input("Press <Enter> to continue...")
    if pause:
        plt.pause(0.02)

    return are_viz


class FigSaver(NicePrint):
    """
    Simplify exporting a figure, especially when it's part of a series.
    """

    def __init__(self, script=None, basename=None, n=-1, ext='.pdf'):

        # Defaults
        if script is None:  # Get __file__ of caller
            script = inspect.getfile(inspect.stack()[1][0])
        if basename is None:
            basename = 'figure'
        # Prep save dir
        sdir = rc.dirs.data / Path(script).stem
        os.makedirs(sdir, exist_ok=True)
        # Set state
        self.fname = sdir + basename
        self.n     = n
        self.ext   = ext

    @property
    def fullname(self):
        f = self.fname            # Abbrev
        if self.n >= 0:           # If indexing:
            f += '_n%d' % self.n  # Add index
        f += self.ext             # Add extension
        return f

    def save(self):
        f = self.fullname           # Abbrev
        print("Saving fig to:", f)  # Print
        plt.savefig(f)              # Save
        if self.n >= 0:             # If indexing:
            self.n += 1                 # Increment
            plt.pause(0.1)              # For safety


def nrowcol(nTotal, AR=1):
    "Return integer nrows and ncols such that nTotal ≈ nrows*ncols."
    nrows = int(np.floor(np.sqrt(nTotal)/AR))
    ncols = int(np.ceil(nTotal/nrows))
    return nrows, ncols


def axes_with_marginals(n_joint, n_marg, **kwargs):
    """
    Create a joint axis along with two marginal axes.

    Example:
    >>> ax_s, ax_x, ax_y = axes_with_marginals(4, 1)
    >>> x, y = np.random.randn(2,500)
    >>> ax_s.scatter(x,y)
    >>> ax_x.hist(x)
    >>> ax_y.hist(y,orientation="horizontal")
    """

    N = n_joint + n_marg

    # Method 1
    # fig, ((ax_s, ax_y), (ax_x, _)) = plt.subplots(2,2,num=plt.gcf().number,
    # sharex='col',sharey='row',gridspec_kw={
    #     'height_ratios':[n_joint,n_marg],
    #     'width_ratios' :[n_joint,n_marg]})
    # _.set_visible(False) # Actually removing would bug the axis ticks etc.

    # Method 2
    gs   = GridSpec(N, N, **kwargs)
    fig  = plt.gcf()
    ax_s = fig.add_subplot(gs[n_marg:N, 0:n_joint])
    ax_x = fig.add_subplot(gs[0:n_marg, 0:n_joint], sharex=ax_s)
    ax_y = fig.add_subplot(gs[n_marg:N, n_joint:N], sharey=ax_s)
    # Cannot delete ticks coz axis are shared
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_yticklabels(), visible=False)

    return ax_s, ax_x, ax_y


def cov_ellipse(ax, mu, sigma, **kwargs):
    """
    Draw ellipse corresponding to (Gaussian) 1-sigma countour of cov matrix.

    Inspired by https://stackoverflow.com/q/17952171

    Example:
    >>> ellipse = cov_ellipse(ax, y, R,
    >>>           facecolor='none', edgecolor='y',lw=4,label='$1\\sigma$')
    """

    # Cov --> Width, Height, Theta
    vals, vecs = sla.eigh(sigma)
    x, y       = vecs[:, -1]  # x-y components of largest (last) eigenvector
    theta      = np.degrees(np.arctan2(y, x))
    theta      = theta % 180

    h, w       = 2 * np.sqrt(vals.clip(0))

    # Get artist
    e = Ellipse(mu, w, h, theta, **kwargs)

    ax.add_patch(e)
    e.set_clip_box(ax.bbox)  # why is this necessary?

    # Return artist
    return e


def axis_scale_by_array(ax, arr, axis='y', nbins=3):
    """Scale axis so that the arr entries appear equidistant.

    The full transformation is piecewise-linear."""
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
