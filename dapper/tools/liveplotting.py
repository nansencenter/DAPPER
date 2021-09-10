"""On-line (live) plots of the DA process for various models and methods.

Liveplotters are given by a list of tuples as property or arguments in
`dapper.mods.HiddenMarkovModel`.

- The first element of the tuple determines whether the liveplotter is shown if
the names of liveplotters are not given by `liveplots` argument in
`assimilate`.

- The second element in the tuple gives the corresponding liveplotter
function/class. See example of function `LPs` in `dapper.mods.Lorenz63`.

The liveplotters can be fine-tuned by each DA experiments via argument of
`liveplots` when calling `assimilate`.

- `liveplots = True` turns on liveplotters set to default in the first
argument of the `HMM.liveplotter` and default liveplotters defined in this module
(`sliding_diagnostics` and `weight_histogram`).

- `liveplots` can also be a list of specified names of liveplotter, which
is the name of the corresponding liveplotting classes/functions.
"""

import matplotlib as mpl
import numpy as np
import scipy.linalg as sla
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import juggle_axes
from mpl_tools import is_notebook_or_qt, place, place_ax
from numpy import arange, nan, ones
from struct_tools import DotDict, deep_getattr

import dapper.tools.progressbar as pb
import dapper.tools.viz as viz
from dapper.dpr_config import rc
from dapper.mods.utils import linspace_int
from dapper.tools.chronos import format_time
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import read1
from dapper.tools.series import FAUSt, RollingArray
from dapper.tools.viz import not_available_text, plot_pause


class LivePlot:
    """Live plotting manager.

    Deals with

    - Pause, skip.
    - Which liveploters to call.
    - `plot_u`
    - Figure window (title and number).
    """

    def __init__(self,
                 stats,
                 liveplots,
                 key0=(0, None, 'u'),
                 E=None,
                 P=None,
                 speed=1.0,
                 replay=False,
                 **kwargs):
        """
        Initialize plots.

        - liveplots: figures to plot; alternatives:
            - `"default"/[]/True`: All default figures for this HMM.
            - `"all"`            : Even more.
            - non-empty `list`   : Only the figures with these numbers
                                 (int) or names (str).
            - `False`            : None.
        - speed: speed of animation.
            - `>100`: instantaneous
            - `1`   : (default) as quick as possible allowing for
                      plt.draw() to work on a moderately fast computer.
            - `<1`  : slower.
        """
        # Disable if not rc.liveplotting
        self.any_figs = False
        if not rc.liveplotting:
            return

        # Determine whether all/universal/intermediate stats are plotted
        self.plot_u = not replay or stats.store_u

        # Set speed/pause params
        self.params = {
            'pause_f': 0.05,
            'pause_a': 0.05,
            'pause_s': 0.05,
            'pause_u': 0.001,
        }
        # If speed>100: set to inf. Coz pause=1e-99 causes hangup.
        for pause in ["pause_"+x for x in "faus"]:
            speed = speed if speed < 100 else np.inf
            self.params[pause] /= speed

        # Write params
        self.params.update(getattr(stats.xp, "LP_kwargs", {}))
        self.params.update(kwargs)

        def get_name(init):
            """Get name of liveplotter function/class."""
            try:
                return init.__qualname__.split(".")[0]
            except AttributeError:
                return init.__class__.__name__

        # Set up dict of liveplotters
        potential_LPs = {}
        for show, init in default_liveplotters:
            potential_LPs[get_name(init)] = show, init
        # Add HMM-specific liveplotters
        for show, init in getattr(stats.HMM, 'liveplotters', {}):
            potential_LPs[get_name(init)] = show, init

        def parse_figlist(lst):
            """Figures requested for this xp. Convert to list."""
            if isinstance(lst, str):
                fn = lst.lower()
                if "all" == fn:
                    lst = ["all"]  # All potential_LPs
                elif "default" in fn:
                    lst = ["default"]         # All show_by_default
            elif hasattr(lst, '__len__'):
                lst = lst            # This list (only)
            elif lst:
                lst = ["default"]             # All show_by_default
            else:
                lst = [None]         # None
            return lst
        figlist = parse_figlist(liveplots)

        # Loop over requeted figures
        self.figures = {}
        for name, (show_by_default, init) in potential_LPs.items():
            if (figlist == ["all"]) or \
                    (name in figlist) or \
                    (figlist == ["default"] and show_by_default):

                # Startup message
                if not self.any_figs:
                    print('Initializing liveplots...')
                    if is_notebook_or_qt:
                        pauses = [self.params["pause_" + x] for x in "faus"]
                        if any((p > 0) for p in pauses):
                            print("Note: liveplotting does not work very well"
                                  " inside Jupyter notebooks. In particular,"
                                  " there is no way to stop/skip them except"
                                  " to interrupt the kernel (the stop button"
                                  " in the toolbar). Consider using instead"
                                  " only the replay functionality (with infinite"
                                  " playback speed).")
                    elif not pb.disable_user_interaction:
                        print('Hit <Space> to pause/step.')
                        print('Hit <Enter> to resume/skip.')
                        print('Hit <i> to enter debug mode.')
                    self.paused = False
                    self.run_ipdb = False
                    self.skipping = False
                    self.any_figs = True

                # Init figure
                post_title = "" if self.plot_u else "\n(obs times only)"
                updater = init(name, stats, key0, self.plot_u, E, P, **kwargs)
                if plt.fignum_exists(name) and getattr(updater, 'is_active', 1):
                    self.figures[name] = updater
                    fig = plt.figure(name)
                    win = fig.canvas
                    ax0 = fig.axes[0]
                    win.manager.set_window_title("%s" % name)
                    ax0.set_title(ax0.get_title() + post_title)
                    self.update(key0, E, P)  # Call initial update
                    plt.pause(0.01)          # Draw

    def update(self, key, E, P):
        """Update liveplots"""
        # Check if there are still open figures
        if self.any_figs:
            open_figns = plt.get_figlabels()
            live_figns = set(self.figures.keys())
            self.any_figs = bool(live_figns.intersection(open_figns))
        else:
            return

        # Playback control
        SPACE  = b' '
        CHAR_I = b'i'
        ENTERs = [b'\n', b'\r']  # Linux + Windows

        def pause():
            """Loop until user decision is made."""
            ch = read1()
            while True:
                # Set state (pause, skipping, ipdb)
                if ch in ENTERs:
                    self.paused = False
                elif ch == CHAR_I:
                    self.run_ipdb = True
                # If keypress valid, resume execution
                if ch in ENTERs + [SPACE, CHAR_I]:
                    break
                ch = read1()
                # Pause to enable zoom, pan, etc. of mpl GUI
                plot_pause(0.01)  # Don't use time.sleep()!

        # Enter pause loop
        if self.paused:
            pause()

        else:
            if key == (0, None, 'u'):
                # Skip read1 for key0 (coz it blocks)
                pass
            else:
                ch = read1()
                if ch == SPACE:
                    # Pause
                    self.paused = True
                    self.skipping = False
                    pause()
                elif ch in ENTERs:
                    # Toggle skipping
                    self.skipping = not self.skipping
                elif ch == CHAR_I:
                    # Schedule debug
                    # Note: The reason we dont set_trace(frame) right here is:
                    # - I could not find the right frame, even doing
                    #   >   frame = inspect.stack()[0]
                    #   >   while frame.f_code.co_name != "assimilate":
                    #   >       frame = frame.f_back
                    # - It just restarts the plot.
                    self.run_ipdb = True

        # Update figures
        if not self.skipping:
            faus = key[-1]
            if faus != 'u' or self.plot_u:
                for name, (updater) in self.figures.items():
                    if plt.fignum_exists(name) and \
                            getattr(updater, 'is_active', 1):
                        _ = plt.figure(name)
                        updater(key, E, P)
                        plot_pause(self.params['pause_'+faus])

        if self.run_ipdb:
            self.run_ipdb = False
            import inspect

            import ipdb
            print("Entering debug mode (ipdb).")
            print("Type '?' (and Enter) for usage help.")
            print("Type 'c' to continue the assimilation.")
            ipdb.set_trace(inspect.stack()[2].frame)


# TODO 6:
# - iEnKS diagnostics don't work at all when store_u=False
star = "${}^*$"


class sliding_diagnostics:
    """Plots a sliding window (like a heart rate monitor) of certain diagnostics."""

    def __init__(self, fignum, stats, key0, plot_u,
                 E, P, Tplot=None, **kwargs):

        # STYLE TABLES - Defines which/how diagnostics get plotted
        styles = {}
        def lin(a, b): return (lambda x: a + b*x)
        divN = 1/getattr(stats.xp, 'N', 99)
        # Columns: transf, shape, plt kwargs
        styles['RMS'] = {
            'err.rms'   : [None, None, dict(c='k', label='Error')],
            'spread.rms': [None, None, dict(c='b', label='Spread', alpha=0.6)],
        }
        styles['Values'] = {
            'skew': [None, None, dict(c='g', label=star+r'Skew/$\sigma^3$')],
            'kurt': [None, None, dict(c='r', label=star+r'Kurt$/\sigma^4{-}3$')],
            'trHK': [None, None, dict(c='k', label=star+'HK')],
            'infl': [lin(-10, 10), 'step', dict(c='c', label='10(infl-1)')],
            'N_eff': [lin(0, divN), 'dirac', dict(c='y', label='N_eff/N', lw=3)],
            'iters': [lin(0, .1), 'dirac', dict(c='m', label='iters/10')],
            'resmpl': [None, 'dirac', dict(c='k', label='resampled?')],
        }

        nAx = len(styles)
        GS = {'left': 0.125, 'right': 0.76}
        fig, axs = place.freshfig(fignum, figsize=(5, 1+nAx),
                                  nrows=nAx, sharex=True, gridspec_kw=GS)

        axs[0].set_title("Diagnostics")
        for style, ax in zip(styles, axs):
            ax.set_ylabel(style)
        ax.set_xlabel('Time (t)')
        place_ax.adjust_position(ax, y0=0.03)

        self.T_lag, K_lag, a_lag = validate_lag(Tplot, stats.HMM.tseq)

        def init_ax(ax, style_table):
            lines = {}
            for name in style_table:

                # SKIP -- if stats[name] is not in existence
                # Note: The nan check/deletion comes after the first ko.
                try:
                    stat = deep_getattr(stats, name)
                except AttributeError:
                    continue
                # try: val0 = stat[key0[0]]
                # except KeyError: continue
                # PS: recall (from series.py) that even if store_u is false, stat[k] is
                # still present if liveplots=True via the k_tmp functionality.

                # Unpack style
                ln = {}
                ln['transf'] = style_table[name][0] or (lambda x: x)
                ln['shape']  = style_table[name][1]
                ln['plt']    = style_table[name][2]

                # Create series
                if isinstance(stat, FAUSt):
                    ln['plot_u'] = plot_u
                    K_plot       = comp_K_plot(K_lag, a_lag, ln['plot_u'])
                else:
                    ln['plot_u'] = False
                    K_plot       = a_lag
                ln['data']   = RollingArray(K_plot)
                ln['tt']     = RollingArray(K_plot)

                # Plot (init)
                ln['handle'], = ax.plot(ln['tt'], ln['data'], **ln['plt'])

                # Plotting only nans yield ugly limits. Revert to defaults.
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                lines[name] = ln
            return lines

        # Plot
        self.d = [init_ax(ax, styles[style]) for style, ax in zip(styles, axs)]

        # Horizontal line at y=0
        self.baseline0, = ax.plot(
            ax.get_xlim(), [0, 0], c=0.5*ones(3), lw=0.7, label='_nolegend_')

        # Store
        self.axs   = axs
        self.stats = stats
        self.init_incomplete = True

    # Update plot
    def __call__(self, key, E, P):
        k, ko, faus = key

        stats  = self.stats
        tseq = stats.HMM.tseq
        ax0, ax1 = self.axs

        def update_arrays(lines):
            for name, ln in lines.items():
                stat = deep_getattr(stats, name)
                t    = tseq.tt[k]  # == tseq.tto[ko]
                if isinstance(stat, FAUSt):
                    # ln['data'] will contain duplicates for f/a times.
                    if ln['plot_u']:
                        val = stat[key]
                        ln['tt']  .insert(k, t)
                        ln['data'].insert(k, ln['transf'](val))
                    elif 'u' not in faus:
                        val = stat[key]
                        ln['tt']  .insert(ko, t)
                        ln['data'].insert(ko, ln['transf'](val))
                else:
                    # ln['data'] will not contain duplicates, coz only 'a' is input.
                    if 'a' in faus:
                        val = stat[ko]
                        ln['tt']  .insert(ko, t)
                        ln['data'].insert(ko, ln['transf'](val))
                    elif 'f' in faus:
                        pass

        def update_plot_data(ax, lines):

            def bend_into(shape, xx, yy):
                # Get arrays. Repeat (to use for intermediate nodes).
                yy = yy.array.repeat(3)
                xx = xx.array.repeat(3)
                if len(xx) == 0:
                    pass  # shortcircuit any modifications
                elif shape == 'step':
                    yy = np.hstack([yy[1:], nan])  # roll leftward
                elif shape == 'dirac':
                    nonlocal nDirac
                    axW      = np.diff(ax.get_xlim())
                    yy[0::3] = False           # set datapoin to 0
                    xx[2::3] = nan             # make datapoint disappear
                    xx      += nDirac*axW/100  # offset datapoint horizontally
                    nDirac  += 1
                return xx, yy

            nDirac = 1
            for _name, ln in lines.items():
                ln['handle'].set_data(*bend_into(ln['shape'], ln['tt'], ln['data']))

        def finalize_init(ax, lines, mm):
            # Rm lines that only contain NaNs
            for name in list(lines):
                ln   = lines[name]
                stat = deep_getattr(stats, name)
                if not stat.were_changed:
                    ln['handle'].remove()  # rm from axes
                    del lines[name]        # rm from dict
            # Add legends
            if lines:
                ax.legend(loc='upper left',
                          bbox_to_anchor=(1.01, 1), borderaxespad=0)
                if mm:
                    ax.annotate(star+": mean of\nmarginals",
                                xy=(0, -1.5/len(lines)),
                                xycoords=ax.get_legend().get_frame(),
                                bbox=dict(alpha=0.0), fontsize='small')
            # coz placement of annotate needs flush sometimes:
            plot_pause(0.01)

        # Insert current stats
        for lines, ax in zip(self.d, self.axs):
            update_arrays(lines)
            update_plot_data(ax, lines)

        # Set x-limits (time)
        sliding_xlim(ax0, self.d[0]['err.rms']['tt'], self.T_lag, margin=True)
        self.baseline0.set_xdata(ax0.get_xlim())

        # Set y-limits
        data0 = [ln['data'].array for ln in self.d[0].values()]
        data1 = [ln['data'].array for ln in self.d[1].values()]
        ax0.set_ylim(0, d_ylim(data0, ax0             , cC=0.2, cE=0.9)[1])
        ax1.set_ylim(*d_ylim(data1, ax1, Max=4, Min=-4, cC=0.3, cE=0.9))

        # Init legend. Rm nan lines.
        if self.init_incomplete and 'a' == faus:
            self.init_incomplete = False
            finalize_init(ax0, self.d[0], False)
            finalize_init(ax1, self.d[1], True)


def sliding_xlim(ax, tt, lag, margin=False):
    dt = lag/20 if margin else 0
    if tt.nFilled == 0:
        return  # Quit
    t1, t2 = tt.span()      # Get suggested span.
    s1, s2 = ax.get_xlim()  # Get previous lims.
    # If zero span (eg tt holds single 'f' and 'a'):
    if t1 == t2:
        t1 -= 1  # add width
        t2 += 1  # add width
    # If user has skipped (too much):
    elif np.isnan(t1):
        s2    -= dt     # Correct for dt.
        span   = s2-s1  # Compute previous span
        # If span<lag:
        if span < lag:
            span  += (t2-s2)  # Grow by "dt".
        span   = min(lag, span)  # Bound
        t1     = t2 - span       # Set span.
    ax.set_xlim(t1, t2 + dt)  # Set xlim to span


class weight_histogram:
    """Plots histogram of weights. Refreshed each analysis."""

    def __init__(self, fignum, stats, key0, plot_u, E, P, **kwargs):
        if not hasattr(stats, 'w'):
            self.is_active = False
            return
        fig, ax = place.freshfig(fignum, figsize=(7, 3), gridspec_kw={'bottom': .15})

        ax.set_xscale('log')
        ax.set_xlabel('Weigth')
        ax.set_ylabel('Count')
        self.stats = stats
        self.ax    = ax
        self.hist  = []
        self.bins  = np.exp(np.linspace(np.log(1e-10), np.log(1), 31))

    def __call__(self, key, E, P):
        k, ko, faus = key
        if 'a' == faus:
            w  = self.stats.w[key]
            N  = len(w)
            ax = self.ax

            self.is_active = N < 10001
            if not self.is_active:
                not_available_text(ax, 'Not computed (N > threshold)')
                return

            counted = w > self.bins[0]
            _ = [b.remove() for b in self.hist]
            nn, _, self.hist = ax.hist(
                w[counted], bins=self.bins, color='b')
            ax.set_ylim(top=max(nn))

            ax.set_title('N: {:d}.   N_eff: {:.4g}.   Not shown: {:d}. '.
                         format(N, 1/(w@w), N-np.sum(counted)))


class spectral_errors:
    """Plots the (spatial-RMS) error as a functional of the SVD index."""

    def __init__(self, fignum, stats, key0, plot_u, E, P, **kwargs):
        fig, ax = place.freshfig(fignum, figsize=(6, 3))
        ax.set_xlabel('Sing. value index')
        ax.set_yscale('log')
        self.init_incomplete = True
        self.ax = ax
        self.plot_u = plot_u

        try:
            self.msft = stats.umisf
            self.sprd = stats.svals
        except AttributeError:
            self.is_active = False
            not_available_text(ax, "Spectral stats not being computed")

    # Update plot
    def __call__(self, key, E, P):
        k, ko, faus = key
        ax = self.ax
        if self.init_incomplete:
            if self.plot_u or 'f' == faus:
                self.init_incomplete = False
                msft = abs(self.msft[key])
                sprd = self.sprd[key]
                if np.any(np.isinf(msft)):
                    not_available_text(ax, "Spectral stats not finite")
                    self.is_active = False
                else:
                    self.line_msft, = ax.plot(
                        msft, 'k', lw=2, label='Error')
                    self.line_sprd, = ax.plot(
                        sprd, 'b', lw=2, label='Spread', alpha=0.9)
                    ax.get_xaxis().set_major_locator(
                        MaxNLocator(integer=True))
                    ax.legend()
        else:
            msft = abs(self.msft[key])
            sprd = self.sprd[key]
            self.line_sprd.set_ydata(sprd)
            self.line_msft.set_ydata(msft)
        # ax.set_ylim(*d_ylim(msft))
        # ax.set_ylim(bottom=1e-5)
        ax.set_ylim([1e-3, 1e1])


class correlations:
    """Plots the state (auto-)correlation matrix."""

    half = True  # Whether to show half/full (symmetric) corr matrix.

    def __init__(self, fignum, stats, key0, plot_u, E, P, **kwargs):

        GS = {'height_ratios': [4, 1], 'hspace': 0.09, 'top': 0.95}
        fig, (ax, ax2) = place.freshfig(fignum, figsize=(5, 6), nrows=2, gridspec_kw=GS)

        if E is None and np.isnan(
                P.diag if isinstance(P, CovMat) else P).all():
            not_available_text(ax, (
                'Not available in replays'
                '\ncoz full Ens/Cov not stored.'))
            self.is_active = False
            return

        Nx = len(stats.mu[key0])
        if Nx <= 1003:
            C = np.eye(Nx)
            # Mask half
            mask = np.zeros_like(C, dtype=np.bool)
            mask[np.tril_indices_from(mask)] = True
            # Make colormap. Log-transform cmap,
            # but not internally in matplotlib,
            # so as to avoid transforming the colorbar too.
            cmap = plt.get_cmap('RdBu_r')
            trfm = mpl.colors.SymLogNorm(linthresh=0.2, linscale=0.2,
                                         base=np.e, vmin=-1, vmax=1)
            cmap = cmap(trfm(np.linspace(-0.6, 0.6, cmap.N)))
            cmap = mpl.colors.ListedColormap(cmap)
            #
            VM   = 1.0  # abs(np.percentile(C,[1,99])).max()
            im   = ax.imshow(C, cmap=cmap, vmin=-VM, vmax=VM)
            # Colorbar
            _ = ax.figure.colorbar(im, ax=ax, shrink=0.8)
            # Tune plot
            plt.box(False)
            ax.set_facecolor('w')
            ax.grid(False)
            ax.set_title("State correlation matrix:", y=1.07)
            ax.xaxis.tick_top()

            # ax2 = inset_axes(ax,width="30%",height="60%",loc=3)
            line_AC, = ax2.plot(arange(Nx), ones(Nx), label='Correlation')
            line_AA, = ax2.plot(arange(Nx), ones(Nx), label='Abs. corr.')
            _        = ax2.hlines(0, 0, Nx-1, 'k', 'dotted', lw=1)
            # Align ax2 with ax
            bb_AC = ax2.get_position()
            bb_C  = ax.get_position()
            ax2.set_position([bb_C.x0, bb_AC.y0, bb_C.width, bb_AC.height])
            # Tune plot
            ax2.set_title("Auto-correlation:")
            ax2.set_ylabel("Mean value")
            ax2.set_xlabel("Distance (in state indices)")
            ax2.set_xticklabels([])
            ax2.set_yticks([0, 1] + list(ax2.get_yticks()[[0, -1]]))
            ax2.set_ylim(top=1)
            ax2.legend(frameon=True, facecolor='w',
                       bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.02)

            self.ax      = ax
            self.ax2     = ax2
            self.im      = im
            self.line_AC = line_AC
            self.line_AA = line_AA
            self.mask    = mask
            if hasattr(stats, 'w'):
                self.w   = stats.w
        else:
            not_available_text(ax)

    # Update plot
    def __call__(self, key, E, P):
        # Get cov matrix
        if E is not None:
            if hasattr(self, 'w'):
                C = np.cov(E, rowvar=False, aweights=self.w[key])
            else:
                C = np.cov(E, rowvar=False)
        else:
            assert P is not None
            C = P.full if isinstance(P, CovMat) else P
            C = C.copy()
        # Compute corr from cov
        std = np.sqrt(np.diag(C))
        C  /= std[:, None]
        C  /= std[None, :]
        # Mask
        if self.half:
            C = np.ma.masked_where(self.mask, C)
        # Plot
        self.im.set_data(C)
        # Auto-corr function
        ACF = circulant_ACF(C)
        AAF = circulant_ACF(C, do_abs=True)
        self.line_AC.set_ydata(ACF)
        self.line_AA.set_ydata(AAF)


def circulant_ACF(C, do_abs=False):
    """Compute the auto-covariance-function corresponding to `C`.

    This assumes it is the cov/corr matrix of a 1D periodic domain.
    """
    M = len(C)
    # cols = np.flipud(sla.circulant(np.arange(M)[::-1]))
    cols = sla.circulant(np.arange(M))
    ACF = np.zeros(M)
    for i in range(M):
        row = C[i, cols[i]]
        if do_abs:
            row = abs(row)
        ACF += row
        # Note: this actually also accesses masked values in C.
    return ACF/M


def sliding_marginals(
    obs_inds     = (),
    dims         = (),
    labels       = (),
    Tplot        = None,
    ens_props    = dict(alpha=0.4),  # noqa
    zoomy        = 1.0,
):

    # Store parameters
    params_orig = DotDict(**locals())

    def init(fignum, stats, key0, plot_u, E, P, **kwargs):
        xx, yy, mu, spread, tseq = \
            stats.xx, stats.yy, stats.mu, stats.spread, stats.HMM.tseq

        # Set parameters (kwargs takes precedence over params_orig)
        p = DotDict(**{
            kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

        # Lag settings:
        T_lag, K_lag, a_lag = validate_lag(p.Tplot, tseq)
        K_plot = comp_K_plot(K_lag, a_lag, plot_u)
        # Extend K_plot forther for adding blanks in resampling (PartFilt):
        has_w = hasattr(stats, 'w')
        if has_w:
            K_plot += a_lag

        # Chose marginal dims to plot
        if not p.dims:
            Nx      = min(10, xx.shape[-1])
            DimsX   = linspace_int(xx.shape[-1], Nx)
        else:
            Nx      = len(p.dims)
            DimsX   = p.dims
        # Pre-process obs dimensions
        # Rm inds of obs if not in DimsX
        iiY   = [i for i, m in enumerate(p.obs_inds) if m in DimsX]
        # Rm obs_inds    if not in DimsX
        DimsY = [m for i, m in enumerate(p.obs_inds) if m in DimsX]
        # Get dim (within y) of each x
        DimsY = [DimsY.index(m) if m in DimsY else None for m in DimsX]
        Ny    = len(iiY)

        # Set up figure, axes
        fig, axs = place.freshfig(fignum, figsize=(5, 7), nrows=Nx, sharex=True)
        if Nx == 1:
            axs = [axs]

        # Tune plots
        axs[0].set_title("Marginal time series")
        for ix, (m, ax) in enumerate(zip(DimsX, axs)):
            # ax.set_ylim(*viz.stretch(*viz.xtrema(xx[:, m]), 1/p.zoomy))
            if not p.labels:
                ax.set_ylabel("$x_{%d}$" % m)
            else:
                ax.set_ylabel(p.labels[ix])
        axs[-1].set_xlabel('Time (t)')

        plot_pause(0.05)
        plt.tight_layout()

        # Allocate
        d = DotDict()  # data arrays
        h = DotDict()  # plot handles
        # Why "if True" ? Just to indent the rest of the line...
        if True:
            d.t  = RollingArray((K_plot,))
        if True:
            d.x  = RollingArray((K_plot, Nx))
            h.x  = []
        if True:
            d.y  = RollingArray((K_plot, Ny))
            h.y  = []
        if E is not None:
            d.E  = RollingArray((K_plot, len(E), Nx))
            h.E  = []
        if P is not None:
            d.mu = RollingArray((K_plot, Nx))
            h.mu = []
        if P is not None:
            d.s  = RollingArray((K_plot, 2, Nx))
            h.s  = []

        # Plot (invisible coz everything here is nan, for the moment).
        for ix, (_m, iy, ax) in enumerate(zip(DimsX, DimsY, axs)):
            if True:
                h.x  += ax.plot(d.t, d.x[:, ix], 'k')
            if iy != None:
                h.y  += ax.plot(d.t, d.y[:, iy], 'g*', ms=10)
            if 'E' in d:
                h.E  += [ax.plot(d.t, d.E[:, :, ix], **p.ens_props)]
            if 'mu' in d:
                h.mu += ax.plot(d.t, d.mu[:, ix], 'b')
            if 's' in d:
                h.s  += [ax.plot(d.t, d.s[:, :, ix], 'b--', lw=1)]

        def update(key, E, P):
            k, ko, faus = key

            EE = duplicate_with_blanks_for_resampled(E, DimsX, key, has_w)

            # Roll data array
            ind = k if plot_u else ko
            for Ens in EE:  # If E is duplicated, so must the others be.
                if 'E' in d:
                    d.E .insert(ind, Ens)
                if 'mu' in d:
                    d.mu.insert(ind, mu[key][DimsX])
                if 's' in d:
                    d.s .insert(ind, mu[key][DimsX] + [[1], [-1]]*spread[key][DimsX])
                if True:
                    d.t .insert(ind, tseq.tt[k])
                if True:
                    d.y .insert(ind, yy[ko, iiY]
                                if ko is not None else nan*ones(Ny))
                if True:
                    d.x .insert(ind, xx[k, DimsX])

            # Update graphs
            for ix, (_m, iy, ax) in enumerate(zip(DimsX, DimsY, axs)):
                sliding_xlim(ax, d.t, T_lag, True)
                if True:
                    h.x[ix]    .set_data(d.t, d.x[:, ix])
                if iy != None:
                    h.y[iy]    .set_data(d.t, d.y[:, iy])
                if 'mu' in d:
                    h.mu[ix]   .set_data(d.t, d.mu[:, ix])
                if 's' in d:
                    [h.s[ix][b].set_data(d.t, d.s[:, b, ix]) for b in [0, 1]]
                if 'E' in d:
                    [h.E[ix][n].set_data(d.t, d.E[:, n, ix]) for n in range(len(E))]
                if 'E' in d:
                    update_alpha(key, stats, h.E[ix])

                # TODO 3: fixup. This might be slow?
                # In any case, it is very far from tested.
                # Also, relim'iting all of the time is distracting.
                # Use d_ylim?
                if 'E' in d:
                    lims = d.E
                elif 'mu' in d:
                    lims = d.mu
                lims = np.array(viz.xtrema(lims[..., ix]))
                if lims[0] == lims[1]:
                    lims += [-.5, +.5]
                ax.set_ylim(*viz.stretch(*lims, 1/p.zoomy))

            return
        return update
    return init


def phase_particles(
    is_3d        = True,
    obs_inds     = (),
    dims         = (),
    labels       = (),
    Tplot        = None,
    ens_props    = dict(alpha=0.4),  # noqa
    zoom         = 1.5,
):

    # Store parameters
    params_orig = DotDict(**locals())

    M = 3 if is_3d else 2

    def init(fignum, stats, key0, plot_u, E, P, **kwargs):
        xx, yy, mu, _, tseq = \
            stats.xx, stats.yy, stats.mu, stats.spread, stats.HMM.tseq

        # Set parameters (kwargs takes precedence over params_orig)
        p = DotDict(**{
            kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

        # Lag settings:
        has_w = hasattr(stats, 'w')
        if p.Tplot == 0:
            K_plot = 1
        else:
            T_lag, K_lag, a_lag = validate_lag(p.Tplot, tseq)
            K_plot = comp_K_plot(K_lag, a_lag, plot_u)
            # Extend K_plot forther for adding blanks in resampling (PartFilt):
            if has_w:
                K_plot += a_lag

        # Dimension settings
        if not p.dims:
            p.dims   = arange(M)
        if not p.labels:
            p.labels = ["$x_%d$" % d for d in p.dims]
        assert len(p.dims) == M

        # Set up figure, axes
        fig, _ = place.freshfig(fignum, figsize=(5, 5))
        ax = plt.subplot(111, projection='3d' if is_3d else None)
        ax.set_facecolor('w')
        ax.set_title("Phase space trajectories")
        # Tune plot
        for ind, (s, i, t) in enumerate(zip(p.labels, p.dims, "xyz")):
            viz.set_ilim(ax, ind, *viz.stretch(*viz.xtrema(xx[:, i]), 1/p.zoom))
            eval("ax.set_%slabel('%s')" % (t, s))

        # Allocate
        d = DotDict()  # data arrays
        h = DotDict()  # plot handles
        s = DotDict()  # scatter handles
        if E is not None:
            d.E  = RollingArray((K_plot, len(E), M))
            h.E = []
        if P is not None:
            d.mu = RollingArray((K_plot, M))
        if True:
            d.x  = RollingArray((K_plot, M))
        if list(p.obs_inds) == list(p.dims):
            d.y  = RollingArray((K_plot, M))

        # Plot tails (invisible coz everything here is nan, for the moment).
        if 'E' in d:
            h.E  += [ax.plot(*xn, **p.ens_props)[0]
                     for xn in np.transpose(d.E, [1, 2, 0])]
        if 'mu' in d:
            h.mu  = ax.plot(*d.mu.T, 'b', lw=2)[0]
        if True:
            h.x   = ax.plot(*d.x .T, 'k', lw=3)[0]
        if 'y' in d:
            h.y   = ax.plot(*d.y .T, 'g*', ms=14)[0]

        # Scatter. NB: don't init with nan's coz it's buggy
        # (wrt. get_color() and _offsets3d) since mpl 3.1.
        if 'E' in d:
            s.E   = ax.scatter(*E.T[p.dims], s=3**2,
                               c=[hn.get_color() for hn in h.E])
        if 'mu' in d:
            s.mu  = ax.scatter(*ones(M), s=8**2,
                               c=[h.mu.get_color()])
        if True:
            s.x  = ax.scatter(*ones(M), s=14**2,
                              c=[h.x.get_color()], marker=(5, 1), zorder=99)

        def update(key, E, P):
            k, ko, faus = key
            show_y = 'y' in d and ko is not None

            def update_tail(handle, newdata):
                handle.set_data(newdata[:, 0], newdata[:, 1])
                if is_3d:
                    handle.set_3d_properties(newdata[:, 2])

            def update_sctr(handle, newdata):
                if is_3d:
                    handle._offsets3d = juggle_axes(*newdata.T, 'z')
                else:
                    handle.set_offsets(newdata)

            EE = duplicate_with_blanks_for_resampled(E, p.dims, key, has_w)

            # Roll data array
            ind = k if plot_u else ko
            for Ens in EE:  # If E is duplicated, so must the others be.
                if 'E' in d:
                    d.E .insert(ind, Ens)
                if True:
                    d.x .insert(ind, xx[k, p.dims])
                if 'y' in d:
                    d.y .insert(ind, yy[ko, :] if show_y else nan*ones(M))
                if 'mu' in d:
                    d.mu.insert(ind, mu[key][p.dims])

            # Update graph
            update_sctr(s.x, d.x[[-1]])
            update_tail(h.x, d.x)
            if 'y' in d:
                update_tail(h.y, d.y)
            if 'mu' in d:
                update_sctr(s.mu, d.mu[[-1]])
                update_tail(h.mu, d.mu)
            else:
                update_sctr(s.E, d.E[-1])
                for n in range(len(E)):
                    update_tail(h.E[n], d.E[:, n, :])
                update_alpha(key, stats, h.E, s.E)

            return
        return update
    return init


def validate_lag(Tplot, tseq):
    """Return validated `T_lag` such that is is:

    - equal to `Tplot` with fallback: `HMM.tseq.Tplot`.
    - no longer than `HMM.tseq.T`.

    Also return corresponding `K_lag`, `a_lag`.
    """
    # Defaults
    if Tplot is None:
        Tplot = tseq.Tplot

    # Rename
    T_lag = Tplot

    assert T_lag >= 0

    # Validate T_lag
    t2 = tseq.tt[-1]
    t1 = max(tseq.tt[0], t2-T_lag)
    T_lag = t2-t1

    K_lag = int(T_lag / tseq.dt) + 1  # Lag in indices
    a_lag = K_lag//tseq.dko + 1     # Lag in obs indices

    return T_lag, K_lag, a_lag


def comp_K_plot(K_lag, a_lag, plot_u):
    K_plot = 2*a_lag     # Sum of lags of {f,a} series.
    if plot_u:
        K_plot += K_lag  # Add lag of u series.
    return K_plot


def update_alpha(key, stats, lines, scatters=None):
    """Adjust color alpha (for particle filters)."""
    k, ko, faus = key
    if ko is None:
        return
    if faus == 'f':
        return
    if not hasattr(stats, 'w'):
        return

    # Compute alpha values
    w     = stats.w[key]
    alpha = (w/w.max()).clip(0.1, 0.4)

    # Set line alpha
    for line, a in zip(lines, alpha):
        line.set_alpha(a)

    # Scatter plot does not have alpha. => Fake it.
    if scatters is not None:
        colors = scatters.get_facecolor()[:, :3]
        if len(colors) == 1:
            colors = colors.repeat(len(w), axis=0)
        scatters.set_color(np.hstack([colors, alpha[:, None]]))


def duplicate_with_blanks_for_resampled(E, dims, key, has_w):
    """Particle filter: insert breaks for resampled particles."""
    if E is None:
        return [E]
    EE = []
    E  = E[:, dims]
    if has_w:
        k, ko, faus = key
        if faus == 'f':
            pass
        elif faus == 'a':
            _Ea[0] = E[:, 0]  # Store (1st dim of) ens.
        elif faus == 'u' and ko is not None:
            # Find resampled particles. Insert duplicate ensemble. Write nans (breaks).
            resampled = _Ea[0] != E[:, 0]  # Mark as resampled if ens changed.
            # Insert current ensemble (copy to avoid overwriting).
            EE.append(E.copy())
            EE[0][resampled] = nan  # Write breaks
    # Always: append current ensemble
    EE.append(E)
    return EE


_Ea = [None]  # persistent storage for ens


def d_ylim(data, ax=None, cC=0, cE=1, pp=(1, 99), Min=-1e20, Max=+1e20):
    """Provide new ylim's intelligently, from percentiles of the data.

    - `data`: iterable of arrays for computing percentiles.
    - `pp`: percentiles

    - `ax`: If present, then the delta_zoom in/out is also considered.

      - `cE`: exansion (widenting) rate ∈ [0,1].
        Default: 1, which immediately expands to percentile.
      - `cC`: compression (narrowing) rate ∈ [0,1].
        Default: 0, which does not allow compression.

    - `Min`/`Max`: bounds

    Despite being a little involved,
    the cost of this subroutine is typically not substantial
    because there's usually not that much data to sort through.
    """
    # Find "reasonable" limits (by percentiles), looping over data
    maxv = minv = -np.inf  # init
    for d in data:
        d = d[np.isfinite(d)]
        if len(d):
            perc = np.array([-1, 1]) * np.percentile(d, pp)
            minv, maxv = np.maximum([minv, maxv], perc)
    minv *= -1

    # Pry apart equal values
    if np.isclose(minv, maxv):
        maxv += 0.5
        minv -= 0.5

    # Make the zooming transition smooth
    if ax is not None:
        current = ax.get_ylim()
        # Set rate factor as compress or expand factor.
        c0 = cC if minv > current[0] else cE
        c1 = cC if maxv < current[1] else cE
        # Adjust
        minv = np.interp(c0, (0, 1), (current[0], minv))
        maxv = np.interp(c1, (0, 1), (current[1], maxv))

    # Bounds
    maxv = min(Max, maxv)
    minv = max(Min, minv)

    # Set (if anything's changed)
    def worth_updating(a, b, curr):
        # Note: should depend on cC and cE
        d = abs(curr[1]-curr[0])
        lower = abs(a-curr[0]) > 0.002*d
        upper = abs(b-curr[1]) > 0.002*d
        return lower and upper
    # if worth_updating(minv,maxv,current):
        # ax.set_ylim(minv,maxv)

    # Some mpl versions don't handle inf limits.
    if not np.isfinite(minv):
        minv = None
    if not np.isfinite(maxv):
        maxv = None

    return minv, maxv


def spatial1d(
    obs_inds     = None,
    periodicity  = None,
    dims         = (),
    ens_props    = {'color': 'b', 'alpha': 0.1},  # noqa
    conf_mult    = None,
):

    # Store parameters
    params_orig = DotDict(**locals())

    def init(fignum, stats, key0, plot_u, E, P, **kwargs):
        xx, yy, mu = stats.xx, stats.yy, stats.mu

        # Set parameters (kwargs takes precedence over params_orig)
        p = DotDict(**{
            kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

        if not p.dims:
            M = xx.shape[-1]
            p.dims = arange(M)
        else:
            M = len(p.dims)

        # Make periodic wrapper
        ii, wrap = viz.setup_wrapping(M, p.periodicity)

        # Set up figure, axes
        fig, ax = place.freshfig(fignum, figsize=(8, 5))
        fig.suptitle("1d amplitude plot")

        # Nans
        nan1 = wrap(nan*ones(M))

        if E is None and p.conf_mult is None:
            p.conf_mult = 2

        # Init plots
        if p.conf_mult:
            lines_s  = ax.plot(ii, nan1, "b-", lw=1,
                               label=(str(p.conf_mult) + r'$\sigma$ conf'))
            lines_s += ax.plot(ii, nan1, "b-", lw=1)
            line_mu, = ax.plot(ii, nan1, 'b-', lw=2, label='DA mean')
        else:
            nanE     = nan*ones((stats.xp.N, M))
            lines_E  = ax.plot(ii, wrap(nanE[0]), **p.ens_props, lw=1, label='Ensemble')
            lines_E += ax.plot(ii, wrap(nanE[1:]).T, **p.ens_props, lw=1)
        # Truth, Obs
        (line_x, )   = ax.plot(ii, nan1, 'k-', lw=3, label='Truth')
        if p.obs_inds is not None:
            p.obs_inds = np.asarray(p.obs_inds)
            (line_y, ) = ax.plot(p.obs_inds, nan*p.obs_inds, 'g*', ms=5, label='Obs')

        # Tune plot
        ax.set_ylim(*viz.xtrema(xx))
        ax.set_xlim(viz.stretch(ii[0], ii[-1], 1))
        # Xticks
        xt = ax.get_xticks()
        xt = xt[abs(xt % 1) < 0.01].astype(int)  # Keep only the integer ticks
        xt = xt[xt >= 0]
        xt = xt[xt < len(p.dims)]
        ax.set_xticks(xt)
        ax.set_xticklabels(p.dims[xt])

        ax.set_xlabel('State index')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

        text_t = ax.text(0.01, 0.01, format_time(None, None, None),
                         transform=ax.transAxes, family='monospace', ha='left')

        # Init visibility (must come after legend):
        if p.obs_inds is not None:
            line_y.set_visible(False)

        def update(key, E, P):
            k, ko, faus = key

            if p.conf_mult:
                sigma = mu[key] + p.conf_mult * stats.spread[key] * [[1], [-1]]
                lines_s[0].set_ydata(wrap(sigma[0, p.dims]))
                lines_s[1].set_ydata(wrap(sigma[1, p.dims]))
                line_mu   .set_ydata(wrap(mu[key][p.dims]))
            else:
                for n, line in enumerate(lines_E):
                    line.set_ydata(wrap(E[n, p.dims]))
                update_alpha(key, stats, lines_E)

            line_x.set_ydata(wrap(xx[k, p.dims]))

            text_t.set_text(format_time(k, ko, stats.HMM.tseq.tt[k]))

            if 'f' in faus:
                if p.obs_inds is not None:
                    line_y.set_ydata(yy[ko])
                    line_y.set_zorder(5)
                    line_y.set_visible(True)

            if 'u' in faus:
                if p.obs_inds is not None:
                    line_y.set_visible(False)

            return
        return update
    return init


def spatial2d(
    square,
    ind2sub,
    obs_inds = None,
    cm       = plt.cm.jet,
    clims    = ((-40, 40), (-40, 40), (-10, 10), (-10, 10)),
):

    def init(fignum, stats, key0, plot_u, E, P, **kwargs):

        GS = {'left': 0.125-0.04, 'right': 0.9-0.04}
        fig, axs = place.freshfig(fignum, figsize=(6, 6),
                                  nrows=2, ncols=2, sharex=True, sharey=True,
                                  gridspec_kw=GS)

        for ax in axs.flatten():
            ax.set_aspect('equal', 'box')

        ((ax_11, ax_12), (ax_21, ax_22)) = axs

        ax_11.grid(color='w', linewidth=0.2)
        ax_12.grid(color='w', linewidth=0.2)
        ax_21.grid(color='k', linewidth=0.1)
        ax_22.grid(color='k', linewidth=0.1)

        # Upper colorbar -- position relative to ax_12
        bb    = ax_12.get_position()
        dy    = 0.1*bb.height
        ax_13 = fig.add_axes([bb.x1+0.03, bb.y0 + dy, 0.04, bb.height - 2*dy])
        # Lower colorbar -- position relative to ax_22
        bb    = ax_22.get_position()
        dy    = 0.1*bb.height
        ax_23 = fig.add_axes([bb.x1+0.03, bb.y0 + dy, 0.04, bb.height - 2*dy])

        # Extract data arrays
        xx, _, mu, spread, err = stats.xx, stats.yy, stats.mu, stats.spread, stats.err
        k = key0[0]
        tt = stats.HMM.tseq.tt

        # Plot
        # - origin='lower' might get overturned by set_ylim() below.
        im_11 = ax_11.imshow(square(mu[key0]), cmap=cm)
        im_12 = ax_12.imshow(square(xx[k]), cmap=cm)
        # hot is better, but needs +1 colorbar
        im_21 = ax_21.imshow(square(spread[key0]), cmap=plt.cm.bwr)
        im_22 = ax_22.imshow(square(err[key0]), cmap=plt.cm.bwr)
        ims = (im_11, im_12, im_21, im_22)
        # Obs init -- a list where item 0 is the handle of something invisible.
        lh = list(ax_12.plot(0, 0)[0:1])

        sx = '$\\psi$'
        ax_11.set_title('mean '    + sx)
        ax_12.set_title('true '    + sx)
        ax_21.set_title('spread. ' + sx)
        ax_22.set_title('err. '    + sx)

        # TODO 7
        # for ax in axs.flatten():
        # Crop boundries (which should be 0, i.e. yield harsh q gradients):
        # lims = (1, nx-2)
        # step = (nx - 1)/8
        # ticks = arange(step,nx-1,step)
        # ax.set_xlim  (lims)
        # ax.set_ylim  (lims[::-1])
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)

        for im, clim in zip(ims, clims):
            im.set_clim(clim)

        fig.colorbar(im_12, cax=ax_13)
        fig.colorbar(im_22, cax=ax_23)
        for ax in [ax_13, ax_23]:
            ax.yaxis.set_tick_params('major', length=2, width=0.5,
                                     direction='in', left=True, right=True)
            ax.set_axisbelow('line')  # make ticks appear over colorbar patch

        # Title
        title = "Streamfunction ("+sx+")"
        fig.suptitle(title)
        # Time info
        text_t = ax_12.text(1, 1.1, format_time(None, None, None),
                            transform=ax_12.transAxes, family='monospace', ha='left')

        def update(key, E, P):
            k, ko, faus = key
            t = tt[k]

            im_11.set_data(square(mu[key]))
            im_12.set_data(square(xx[k]))
            im_21.set_data(square(spread[key]))
            im_22.set_data(square(err[key]))

            # Remove previous obs
            try:
                lh[0].remove()
            except ValueError:
                pass
            # Plot current obs.
            #  - plot() automatically adjusts to direction of y-axis in use.
            #  - ind2sub returns (iy,ix), while plot takes (ix,iy) => reverse.
            if ko is not None and obs_inds is not None:
                lh[0] = ax_12.plot(*ind2sub(obs_inds(t))[::-1], 'k.', ms=1, zorder=5)[0]

            text_t.set_text(format_time(k, ko, t))

            return
        return update
    return init


# List of liveplotters available for all HMMs.
# Columns:
# - fignum
# - show_by_default
# - function/class
default_liveplotters = [
    (1, sliding_diagnostics),
    (1, weight_histogram),
]
