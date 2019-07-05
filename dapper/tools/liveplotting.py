from dapper import *


class LivePlot:
  """Live plotting manager.

  Deals with
   - Pause, skip.
   - Which liveploters to call.
   - plot_u
   - Figure window (title and number)."""
  def __init__(self,stats,liveplotting,key0=(0,None,'u'),E=None,P=None,speed=1.0,**kwargs):
    """
    Initialize plots.
    - figlist: figures to plot; alternatives:
      - "default"/[]/True: All default figures for this HMM.
      - "all"            : Even more.
      - non-empty list   : Only the figures with these numbers (int) or names (str).
      - False            : None.
    - speed: speed of animation.
        - np.inf : instantaneous (works well for replays)
        - 1      : default (as quick as possible while allowing for plt.draw())
        - below 1: slower."""

    # Set speed/pause 
    self.params = {
        'pause_f' : 0.05,
        'pause_a' : 0.05,
        'pause_s' : 0.05,
        'pause_u' : 0.001,
        }
    for pause in ["pause_"+x for x in "fau"]:
      self.params[pause] /= speed
    # Write params
    self.params.update(getattr(stats.config, "LP_kwargs", dict()))
    self.params.update(kwargs)

    def get_name(init):
      "Get name of liveplotter function/class."
      try:                    return init.__qualname__.split(".")[0]
      except AttributeError:  return init.__class__.__name__

    # Set up dict of liveplotters
    potential_LPs = OrderedDict()
    for num, show, init in default_liveplotters:
      potential_LPs[get_name(init)] = num, show, init
    # Add HMM-specific liveplotters
    for num, show, init in getattr(stats.HMM,'liveplotters',dict()):
      assert num>10, "Liveplotters specified in the HMM should have fignum>10."
      potential_LPs[get_name(init)] = num, show, init

    figlist = parse_figlist(liveplotting)

    # On the 2nd run (of example_1 e.g.) the figures don't appear
    # if they've been closed. For some reason, this fixes it:
    plt.ion()

    # Loop over requeted figures
    self.any_figs = False
    self.figures = OrderedDict()
    for name, (num, show_by_default, init) in potential_LPs.items():
      if (num in figlist) or (name in figlist) or (figlist==[] and show_by_default):

        # Startup message
        if not self.any_figs:
          print('Initializing liveplotting...')
          print('Hit <Space> to pause/step.')
          print('Hit <Enter> to resume/skip.')
          self.paused = False
          self.skipping = False
          self.any_figs = True

        # Init figure
        self.plot_u = plot_u(stats.mu,key0)
        post_title = "" if self.plot_u else "\n(obs times only)"
        updater = init(num,stats,key0,self.plot_u,E,P,**kwargs)
        if plt.fignum_exists(num) and getattr(updater,'is_active',1):
            self.figures[name] = (num, updater)
            fig = plt.figure(num)
            win = fig.canvas
            ax0 = fig.axes[0]
            win.set_window_title("%s [%d]"%(name,num))
            ax0.set_title(ax0.get_title() + post_title)
            self.update(key0,E,P) # Call initial update
            plot_pause(0.01)      # Draw


  def update(self,key,E,P):
    """Update liveplots"""

    # Check if there are still open figures
    if self.any_figs:
      open_figns = plt.get_fignums()
      live_figns = set(num for (num, updater) in self.figures.values())
      self.any_figs = bool(live_figns.intersection(open_figns))
    # If no open figures: don't update
    if not self.any_figs:
      return


    # Playback control
    SPACE  = b' '   
    ENTERs = [b'\n', b'\r'] # Linux + Windows
    def pause():
      "Loop until user decision is made."
      ch = read1() 
      while True:
        if ch in ENTERs:
          self.paused = False
        if ch in ENTERs + [SPACE]:
          break
        ch = read1()
        # Pause to enable zoom, pan, etc. of mpl GUI
        plot_pause(0.01) # Don't use time.sleep()!
    #
    if self.paused:
      pause()
    else:
      if key==(0,None,'u'):
        pass # Skip read1 for key0 (coz it blocks)
      else:
        # Set switches for pause & skipping
        ch = read1()
        if ch==SPACE: # Turn ON pause & turn OFF skipping.
          self.paused = True
          self.skipping = False
          pause()
        elif ch in ENTERs: # Toggle skipping
          self.skipping = not self.skipping 
        
    if not self.skipping:
      # Update figures
      f_a_u = key[2]
      if f_a_u is not 'u' or self.plot_u:
        for name, (num, updater) in self.figures.items():
          if plt.fignum_exists(num) and getattr(updater,'is_active',1):
            fig = plt.figure(num)
            updater(key,E,P)
            plot_pause(self.params['pause_'+f_a_u])

def parse_figlist(figlist):
  "Figures requested for this config. Convert to list."
  if isinstance(figlist,str):
    fn = figlist.lower()                               # Yields:
    if   "all" == fn:              figlist = range(99) # All potential_LPs
    elif "default" in fn:          figlist = []        # All show_by_default
  elif hasattr(figlist,'__len__'): figlist = figlist   # This list (only)
  elif figlist:                    figlist = []        # All show_by_default
  else:                            figlist = [None]    # None
  return figlist





def replay(stats, figlist=None, speed=np.inf, t1=0, t2=None, **kwargs):
  """Replay LivePlot with what's been stored in 'stats'.

  - t1, t2: time window to plot.
  - 'figlist' and 'speed': See LivePlot's doc.

  .. note:: store_u (specify in the config to store intermediate stats)
            must have been True to have smooth graphs as in the actual LivePlot.

  .. note:: Ensembles are generally not stored in the stats and so cannot be replayed.
  """

  # Time settings
  chrono = stats.HMM.t
  if t2 is None:
    t2 = t1 + chrono.Tplot

  # Ens does not get stored in stats, so we cannot replay that.
  # If the LPs are initialized with P0!=None, then they will avoid ens plotting.
  # TODO: This system for switching from Ens to stats must be replaced.
  #       It breaks down when M is very large.
  P0 = np.full_like(stats.HMM.X0.C.full, nan) 

  if figlist is None: figlist = stats.config.liveplotting
  figlist = parse_figlist(figlist)

  LP = LivePlot(stats, figlist, P=P0, speed=speed, Tplot=t2-t1, **kwargs)

  # Remember: must use progbar to unblock read1.
  # Let's also make a proper description.
  desc = stats.config.da_method.__name__ + " (replay)"

  # Play through assimilation cycles
  for k,kObs,t,dt in progbar(chrono.ticker, desc):
    if t1 <= t <= t2:
      if kObs is not None:
        LP.update((k,kObs,'f'),None,None)
        LP.update((k,kObs,'a'),None,None)
      LP.update((k,kObs,'u'),None,None)



# TODO:
# - iEnKS diagnostics don't work at all when store_u=False
star = "${}^*$"
class sliding_diagnostics:

  def __init__(self,fignum,stats,key0,_,E,P,Tplot=None,**kwargs):
      GS = {'left':0.125,'right':0.76}
      fig, (ax1, ax2) = freshfig(fignum, (5,3.5), loc='2311', nrows=2, sharex=True, gridspec_kw=GS)

      ax1.set_title("Diagnostics")
      ax1.set_ylabel('RMS')
      ax2.set_ylabel('Values') 
      ax2.set_xlabel('Time (t)')
      adjust_position(ax2, y0=0.03)

      self.T_lag, K_lag, a_lag = validate_lag(Tplot, stats.HMM.t)

      def init_ax(ax,style_table):
        plotted_lines = OrderedDict()
        for name in style_table:

            # SKIP -- if stats[name] is not in existence
            # Note: The nan check/deletion comes after the first kObs.
            try: stat = getattr(stats,name)
            except AttributeError: continue
            # try: val0 = stat[key0[0]]
            # except KeyError: continue
            # PS: recall (from series.py) that even if store_u is false, stat[k] is
            # still present if liveplotting=True via the k_tmp functionality.
            
            # Unpack style
            ln = {}
            ln['transf'] = style_table[name][0]
            ln['shape']  = style_table[name][1]
            ln['plt']    = style_table[name][2]

            # Create series
            if isinstance(stat,FAU_series):
              ln['plot_u'] = plot_u(stat,key0)
              K_plot       = comp_K_plot(K_lag,a_lag,ln['plot_u'])
            else:
              ln['plot_u'] = False
              K_plot       = a_lag
            ln['data']   = RollingArray(K_plot)
            ln['tt']     = RollingArray(K_plot)

            # Plot (init)
            ln['handle'], = ax.plot(ln['tt'],ln['data'],**ln['plt'])

            # Plotting only nans yield ugly limits. Revert to defaults.
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)

            plotted_lines[name] = ln
        return plotted_lines

      # Plot
      self.d1 = init_ax(ax1, stats.style1);
      self.d2 = init_ax(ax2, stats.style2);

      # Horizontal line at y=0
      self.baseline0, = ax2.plot(ax2.get_xlim(),[0,0],c=0.5*ones(3),lw=0.7,label='_nolegend_')

      # Store
      self.ax1 = ax1
      self.ax2 = ax2
      self.stats = stats
      self.init_incomplete = True

  # Update plot
  def __call__(self,key,E,P):
      k, kObs, f_a_u = key

      stats  = self.stats
      chrono = stats.HMM.t
      ax1    = self.ax1
      ax2    = self.ax2

      def update_arrays(plotted_lines):
        for name, ln in plotted_lines.items():
          stat = getattr(stats,name)
          t    = chrono.tt[k] # == chrono.ttObs[kObs]
          if isinstance(stat,FAU_series):
            # ln['data'] will contain duplicates for f/a times.
            if ln['plot_u']:
              val = stat[key]
              ln['tt']  .insert(k   , t)
              ln['data'].insert(k   , ln['transf'](val))
            elif 'u' not in f_a_u:
              val = stat[key]
              ln['tt']  .insert(kObs, t)
              ln['data'].insert(kObs, ln['transf'](val))
          else:
            # ln['data'] will not contain duplicates, coz only 'a' is input.
            if 'a' in f_a_u: 
              val = stat[kObs]
              ln['tt']  .insert(kObs, t)
              ln['data'].insert(kObs, ln['transf'](val))
            elif 'f' in f_a_u:
              pass

      def update_plot_data(ax,plotted_lines):

          def bend_into(shape, xx, yy):
            # Get arrays. Repeat (to use for intermediate nodes). 
            yy = yy.array.repeat(3)
            xx = xx.array.repeat(3)
            if len(xx)==0:
              pass # shortcircuit any modifications
            elif shape == 'step':
              yy = np.hstack([yy[1:], nan]) # roll leftward
            elif shape == 'dirac':
              nonlocal nDirac
              axW      = np.diff(ax.get_xlim())
              yy[0::3] = False          # set datapoin to 0
              xx[2::3] = nan            # make datapoint disappear
              xx      += nDirac*axW/100 # offset datapoint horizontally
              nDirac  +=1
            return xx, yy

          nDirac = 1
          for name, ln in plotted_lines.items():
            ln['handle'].set_data(*bend_into(ln['shape'], ln['tt'], ln['data']))

      def finalize_init(ax,plotted_lines,mm):
          # Rm lines that only contain NaNs
          for name in list(plotted_lines):
            ln = plotted_lines[name]
            if not np.any(np.isfinite(ln['data'])):
              ln['handle'].remove()
              del plotted_lines[name]
          # Add legends
          if plotted_lines:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),borderaxespad=0)
            if mm:
              ax.annotate(star+": mean of\nmarginals",
                  xy=(0,-1.5/len(plotted_lines)),
                  xycoords=ax.get_legend().get_frame(),
                  bbox=dict(alpha=0.0), fontsize='small')
          plot_pause(0.01) # coz placement of annotate needs flush sometimes

      # Insert current stats
      update_arrays(self.d1)
      update_arrays(self.d2)

      # Plot
      update_plot_data(ax1, self.d1)
      update_plot_data(ax2, self.d2)

      # Set x-limits (time)
      sliding_xlim(ax1, self.d1['rmse']['tt'], self.T_lag, margin=True)
      self.baseline0.set_xdata(ax1.get_xlim())

      # Set y-limits
      data1 = [ln['data'].array for ln in self.d1.values()]
      data2 = [ln['data'].array for ln in self.d2.values()]
      ax1.set_ylim(0, d_ylim(data1, ax1,                cC=0.2,cE=0.9)[1])
      ax2.set_ylim(  *d_ylim(data2, ax2, Max=4, Min=-4, cC=0.3,cE=0.9))

      # Init legend. Rm nan lines.
      if self.init_incomplete and 'a'==f_a_u:
         self.init_incomplete = False
         finalize_init(ax1, self.d1, False)
         finalize_init(ax2, self.d2, True)



def sliding_xlim(ax, tt, lag, margin=False):
  dt = lag/20 if margin else 0
  if tt.nFilled==0: return        # Quit
  t1, t2 = tt.span()              # Get suggested span.
  s1, s2 = ax.get_xlim()          # Get previous lims.
  if t1==t2:                      # If zero span (eg tt holds single 'f' and 'a'):
    t1 -= 1                       #   add width
    t2 += 1                       #   add width
  elif np.isnan(t1):              # If user has skipped (too much):
    s2    -= dt                   #   Correct for dt.
    span   = s2-s1                #   Compute previous span
    if span < lag:                #   If span<lag:
      span  += (t2-s2)            #     Grow by "dt".
    span   = min(lag, span)       #   Bound
    t1     = t2 - span            #   Set span.
  ax.set_xlim(t1, t2 + dt)        # Set xlim to span


class weight_histogram:

  def __init__(self,fignum,stats,key0,plot_u,E,P,**kwargs):
    if not hasattr(stats,'w'):
      self.is_active = False
      return
    fig, ax = freshfig(fignum, (7,3), loc='3323', gridspec_kw={'bottom':.15})

    ax.set_xscale('log')
    ax.set_xlabel('Weigth')
    ax.set_ylabel('Count')
    self.stats = stats
    self.ax    = ax
    self.init_incomplete = True

  def __call__(self,key,E,P):
    k,kObs,f_a_u = key
    if 'a'==f_a_u:
      w  = self.stats.w[key]
      N  = len(w)
      ax = self.ax

      if self.init_incomplete:
        self.init_incomplete = False
        self.is_active = N<10001
        self.bins = exp( linspace( log(1e-10), log(1), 31 ) )
        if not self.is_active:
          not_available_text(ax,'Not computed (N > threshold)')
          return
      else:
        _ = [b.remove() for b in self.hist]

      counted = w>self.bins[0]
      nn,_,self.hist = ax.hist(w[counted], bins=self.bins, color='b')
      ax.set_ylim(top=max(nn))

      ax.set_title('N: {:d}.   N_eff: {:.4g}.   Not shown: {:d}. '.\
          format(N, 1/(w@w), N-np.sum(counted)))


class spectral_errors:

  def __init__(self,fignum,stats,key0,plot_u,E,P,**kwargs):
    fig, ax = freshfig(fignum, (6,3), loc='3333')
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
  def __call__(self,key,E,P):
    k,kObs,f_a_u = key
    ax = self.ax
    if self.init_incomplete:
      if self.plot_u or 'f'==f_a_u:
        self.init_incomplete = False
        msft = abs(self.msft[key])
        sprd =     self.sprd[key]
        if np.any(np.isinf(msft)):
          not_available_text(ax, "Spectral stats not finite")
          self.is_active = False
        else:
          self.line_msft, = ax.plot(msft,'k',lw=2,label='Error')
          self.line_sprd, = ax.plot(sprd,'b',lw=2,label='Spread',alpha=0.9)
          ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
          ax.legend()
    else:
      msft = abs(self.msft[key])
      sprd =     self.sprd[key]
      self.line_sprd.set_ydata(sprd)
      self.line_msft.set_ydata(msft)
    # ax.set_ylim(*d_ylim(msft))
    # ax.set_ylim(bottom=1e-5)
    ax.set_ylim([1e-3,1e1])


class correlations:
  half = True # Whether to show half/full (symmetric) corr matrix.

  def __init__(self,fignum,stats,key0,plot_u,E,P,**kwargs):

    GS = {'height_ratios':[4, 1],'hspace':0.09,'top':0.95}
    fig, (ax,ax2) = freshfig(fignum, (5,6), loc='2321', nrows=2, gridspec_kw=GS)

    if E is None and np.isnan(P.diag if isinstance(P,CovMat) else P).all():
      not_available_text(ax,'Not available in replays\ncoz full Ens/Cov not stored.')
      self.is_active = False
      return

    Nx = len(stats.mu[key0])
    if Nx<=1003:
      C = eye(Nx)
      # Mask half
      mask = np.zeros_like(C, dtype=np.bool)
      mask[np.tril_indices_from(mask)] = True
      # Make colormap. Log-transform cmap, but not internally in matplotlib,
      # so as to avoid transforming the colorbar too.
      cmap = plt.get_cmap('RdBu')
      trfm = colors.SymLogNorm(linthresh=0.2,linscale=0.2,vmin=-1, vmax=1)
      cmap = cmap(trfm(linspace(-0.6,0.6,cmap.N)))
      cmap = colors.ListedColormap(cmap)
      #
      VM   = 1.0 # abs(np.percentile(C,[1,99])).max()
      im   = ax.imshow(C,cmap=cmap,vmin=-VM,vmax=VM)
      # Colorbar
      cax = ax.figure.colorbar(im,ax=ax,shrink=0.8)
      # Tune plot
      plt.box(False)
      ax.set_facecolor('w') 
      ax.grid(False)
      ax.set_title("State correlation matrix:", y=1.07)
      ax.xaxis.tick_top()
      
      # ax2 = inset_axes(ax,width="30%",height="60%",loc=3)
      line_AC, = ax2.plot(arange(Nx), ones(Nx), label='Correlation')
      line_AA, = ax2.plot(arange(Nx), ones(Nx), label='Abs. corr.')
      _        = ax2.hlines(0,0,Nx-1,'k','dotted',lw=1)
      # Align ax2 with ax
      bb_AC = ax2.get_position()
      bb_C  = ax.get_position()
      ax2.set_position([bb_C.x0, bb_AC.y0, bb_C.width, bb_AC.height])
      # Tune plot
      ax2.set_title("Auto-correlation:")
      ax2.set_ylabel("Mean value")
      ax2.set_xlabel("Distance (in state indices)")
      ax2.set_xticklabels([])
      ax2.set_yticks([0,1] + list(ax2.get_yticks()[[0,-1]]))
      ax2.set_ylim(top=1)
      ax2.legend(frameon=True,facecolor='w',
          bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.02)

      self.ax      = ax
      self.ax2     = ax2
      self.im      = im
      self.line_AC = line_AC
      self.line_AA = line_AA
      self.mask    = mask
    else:
      not_available_text(ax)

  # Update plot
  def __call__(self,key,E,P):
    # Get cov matrix
    if E is not None:
      C = np.cov(E,rowvar=False)
    else:
      assert P is not None
      C = P.full if isinstance(P,CovMat) else P
      C = C.copy()
    # Compute corr from cov
    std = sqrt(diag(C))
    C  /= std[:,None]
    C  /= std[None,:]
    # Mask
    if self.half:
      C = np.ma.masked_where(self.mask, C)
    # Plot
    self.im.set_data(C)
    # Auto-corr function
    ACF = circulant_ACF(C)
    AAF = circulant_ACF(C,do_abs=True)
    self.line_AC.set_ydata(ACF)
    self.line_AA.set_ydata(AAF)




def sliding_marginals(
    obs_inds     = [],
    dims         = [],
    labels       = [],
    Tplot        = None,
    ens_props    = dict(alpha=0.4),
    zoomy        = 1.0,
    ):

  # Store parameters
  params_orig = Bunch(**locals())

  def init(fignum,stats,key0,plot_u,E,P,**kwargs):
    xx, yy, mu, var, chrono = stats.xx, stats.yy, stats.mu, stats.var, stats.HMM.t

    # Set parameters (kwargs takes precedence over params_orig)
    p = Bunch(**{kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

    # Lag settings:
    T_lag, K_lag, a_lag = validate_lag(p.Tplot, chrono)
    K_plot = comp_K_plot(K_lag,a_lag,plot_u)
    # Extend K_plot forther for adding blanks in resampling (PartFilt):
    has_w = hasattr(stats,'w')
    if has_w: K_plot += a_lag

    # Chose marginal dims to plot
    if p.dims==[]:
      Nx      = min(10,xx.shape[-1])
      DimsX   = equi_spaced_integers(xx.shape[-1], Nx)
    else:
      Nx      = len(p.dims)
      DimsX   = p.dims
    # Pre-process obs dimensions
    iiY   = [i for i,m in enumerate(p.obs_inds) if m in DimsX]      # Rm inds of obs if not in DimsX
    DimsY = [m for i,m in enumerate(p.obs_inds) if m in DimsX]      # Rm obs_inds    if not in DimsX
    DimsY = [DimsY.index(m) if m in DimsY else None for m in DimsX] # Get dim (within y) of each x
    Ny    = len(iiY)

    # Set up figure, axes
    fig, axs = freshfig(fignum, (5,7), loc='231-22', nrows=Nx, sharex=True)
    if Nx==1: axs = [axs]

    # Tune plots
    axs[0].set_title("Marginal time series")
    for ix, (m,ax) in enumerate(zip(DimsX,axs)):
      ax.set_ylim(*stretch(*xtrema(xx[:,m]), 1/p.zoomy))
      if p.labels==[]: ax.set_ylabel("$x_{%d}$"%m)
      else:            ax.set_ylabel(p.labels[ix])
    axs[-1].set_xlabel('Time (t)')

    plot_pause(0.05)
    plt.tight_layout()

    # Allocate
    d = Bunch() # data arrays
    h = Bunch() # plot handles
    # Why "if True" ? Just to indent the rest of the line...
    if True         : d.t  = RollingArray((K_plot,))          ; 
    if True         : d.x  = RollingArray((K_plot,Nx))        ; h.x  = []
    if True         : d.y  = RollingArray((K_plot,Ny))        ; h.y  = []
    if E is not None: d.E  = RollingArray((K_plot,len(E),Nx)) ; h.E  = []
    if P is not None: d.mu = RollingArray((K_plot,Nx))        ; h.mu = []
    if P is not None: d.s  = RollingArray((K_plot,2,Nx))      ; h.s  = []

    # Plot (invisible coz everything here is nan, for the moment).
    for ix, (m, iy, ax) in enumerate(zip(DimsX,DimsY,axs)):
      if True     : h.x  +=  ax.plot(d.t, d.x [:,ix]  , 'k')
      if iy!=None : h.y  +=  ax.plot(d.t, d.y [:,iy]  , 'g*', ms=10)
      if 'E'  in d: h.E  += [ax.plot(d.t, d.E [:,:,ix], **p.ens_props)]
      if 'mu' in d: h.mu +=  ax.plot(d.t, d.mu[:,ix]  , 'b')
      if 's'  in d: h.s  += [ax.plot(d.t, d.s [:,:,ix], 'b--',lw=1)]


    def update(key,E,P):
      k,kObs,f_a_u = key

      EE = duplicate_with_blanks_for_resampled(E, DimsX, key, has_w)

      # Roll data array
      ind = k if plot_u else kObs
      for Ens in EE: # If E is duplicated, so must the others be.
        if 'E'  in d: d.E .insert(ind, Ens)
        if 'mu' in d: d.mu.insert(ind, mu[key][DimsX])
        if 's'  in d: d.s .insert(ind, mu[key][DimsX] + [[1],[-1]]*sqrt(var[key][DimsX]))
        if True     : d.t .insert(ind, chrono.tt[k])
        if True     : d.y .insert(ind, yy[kObs,iiY] if kObs is not None else nan*ones(Ny))
        if True     : d.x .insert(ind, xx[k,DimsX])

      # Update graphs
      for ix, (m, iy, ax) in enumerate(zip(DimsX,DimsY,axs)):
        sliding_xlim(ax, d.t, T_lag, True)
        if True:       h.x [ix]   .set_data(d.t, d.x [:,ix])
        if iy!=None:   h.y [iy]   .set_data(d.t, d.y [:,iy])
        if 'mu' in d:  h.mu[ix]   .set_data(d.t, d.mu[:,ix])
        if 's'  in d: [h.s [ix][b].set_data(d.t, d.s [:,b,ix]) for b in [0,1]]
        if 'E'  in d: [h.E [ix][n].set_data(d.t, d.E [:,n,ix]) for n in range(len(E))]
        if 'E'  in d: update_alpha(key, stats, h.E[ix])


      return # end update()
    return update # end init()
  return init # end sliding_marginals()


def phase3d(
    obs_inds     = [],
    dims         = [],
    labels       = [],
    Tplot        = None,
    ens_props    = dict(alpha=0.4),
    zoom         = 1.5,
    ):

  # Store parameters
  params_orig = Bunch(**locals())

  def init(fignum,stats,key0,plot_u,E,P,**kwargs):
    xx, yy, mu, var, chrono = stats.xx, stats.yy, stats.mu, stats.var, stats.HMM.t
    M = 3 # Only applicable for 3d plots

    # Set parameters (kwargs takes precedence over params_orig)
    p = Bunch(**{kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

    # Lag settings:
    T_lag, K_lag, a_lag = validate_lag(p.Tplot, chrono)
    K_plot = comp_K_plot(K_lag,a_lag,plot_u)
    # Extend K_plot forther for adding blanks in resampling (PartFilt):
    has_w = hasattr(stats,'w')
    if has_w: K_plot += a_lag
    
    # Dimension settings
    if p.dims   == []: p.dims   = [0,1,2]
    if p.labels == []: p.labels = ["$x_%s$"%i for i in "123"]
    assert len(p.dims)==M

    # Set up figure, axes
    fig = freshfig(fignum, figsize=(5,5), loc='2321', nrows=0, ncols=0)
    ax3 = plt.subplot(111, projection='3d')
    ax3.set_facecolor('w')
    ax3.set_title("Phase space trajectories")
    # Tune plot
    for ind, (s,i) in enumerate(zip(p.labels, p.dims)):
      set_ilim(ax3, ind, *stretch(*xtrema(xx[:,i]),1/p.zoom))
    ax3.set_xlabel(p.labels[0])
    ax3.set_ylabel(p.labels[1])
    ax3.set_zlabel(p.labels[2])

    # Allocate
    d = Bunch() # data arrays
    h = Bunch() # plot handles
    s = Bunch() # scatter handles
    if E is not None                 : d.E  = RollingArray((K_plot,len(E),M)); h.E = []
    if P is not None                 : d.mu = RollingArray((K_plot,M))
    if True                          : d.x  = RollingArray((K_plot,M))
    if list(p.obs_inds)==list(p.dims): d.y  = RollingArray((K_plot,M))

    # Plot tails (invisible coz everything here is nan, for the moment).
    if 'E'  in d: h.E  += [ax3.plot(*xn    , **p.ens_props)[0] for xn in np.transpose(d.E,[1,2,0])]
    if 'mu' in d: h.mu  =  ax3.plot(*d.mu.T, 'b' , lw=2   )[0]
    if True     : h.x   =  ax3.plot(*d.x .T, 'k' , lw=3   )[0]
    if 'y'  in d: h.y   =  ax3.plot(*d.y .T, 'g*',ms=14   )[0]

    # Scatter. NB: don't init with nan's coz it's buggy (wrt. get_color() and _offsets3d) since mpl 3.1.
    if 'E'  in d: s.E   =  ax3.scatter(*E.T[p.dims],s=3 **2, c=[hn  .get_color() for hn in h.E])
    if 'mu' in d: s.mu  =  ax3.scatter(*ones(M)    ,s=8 **2, c=[h.mu.get_color(),])
    if True     : s.x   =  ax3.scatter(*ones(M)    ,s=14**2, c=[h.x .get_color(),], marker=(5, 1), zorder=99)


    def update(key,E,P):
      k,kObs,f_a_u = key
      show_y = 'y' in d and kObs is not None

      def update_tail(handle,newdata):
        handle.set_data(newdata[:,0],newdata[:,1])
        handle.set_3d_properties(newdata[:,2])

      EE = duplicate_with_blanks_for_resampled(E, p.dims, key, has_w)

      # Roll data array
      ind = k if plot_u else kObs
      for Ens in EE: # If E is duplicated, so must the others be.
        if 'E'  in d: d.E .insert(ind, Ens)
        if True     : d.x .insert(ind, xx[k,   p.dims])
        if 'y'  in d: d.y .insert(ind, yy[kObs,   :  ] if show_y else nan*ones(M))
        if 'mu' in d: d.mu.insert(ind, mu[key][p.dims])

      # Update graph
      s.x._offsets3d = juggle_axes(*d.x[[-1]].T,'z')
      update_tail(h.x, d.x)
      if 'y' in d:
          update_tail(h.y, d.y)
      if 'mu' in d:
          s.mu._offsets3d = juggle_axes(*d.mu[[-1]].T,'z')
          update_tail(h.mu, d.mu)
      else:
          s.E._offsets3d = juggle_axes(*d.E[-1].T,'z')
          for n in range(len(E)):
            update_tail(h.E[n],d.E[:,n,:])
          update_alpha(key, stats, h.E, s.E)

      return # end update()
    return update # end init()
  return init # end phase3d()



def validate_lag(Tplot, chrono):
  """Return T_lag:
   - equal to Tplot with fallback: HMM.t.Tplot.
   - no longer than HMM.t.T.
   Also return corresponding K_lag, a_lag."""

  # Defaults
  if Tplot is None:
    Tplot = chrono.Tplot
  
  # Rename
  T_lag = Tplot

  assert T_lag >= 0

  # Validate
  t2 = chrono.tt[-1]
  t1 = max(chrono.tt[0], t2-T_lag)
  T_lag = t2-t1
  
  K_lag = int(T_lag / chrono.dt) + 1 # Lag in indices
  a_lag = K_lag//chrono.dkObs + 1    # Lag in obs indices

  return T_lag, K_lag, a_lag


def plot_u(ref_stat,key0):
  """Determine whether to intermediate (between obs times) statistics are plotted.
  This is determine this by inspecting the reference statistic passed in.
  True if available (i.e. store_u) or if live.
  """
  return ref_stat.store_u or ref_stat.k_tmp==key0[0]

def comp_K_plot(K_lag,a_lag,plot_u):
  K_plot = 2*a_lag  # Sum of lags of {f,a} series.
  if plot_u:
    K_plot += K_lag # Add lag of u series.
  return K_plot

def determine_K_plot(stat,key0,K_lag,a_lag):
  """Determine K_plot: the time (in inds) window of plotting,
  i.e. the length of the RollingArray to be used."""

  if isinstance(stat,FAU_series):
    plot_u  = stat.store_u or stat.k_tmp==key0[0]
    K_plot  = 2*a_lag          # f+a series
    if plot_u: K_plot += K_lag # u series.

  else:
    plot_u = False
    K_plot = a_lag

  return K_plot, plot_u


def update_alpha(key, stats, lines, scatters=None):
  "Adjust color alpha (for particle filters)"

  k,kObs,f_a_u = key
  if kObs is None: return
  if f_a_u=='f': return
  if not hasattr(stats,'w'): return

  # Compute alpha values
  alpha = stats.w[key]
  alpha = (alpha/alpha.max()).clip(0.1,0.4)

  # Set line alpha
  for line, a in zip(lines, alpha):
    line.set_alpha(a)

  # Scatter plot does not have alpha. => Fake it.
  if scatters is not None:
    colors = scatters.get_facecolor()[:,:3]
    if len(colors)==1:
      colors = colors.repeat(len(w),axis=0)
    scatters.set_color(np.hstack([colors, alpha[:,None]]))



def duplicate_with_blanks_for_resampled(E,dims,key,has_w): 
  "Particle filter: insert breaks for resampled particles."
  if E is None:
    return [E]
  EE = []
  E  = E[:,dims]
  if has_w:
    k,kObs,f_a_u = key
    if   f_a_u=='f': pass
    elif f_a_u=='a': _Ea[0] = E[:,0] # Store (1st dim of) ens.
    elif f_a_u=='u' and kObs is not None:
      # Find resampled particles. Insert duplicate ensemble. Write nans (breaks).
      resampled = _Ea[0] != E[:,0]  # Mark as resampled if ens changed.
      EE.append( E.copy() )         # Insert current ensemble (copy to avoid overwriting).
      EE[0][resampled] = nan        # Write breaks
  # Always: append current ensemble
  EE.append(E)                   
  return EE
_Ea = [None] # persistent storage for ens



def d_ylim(data,ax=None,cC=0,cE=1,pp=(1,99),Min=-1e20,Max=+1e20):
  """Provide new ylim's intelligently,
  computed from percentiles of the data.

  - data: iterable of arrays for computing percentiles.
  - pp: percentiles

  - ax: If present, then the delta_zoom in/out is also considered.

    - cE: exansion (widenting) rate ∈ [0,1].
      Default: 1, which immediately expands to percentile.
    - cC: compression (narrowing) rate ∈ [0,1].
      Default: 0, which does not allow compression.
  
  - Min/Max: bounds

  Despite being a little involved,
  the cost of this subroutine is typically not substantial
  because there's usually not that much data to sort through.
  """

  # Find "reasonable" limits (by percentiles), looping over data
  maxv = minv = -np.inf # init
  for d in data:
    d = d[np.isfinite(d)]
    if len(d):
      minv, maxv = np.maximum([minv, maxv], \
          array([-1, 1]) * np.percentile(d,pp))
  minv *= -1

  # Pry apart equal values
  if np.isclose(minv,maxv):
    maxv += 0.5
    minv -= 0.5

  # Make the zooming transition smooth
  if ax is not None:
    current = ax.get_ylim()
    # Set rate factor as compress or expand factor. 
    c0 = cC if minv>current[0] else cE
    c1 = cC if maxv<current[1] else cE
    # Adjust
    minv = np.interp(c0, (0,1), (current[0], minv))
    maxv = np.interp(c1, (0,1), (current[1], maxv))

  # Bounds 
  maxv = min(Max,maxv)
  minv = max(Min,minv)

  # Set (if anything's changed)
  def worth_updating(a,b,curr):
    # Note: should depend on cC and cE
    d = abs(curr[1]-curr[0])
    lower = abs(a-curr[0]) > 0.002*d
    upper = abs(b-curr[1]) > 0.002*d
    return lower and upper
  #if worth_updating(minv,maxv,current):
    #ax.set_ylim(minv,maxv)

  # Some mpl versions don't handle inf limits.
  if not np.isfinite(minv): minv = None
  if not np.isfinite(maxv): maxv = None

  return minv, maxv



from .viz import setup_wrapping
def spatial1d(
    obs_inds     = None,
    periodic     = True,
    dims         = [],
    ens_props    = {'color': 0.7*RGBs['w'],'alpha':0.5},
    conf_mult    = None,
    ):

  # Store parameters
  params_orig = Bunch(**locals())

  def init(fignum,stats,key0,plot_u,E,P,**kwargs):
    xx, yy, mu = stats.xx, stats.yy, stats.mu

    # Set parameters (kwargs takes precedence over params_orig)
    p = Bunch(**{kw: kwargs.get(kw, val) for kw, val in params_orig.items()})

    if p.dims==[]:
      M = xx.shape[-1]
      p.dims = arange(M)
    else:
      M = len(p.dims)

    # Make periodic wrapper
    ii, wrap = setup_wrapping(M,p.periodic)

    # Set up figure, axes
    fig, ax = freshfig(fignum, (8,5), loc='2312-3')
    fig.suptitle("1d amplitude plot")

    # Nans
    nan1 = wrap(nan*ones(M))

    if E is None and p.conf_mult is None:
      p.conf_mult = 2

    # Init plots
    if p.conf_mult:
      lines_s  = ax.plot(ii,nan1,                     "b-" ,lw=1,label=(str(p.conf_mult) + r'$\sigma$ conf'))
      lines_s += ax.plot(ii,nan1,                     "b-" ,lw=1)
      line_mu, = ax.plot(ii,nan1,                     'b-' ,lw=2,label='DA mean')
    else:                                                      
      nanE     = nan*ones((stats.config.N,M))
      lines_E  = ax.plot(ii,wrap(nanE[0] .T), **p.ens_props,lw=1,label='Ensemble')
      lines_E += ax.plot(ii,wrap(nanE[1:].T), **p.ens_props,lw=1)
    # Truth, Obs
    line_x,    = ax.plot(ii,nan1,                     'k-' ,lw=3,label='Truth')
    if p.obs_inds is not None:                                 
      line_y,  = ax.plot(p.obs_inds, nan*p.obs_inds,  'g*' ,ms=5,label='Obs')

    # Tune plot
    ax.set_ylim( *xtrema(xx) )
    ax.set_xlim(stretch(ii[0],ii[-1],1))
    # Xticks
    xt = ax.get_xticks()
    xt = xt[abs(xt%1)<0.01].astype(int) # Keep only the integer ticks 
    xt = xt[xt >= 0]
    xt = xt[xt < len(p.dims)]
    ax.set_xticks(xt)
    ax.set_xticklabels(p.dims[xt])

    ax.set_xlabel('State index')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')

    text_t = ax.text(0.01, 0.01, format_time(None,None,None),
        transform=ax.transAxes,family='monospace',ha='left')

    # Init visibility (must come after legend):
    if p.obs_inds is not None:
      line_y.set_visible(False)


    def update(key,E,P):
      k,kObs,f_a_u = key

      if p.conf_mult:
        sigma = mu[key] + p.conf_mult * sqrt(stats.var[key]) * [[1],[-1]]
        lines_s[0].set_ydata(wrap(sigma[0,p.dims]))
        lines_s[1].set_ydata(wrap(sigma[1,p.dims]))
        line_mu   .set_ydata(wrap(mu[key][p.dims]))
      else:
        for n,line in enumerate(lines_E):
          line.set_ydata(wrap(E[n,p.dims]))

        if hasattr(stats,'w'):
          w    = stats.w[key]
          wmax = w.max()
          for n,line in enumerate(lines_E):
            line.set_alpha((w[n]/wmax).clip(0.1))

      line_x.set_ydata(wrap(xx[k,p.dims]))

      text_t.set_text(format_time(k,kObs,stats.HMM.t.tt[k]))

      if 'f' in f_a_u:
        if p.obs_inds is not None:
          line_y.set_ydata(yy[kObs])
          line_y.set_zorder(5)
          line_y.set_visible(True)

      if 'u' in f_a_u:
        if p.obs_inds is not None:
          line_y.set_visible(False)

      return # end update
    return update # end init()
  return init # end spatial1d()



def spatial2d(
    square,
    ind2sub,
    obs_inds = None,
    cm = plt.cm.jet,
    clims = [(-40,40),(-40,40),(-10,10),(-10,10)],
    ):

  def init(fignum,stats,key0,plot_u,E,P,**kwargs):

    GS = {'left':0.125-0.04,'right':0.9-0.04}
    fig, axs = freshfig(fignum, (6,6), loc='231-22-3',
        nrows=2,ncols=2,sharex=True,sharey=True, gridspec_kw=GS)

    for ax in axs.flatten():ax.set_aspect('equal',adjustable_box_or_forced())

    ((ax_11, ax_12), (ax_21, ax_22)) = axs

    ax_11.grid(color='w',linewidth=0.2)
    ax_12.grid(color='w',linewidth=0.2)
    ax_21.grid(color='k',linewidth=0.1)
    ax_22.grid(color='k',linewidth=0.1)


    # Upper colorbar -- position relative to ax_12
    bb    = ax_12.get_position()
    dy    = 0.1*bb.height
    ax_13 = fig.add_axes([bb.x1+0.03, bb.y0 + dy, 0.04, bb.height - 2*dy])
    # Lower colorbar -- position relative to ax_22
    bb    = ax_22.get_position()
    dy    = 0.1*bb.height
    ax_23 = fig.add_axes([bb.x1+0.03, bb.y0 + dy, 0.04, bb.height - 2*dy])

    # Extract data arrays
    xx, yy, mu, var, err = stats.xx, stats.yy, stats.mu, stats.var, stats.err
    k = key0[0]
    tt = stats.HMM.t.tt

    # Plot
    # - origin='lower' might get overturned by set_ylim() below.
    im_11 = ax_11.imshow(square(mu[k])       , cmap=cm) 
    im_12 = ax_12.imshow(square(xx[k])       , cmap=cm)
    im_21 = ax_21.imshow(square(sqrt(var[k])), cmap=plt.cm.bwr) # hot is better, but needs +1 colorbar
    im_22 = ax_22.imshow(square(err[k])      , cmap=plt.cm.bwr)
    ims = (im_11, im_12, im_21, im_22)
    # Obs init -- a list where item 0 is the handle of something invisible.
    lh = list(ax_12.plot(0,0)[0:1])
    
    sx = '$\\psi$'
    ax_11.set_title('mean '+sx)
    ax_12.set_title('true '+sx)
    ax_21.set_title('std. '+sx)
    ax_22.set_title('err. '+sx)

    # TODO
    # for ax in axs.flatten():
      # lims = (1, nx-2) # crop boundries (which should be 0, i.e. yield harsh q gradients).
      # step = (nx - 1)/8
      # ticks = arange(step,nx-1,step)
      # ax.set_xlim  (lims)
      # ax.set_ylim  (lims[::-1])
      # ax.set_xticks(ticks)
      # ax.set_yticks(ticks)
    
    for im,clim in zip(ims,clims):
      im.set_clim(clim)

    fig.colorbar(im_12,cax=ax_13)
    fig.colorbar(im_22,cax=ax_23)
    for ax in [ax_13, ax_23]:
      ax.yaxis.set_tick_params('major',length=2,width=0.5,direction='in',left=True,right=True)
      ax.set_axisbelow('line') # make ticks appear over colorbar patch

    # Title
    title = "Streamfunction ("+sx+")"
    fig.suptitle(title)
    # Time info
    text_t = ax_12.text(1, 1.1, format_time(None,None,None),
        transform=ax_12.transAxes,family='monospace',ha='left')

    def update(key,E,P):
      k,kObs,f_a_u = key
      t = tt[k]

      im_11.set_data(square( mu[key])        )
      im_12.set_data(square( xx[k])          )
      im_21.set_data(square( sqrt(var[key])) )
      im_22.set_data(square( err[key])       )

      # Remove previous obs
      try:
        lh[0].remove()
      except ValueError:
        pass
      # Plot current obs. 
      #  - plot() automatically adjusts to direction of y-axis in use.
      #  - ind2sub returns (iy,ix), while plot takes (ix,iy) => reverse.
      if kObs is not None and obs_inds is not None:
        lh[0] = ax_12.plot(*ind2sub(obs_inds(t))[::-1],'k.',ms=1,zorder=5)[0]

      text_t.set_text(format_time(k,kObs,t))

      return       # end update()
    return update  # end init()
  return init      # end LP_setup()




# List of liveplotters available for all HMMs.
default_liveplotters = [
    # num  show_by_default  function/class
    (  1,  1,               sliding_diagnostics),
    (  4,  1,               weight_histogram   ),
    ]




