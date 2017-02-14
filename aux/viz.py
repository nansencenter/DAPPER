from common import *

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from matplotlib import colors
from matplotlib.ticker import MaxNLocator



class LivePlot:
  """
  Live plotting functionality.
  """
  def __init__(self,stats,E=None,P=None):
    setup  = stats.setup
    config = stats.config
    m   = setup.f.m
    dt  = setup.t.dt

    ii,wrap = setup_wrapping(m)

    # Store
    self.setup  = setup
    self.stats  = stats
    self.xx     = stats.xx ; xx = stats.xx
    self.yy     = stats.yy ; yy = stats.yy

    # Abbreviate
    mu = stats.mu

    # Set up prompts
    self.is_available = config.liveplotting
    if not self.is_available: return
    self.is_on     = False
    self.is_paused = False
    print('Initializing liveplotting...')
    print('Press <Enter> to toggle live plot ON/OFF.')
    print('Press <Space> and then <Enter> to pause.')

    #ens_props = {} yields rainbow
    ens_props = {'color': 0.7*RGBs['w'],'alpha':0.3}


    #####################
    # Dashboard
    #####################
    self.fga = plt.figure("Dashboard",figsize=(8,8))
    self.fga.clf()
    set_figpos('2311')

    ax = plt.subplot(311)
    if m<401:
      if E is not None and len(E)<4001:
        self.lE = plt.plot(ii,wrap(E.T),lw=1,**ens_props)
      else:
        self.ks  = 2.0
        self.CI  = ax.fill_between(ii, \
            wrap(mu[0] - self.ks*sqrt(stats.var[0])), \
            wrap(mu[0] + self.ks*sqrt(stats.var[0])), \
            alpha=0.4,label=(str(self.ks) + ' sigma'))
      
      self.lx,    = plt.plot(ii,wrap(xx[0]),'k',lw=3,ls='-',label='Truth')
      self.lmu,   = plt.plot(ii,wrap(mu[0]),'b',lw=2,ls='-',label='DA estim.')
      self.fcast, = plt.plot(ii,wrap(mu[0]),'r',lw=1,ls='-',label='forecast')

      if hasattr(setup.h,'plot'):
        # tmp
        self.yplot = setup.h.plot
        self.obs = setup.h.plot(yy[0])
        self.obs.set_label('Obs')
        #self.obs.set_visible(False)
      
      ax.legend()
      #ax.xaxis.tick_top()
      #ax.xaxis.set_label_position('top') 
      ax.set_xlabel('State index')
      ax.set_xlim((ii[0],ii[-1]))
      ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
      tcks = ax.get_xticks()
      self.ax = ax

    # TODO: Spectral plot
    #axS = plt.subplot(412)


    # Correlation plot. Inspired by seaborn.heatmap.
    axC = plt.subplot(312)
    if m<400:
      divdr = make_axes_locatable(axC)
      # Append axes to the right for colorbar
      cax = divdr.append_axes("bottom", size="10%", pad=0.05)
      if E is not None:
        C = np.cov(E,rowvar=False)
      else:
        assert P is not None
        C = P.copy()
      std = sqrt(diag(C))
      C  /= std[:,None]
      C  /= std[None,:]
      mask = np.zeros_like(C, dtype=np.bool)
      mask[np.tril_indices_from(mask)] = True
      C2 = np.ma.masked_where(mask, C)[::-1]
      try:
        cmap = sns.diverging_palette(220,10,as_cmap=True)
      except NameError:
        cmap = plt.get_cmap('coolwarm')
      # Log-transform cmap, but not internally in matplotlib,
      # to avoid transforming the colorbar too.
      trfm = colors.SymLogNorm(linthresh=0.2,linscale=0.2,vmin=-1, vmax=1)
      cmap = cmap(trfm(linspace(-1,1,cmap.N)))
      cmap = colors.ListedColormap(cmap)
      #VM  = max(abs(np.percentile(C2,[1,99])))
      VM   = 1.0
      mesh = axC.pcolormesh(C2,cmap=cmap,vmin=-VM,vmax=VM)
      axC.figure.colorbar(mesh,cax=cax,orientation='horizontal')
      plt.box(False)
      axC.yaxis.tick_right()
      tcks = tcks[np.logical_and(tcks >= ii[0], tcks <= ii[-1])]
      tcks = tcks.astype(int)
      axC.set_yticks(m-tcks-0.5)
      axC.set_yticklabels([str(x) for x in tcks])
      axC.set_xticklabels([])
      cax.set_xlabel('Correlation')
      axC.set_axis_bgcolor('w')
      

      acf = circulant_ACF(C)
      ac6 = circulant_ACF(C,do_abs=True)
      ax5 = inset_axes(axC,width="30%",height="60%",loc=3)
      l5, = ax5.plot(arange(m), acf,     label='auto-corr')
      l6, = ax5.plot(arange(m), ac6, label='abs(")')
      ax5.set_xticklabels([])
      ax5.set_yticks([0,1] + list(ax5.get_yticks()[[0,-1]]))
      ax5.set_ylim(top=1)
      ax5.legend(frameon=False,loc=1)
      #ax5.text(m/2-.5,0.9,'Auto corr.',va='center',ha='center')

      self.axC = axC
      self.ax5 = ax5
      self.mask = mask
      self.mesh = mesh
      self.l5 = l5
      self.l6 = l6
    


    ax2  = plt.subplot(313)
    self.ax2 = ax2
    msft = abs(stats.umisf[0])
    sprd =     stats.svals[0]
    ax2.set_xlabel('Sing. value index')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-5)
    #ax2.set_ylim([1e-3,1e1])
    self.do_spectral_error = not all(msft == 0)
    if self.do_spectral_error:
      self.lmf, = ax2.plot(arange(len(msft)),msft,'k',lw=2,label='Error')
      self.lew, = ax2.plot(arange(len(sprd)),sprd,'b',lw=2,label='Spread',alpha=0.9)
      ax2.get_xaxis().set_major_locator(MaxNLocator(integer=True))
      ax2.legend()
    else:
      not_available_text(ax2)


    self.fga.subplots_adjust(top=0.95,bottom=0.08,hspace=0.4)
    adjust_position(axC,y0=0.03)
    self.fga.suptitle('t=Init')


    #####################
    # Weight histogram
    #####################
    if E is not None and len(E)<1001 and stats.has_w:
      fgh = plt.figure("Weight histogram",figsize=(8,4))
      fgh.clf()
      set_figpos('2321')
      axh = fgh.add_subplot(111)
      fgh.subplots_adjust(bottom=.15)
      hst = axh.hist(stats.w[0])[2]
      N = len(E)
      axh.set_xscale('log')
      xticks = 1/N * 10**arange(-4,log10(N)+1)
      xtlbls = array(['$10^{'+ str(int(log10(w*N))) + '}$' for w in xticks])
      xtlbls[xticks==1/N] = '1'
      axh.set_xticks(xticks)
      axh.set_xticklabels(xtlbls)
      axh.set_xlabel('weigth [× N]')
      axh.set_ylabel('count')
      self.fgh = fgh
      self.axh = axh
      self.hst = hst


    #####################
    # 3D phase space
    #####################
    self.fg3 = plt.figure("3D trajectories",figsize=(8,6))
    self.fg3.clf()
    set_figpos('2321')

    ax3      = self.fg3.add_subplot(111,projection='3d')
    self.ax3 = ax3

    tail_k = max([2,int(1/dt)])
    
    if E is not None and len(E)<1001:
      # Ensemble
      self.sE = ax3.scatter(*E.T[:3],s=10,**ens_props)
      # Tail
      N = E.shape[0]
      self.tail_E  = zeros((tail_k,N,3))
      for k in range(tail_k):
        self.tail_E[k] = E[:,:3]
      self.ltE = []
      for n in range(N):
        lEn, = ax3.plot(*self.tail_E[:,n].squeeze().T,**ens_props)
        self.ltE.append(lEn)
    else:
      # Mean
      self.smu = ax3.scatter(*mu[0,:3],s=100,c='b')
      # Tail
      self.tail_mu = ones((tail_k,3)) * mu[0,:3] # init
      self.ltmu,   = ax3.plot(*self.tail_mu.T,'b',lw=2)

    # Truth
    self.sx      = ax3.scatter(*xx[0,:3],s=300,c='y',marker=(5, 1))
    # Tail
    self.tail_xx = ones((tail_k,3)) * xx[0,:3] # init
    self.ltx,    = ax3.plot(*self.tail_xx.T,'y',lw=4)

    #ax3.axis('off')
    for i in range(3):
      set_ilim(ax3,i,xx,1.7)
    ax3.set_axis_bgcolor('w')


    #####################
    # Diagnostics
    #####################
    self.fgd = plt.figure("Scalar diagnostics",figsize=(8,6))
    self.fgd.clf()
    set_figpos('2312')

    chrono = setup.t
    self.K = estimate_good_plot_length(xx,chrono,mult=80)
    pkk = arange(self.K)
    ptt = chrono.tt[pkk]

    self.ax_e = plt.subplot(211)
    self.le,  = self.ax_e.plot(ptt,stats.rmse[pkk],'k',lw=2,label='Error')
    self.lv,  = self.ax_e.plot(ptt,stats.rmv [pkk],'b',lw=2,label='Spread',alpha=0.6)
    self.ax_e.set_ylabel('RMS')
    self.ax_e.legend()
    self.ax_e.set_xticklabels([])

    self.ax_i = plt.subplot(212)
    self.ls,  = self.ax_i.plot(ptt,stats.skew[pkk],'g',lw=2,label='Skew')
    self.lk,  = self.ax_i.plot(ptt,stats.kurt[pkk],'r',lw=2,label='Kurt')
    self.ax_i.legend()
    self.ax_i.set_xlabel('time (t)')

    #####################
    # User-defined state
    #####################
    if hasattr(setup.f,'plot'):
      self.fgu = plt.figure(25,figsize=(8,8))
      self.fgu.clf()
      set_figpos('2322')

      self.axu1 = plt.subplot(211)
      self.setter_truth = setup.f.plot(xx[0])
      self.axu1.set_title('Truth')

      self.axu2 = plt.subplot(212)
      plt.subplots_adjust(hspace=0.3)
      self.setter_mean = setup.f.plot(mu[0])
      self.axu2.set_title('Mean')



    self.prev_k = 0
    plt.pause(0.01)



  def skip_plotting(self):
    """
    Poll user for keypresses.
    Decide on toggling pause/step/plot:
    """
    if not self.is_available:
      return True

    if self.is_paused:
      # If paused
      ch = getch()
      # Wait for <space> or <enter>
      while ch not in [' ','\r']:
        ch = getch()
      # If <enter>, turn off pause
      if '\r' in ch:
        self.is_paused = False

    key = poll_input() # =None if <space> was pressed above
    if key is not None:
      if key == '\n':
        # If <enter> 
        self.is_on = not self.is_on # toggle plotting on/off
      elif key == ' \n':
        # If <space>+<enter> 
        self.is_on = True # turn on plotting
        self.is_paused = not self.is_paused # toggle pause
        print("Press <Space> to step. Press <Enter> to resume.")
    return not self.is_on

    
    
  def insert_forecast(self,mu):
    """Plot amplitudes of forecast state"""
    if not self.is_available or not self.is_on: return
    if plt.fignum_exists(self.fga.number):
      ii,wrap = setup_wrapping(len(mu))
      plt.figure(self.fga.number)
      self.fcast.set_ydata(wrap(mu))
      self.fcast.set_visible(True)



  def update(self,k,kObs,E=None,P=None):
    """Plot forecast state"""
    if self.skip_plotting(): return

    #open_figns = plt.get_fignums()

    stats = self.stats
    mu    = stats.mu
    m     = self.xx.shape[1]

    ii,wrap = setup_wrapping(m)
    
    #####################
    # Dashboard
    #####################
    if plt.fignum_exists(self.fga.number):

      plt.figure(self.fga.number)
      t = self.setup.t.tt[k]
      self.fga.suptitle('t[k]={:<5.2f}, k={:<d}'.format(t,k))

      if hasattr(self,'ax'):
        self.lmu.set_ydata(wrap(mu[k]))
        self.lx .set_ydata(wrap(self.xx[k]))

        if kObs is None:
          self.fcast.set_visible(False)

        if hasattr(self,'lE'):
          w    = stats.w[k]
          wmax = w.max()
          for i,l in enumerate(self.lE):
            l.set_ydata(wrap(E[i]))
            l.set_alpha((w[i]/wmax).clip(0.1))
        else:
          self.CI.remove()
          self.CI  = self.ax.fill_between(ii, \
              wrap(mu[k] - self.ks*sqrt(stats.var[k])), \
              wrap(mu[k] + self.ks*sqrt(stats.var[k])), alpha=0.4)

        plt.sca(self.ax)
        if hasattr(self,'obs'):
          try:
            self.obs.remove()
          except Exception:
            pass
          if kObs is not None:
            self.obs = self.yplot(self.yy[kObs])

        update_ylim([mu[k], self.xx[k]], self.ax)



      if hasattr(self,'axC'):
        if E is not None:
          C = np.cov(E,rowvar=False)
        else:
          assert P is not None
          C = P.copy()
        std = sqrt(diag(C))
        C  /= std[:,None]
        C  /= std[None,:]
        C2 = np.ma.masked_where(self.mask, C)[::-1]
        self.mesh.set_array(C2.ravel())

        acf = circulant_ACF(C)
        ac6 = circulant_ACF(C,do_abs=True)
        self.l5.set_ydata(acf)
        self.l6.set_ydata(ac6)
        update_ylim([acf, ac6], self.ax5)


      if self.do_spectral_error:
        msft = abs(stats.umisf[k])
        sprd =     stats.svals[k]
        self.lew.set_ydata(sprd)
        self.lmf.set_ydata(msft)
        update_ylim(msft, self.ax2)

      plt.pause(0.01)


    #####################
    # Weight histogram
    #####################
    if kObs and hasattr(self, 'fgh') and plt.fignum_exists(self.fgh.number):
      plt.figure(self.fgh.number)
      axh      = self.axh
      _        = [b.remove() for b in self.hst]
      w        = stats.w[k]
      N        = len(w)
      wmax     = w.max()
      bins     = exp(linspace(log(1e-5/N), log(1), int(N/20)))
      counted  = w>bins[0]
      nC       = sum(counted)
      nn,_,pp  = axh.hist(w[counted],bins=bins,color='b')
      self.hst = pp
      #thresh   = '#(w<$10^{'+ str(int(log10(bins[0]*N))) + '}/N$ )'
      axh.set_title('N: {:d}.   N_eff: {:.4g}.   Not shown: {:d}. '.\
          format(N, 1/(w@w), N-nC))
      update_ylim([nn], axh, do_narrow=True)
      plt.pause(0.01)

    #####################
    # 3D phase space
    #####################

    def update_tail(handle,newdata):
      handle.set_data(newdata[:,0],newdata[:,1])
      handle.set_3d_properties(newdata[:,2])
    def np_popleft(arr,item):
      arr    = np.roll(arr,1,axis=0)
      arr[0] = item
      return arr

    if hasattr(self,'fg3') and plt.fignum_exists(self.fg3.number):
      plt.figure(self.fg3.number)

      # Reset
      if self.prev_k != (k-1):
        self.tail_xx  [:] = self.xx[k,:3]
        if hasattr(self, 'sE'):
          self.tail_E [:] = E[:,:3]
        else:
          self.tail_mu[:] = mu[k,:3]

      # Truth
      self.sx._offsets3d = juggle_axes(*tp(self.xx[k,:3]),'z')
      self.tail_xx       = np_popleft(self.tail_xx,self.xx[k,:3])
      update_tail(self.ltx, self.tail_xx)

      if hasattr(self, 'sE'):
        # Ensemble
        self.sE._offsets3d = juggle_axes(*E.T[:3],'z')

        clrs  = self.sE.get_facecolor()[:,:3]
        w     = stats.w[k]
        alpha = (w/w.max()).clip(0.1,0.4)
        if len(clrs) == 1: clrs = clrs.repeat(len(w),axis=0)
        self.sE.set_color(np.hstack([clrs, alpha[:,None]]))
        
        self.tail_E = np_popleft(self.tail_E,E[:,:3])
        for n in range(E.shape[0]):
          update_tail(self.ltE[n],self.tail_E[:,n,:])
          self.ltE[n].set_alpha(alpha[n])
      else:
        # Mean
        self.smu._offsets3d = juggle_axes(*tp(mu[k,:3]),'z')
        self.tail_mu        = np_popleft(self.tail_mu,mu[k,:3])
        update_tail(self.ltmu, self.tail_mu)

      plt.pause(0.01)

      # For animation:
      #self.fg3.savefig('figs/l63_' + str(k) + '.png',format='png',dpi=70)
    
    #####################
    # Diagnostics
    #####################
    if plt.fignum_exists(self.fgd.number):
      plt.figure(self.fgd.number)
      pkk = arange(self.K)
      if k > self.K:
        pkk += (k-self.K)
      pkk = pkk.astype(int)
      ptt = self.setup.t.tt[pkk]

      self.le.set_data(ptt,stats.rmse[pkk])
      self.lv.set_data(ptt,stats.rmv[pkk])
      self.ax_e.set_xlim(ptt[0],ptt[0] + 1.1 * (ptt[-1]-ptt[0]))
      update_ylim([stats.rmse[pkk],stats.rmv[pkk]], self.ax_e,Min=0)
      
      self.ls.set_data(ptt,stats.skew[pkk])
      self.lk.set_data(ptt,stats.kurt[pkk])
      self.ax_i.set_xlim(ptt[0],ptt[0] + 1.1 * (ptt[-1]-ptt[0]))
      update_ylim([stats.skew[pkk],stats.kurt[pkk]], self.ax_i)

      plt.pause(0.01)


    #####################
    # User-defined state
    #####################
    if hasattr(self,'fgu'):
      plt.figure(self.fgu.number)
      self.setter_truth(self.xx[k])
      self.setter_mean(mu[k])
      plt.pause(0.01)

    self.prev_k = k

def setup_wrapping(m):
  """
  Make periodic indices and a corresponding function
  (that works for ensemble input).
  """
  ii  = np.hstack([-0.5, range(m), m-0.5])
  def wrap(E):
    midpoint = (E[[0],...] + E[[-1],...])/2
    return np.concatenate((midpoint,E,midpoint),axis=0)
  return ii, wrap
  
def adjust_position(ax,adjust_extent=False,**kwargs):
  """
  Adjust values (add) to get_position().
  kwarg must be one of 'x0','y0','width','height'.
  """
  # Load get_position into d
  pos = ax.get_position()
  d   = OrderedDict()
  for key in ['x0','y0','width','height']:
    d[key] = getattr(pos,key)
  # Make adjustments
  for key,item in kwargs.items():
    d[key] += item
    if adjust_extent:
      if key=='x0': d['width']  -= item
      if key=='y0': d['height'] -= item
  # Set
  ax.set_position(d.values())

def update_ylim(data,ax,Min=None,Max=None,do_narrow=False):
  """
  Update ylims intelligently.
  Better to use mpl.relim() and self.ax.autoscale_view() ?
  """
  current = ax.get_ylim()
  maxv = minv = -np.inf
  # Find "reasonable" limits, looping over data
  for d in data:
    minv, maxv = np.maximum([minv, maxv], \
        1.1 * array([-1, 1]) * np.percentile(d,[1,99]))
  minv *= -1
  # Allow making limits more narrow?
  if not do_narrow:
    minv = min([minv,current[0]])
    maxv = max([maxv,current[1]])
  # Overrides
  if Max is not None: maxv = Max
  if Min is not None: minv = Min
  # Set (if anything's changed)
  if (minv, maxv) != current and minv != maxv:
    ax.set_ylim(minv,maxv)


def set_ilim(ax,i,data,zoom=1.0):
  """Set bounds (taken from data) on axis i.""" 
  Min  = min(data[:,i])
  Max  = max(data[:,i])
  lims = round2sigfig([Min, Max])
  lims = inflate_ens(lims,1/zoom)
  if i is 0: ax.set_xlim(lims)
  if i is 1: ax.set_ylim(lims)
  if i is 2: ax.set_zlim(lims)

def set_ilabel(ax,i):
  if i is 0: ax.set_xlabel('x')
  if i is 1: ax.set_ylabel('y')
  if i is 2: ax.set_zlabel('z')



def estimate_good_plot_length(xx,chrono,mult):
  """Estimate good length for plotting stuff
  from the time scale of the system.
  Provide sensible fall-backs."""
  t = chrono
  if xx.ndim == 2:
    # If mult-dim, then average over dims (by ravel)....
    # But for inhomogeneous variables, it is important
    # to subtract the mean first!
    xx = xx - mean(xx,axis=0)
    xx = xx.ravel(order='F')
  try:
    K = mult * estimate_corr_length(xx)
  except ValueError:
    K = 0
  K = int(min([max([K,t.dkObs]),t.K]))
  T = round2sigfig(t.tt[K],2) # Could return T; T>tt[-1]
  K = find_1st_ind(t.tt >= T)
  if K: return K
  else: return t.K

def get_plot_inds(chrono,xx,mult,K=None,T=None):
  """
  Def subset of kk for plotting, from one of
   - K
   - T
   - mult * auto-correlation length of xx
  """
  t = chrono
  if K is None:
    if T: K = find_1st_ind(t.tt >= min((T,t.T)))
    else: K = estimate_good_plot_length(xx,t,mult)
  plot_kk    = t.kk[:K+1]
  plot_kkObs = t.kkObs[t.kkObs<=K]
  return plot_kk, plot_kkObs


def plot_3D_trajectory(stats,dims=0,**kwargs):
  """
  Plot 3D phase-space trajectory.
  kwargs forwarded to get_plot_inds().
  """
  if isinstance(dims,int):
    dims = dims + arange(3)
  assert len(dims)==3

  xx     = stats.xx
  chrono = stats.setup.t

  kk = get_plot_inds(chrono,xx,mult=100,**kwargs)[0]
  T  = chrono.tt[kk[-1]]

  xx = xx      [kk][:,dims]
  mu = stats.mu[kk][:,dims]

  plt.figure(14).clf()
  set_figpos('3321 mac')
  ax3 = plt.subplot(111, projection='3d')

  ax3.plot   (xx[:,0] ,xx[:,1] ,xx[:,2] ,c='k',label='Truth')
  ax3.plot   (mu[:,0] ,mu[:,1] ,mu[:,2] ,c='b',label='DA estim.')
  ax3.scatter(xx[0 ,0],xx[0 ,1],xx[0 ,2],c='g',s=40)
  ax3.scatter(xx[-1,0],xx[-1,1],xx[-1,2],c='r',s=40)
  ax3.set_title('Phase space trajectory up to t={:<5.2f}'.format(T))
  ax3.set_xlabel('dim ' + str(dims[0]))
  ax3.set_ylabel('dim ' + str(dims[1]))
  ax3.set_zlabel('dim ' + str(dims[2]))
  ax3.legend(frameon=False)
  ax3.set_axis_bgcolor('w')
  # Don't do the following, coz it also needs the white grid,
  # which I can't get working for 3d.
  #for i in 'xyz': eval('ax3.w_' + i + 'axis.set_pane_color(sns_bg)')


def plot_time_series(stats,dim=0,hov=False,**kwargs):
  """
  Plot time series of various statistics.
  kwargs forwarded to get_plot_inds().
  """
  s      = stats
  xx     = stats.xx
  chrono = stats.setup.t

  fg = plt.figure(12,figsize=(8,8)).clf()
  set_figpos('1313 mac')

  pkk,pkkObs = get_plot_inds(chrono,xx[:,dim],mult=80,**kwargs)
  tt = chrono.tt

  ax_d = plt.subplot(3,1,1)
  ax_d.plot(tt[pkk],xx  [pkk,dim],'k',lw=3,label='Truth')
  ax_d.plot(tt[pkk],s.mu[pkk,dim],lw=2,label='DA estim.')
  #ax_d.set_ylabel('$x_{' + str(dim) + '}$',usetex=True,size=20)
  #ax_d.set_ylabel('$x_{' + str(dim) + '}$',size=20)
  ax_d.set_ylabel('dim ' + str(dim))
  ax_d.legend()
  ax_d.set_xticklabels([])

  ax_K = plt.subplot(3,1,2)
  ax_K.plot(tt[pkkObs], s.trHK[:len(pkkObs)],'k',lw=2,label='tr(HK)')
  ax_K.plot(tt[pkkObs], s.skew[:len(pkkObs)],'g',lw=2,label='Skew')
  ax_K.plot(tt[pkkObs], s.kurt[:len(pkkObs)],'r',lw=2,label='Kurt')
  ax_K.legend()
  ax_K.set_xticklabels([])

  ax_e = plt.subplot(3,1,3)
  ax_e.plot(        tt[pkk], s.rmse[pkk],'k',lw=2 ,label='Error')
  ax_e.fill_between(tt[pkk], s.rmv [pkk],alpha=0.7,label='Spread') 
  ylim = np.percentile(s.rmse[pkk],99)
  ylim = 1.1*max([ylim, max(s.rmv[pkk])])
  ax_e.set_ylim(0,ylim)
  ax_e.set_ylabel('RMS')
  ax_e.legend()
  ax_e.set_xlabel('time (t)')


  if hov:
    #cm = mpl.colors.ListedColormap(sns.color_palette("BrBG", 256)) # RdBu_r
    #cm = plt.get_cmap('BrBG')
    fgH = plt.figure(16,figsize=(6,5)).clf()
    set_figpos('3311 mac')
    axH = plt.subplot(111)
    m = xx.shape[1]
    plt.contourf(arange(m),tt[pkk],xx[pkk],25)
    plt.colorbar()
    axH.set_position([0.125, 0.20, 0.62, 0.70])
    axH.set_title("Hovmoller diagram (of 'Truth')")
    axH.set_xlabel('Dimension index (i)')
    axH.set_ylabel('Time (t)')
    add_endpoint_xtick(axH)


def add_endpoint_xtick(ax):
  """Useful when xlim(right) is e.g. 39 (instead of 40)."""
  xF = ax.get_xlim()[1]
  ticks = ax.get_xticks()
  if ticks[-1] > xF:
    ticks = ticks[:-1]
  ticks = np.append(ticks, xF)
  ax.set_xticks(ticks)


def integer_hist(E,N,centrd=False,weights=None,**kwargs):
  """Histogram for integers."""
  ax = plt.gca()
  rnge = (-0.5,N+0.5) if centrd else (0,N+1)
  ax.hist(E,bins=N+1,range=rnge,normed=1,weights=weights,**kwargs)
  ax.set_xlim(rnge)


def not_available_text(ax,fs=20):
  ax.text(0.5,0.5,'[Not available]',
      fontsize=fs,
      transform=ax.transAxes,
      va='center',ha='center')

def plot_err_components(stats):
  """
  Plot components of the error.
  Note: it was chosen to plot(ii, mean_in_time(abs(err_i))),
        and thus the corresponding spread measure is MAD.
        If one chose instead: plot(ii, std_in_time(err_i)),
        then the corrensopnding spread measure would have been std.
        This choice was made in part because (wrt subplot 2)
        the singular values (svals) correspond to rotated MADs,
        and because rms(umisf) seems to convoluted for interpretation.
  """
  s      = stats
  chrono = stats.setup.t
  m      = s.xx.shape[1]

  fgE = plt.figure(15,figsize=(8,8)).clf()
  set_figpos('1312 mac')
  ax_r = plt.subplot(311)
  ax_r.set_xlabel('Dimension index (i)')
  ax_r.set_ylabel('Time-average magnitude')
  ax_r.plot(arange(m),mean(abs(s.err.a),0),'k',lw=2, label='Error')
  sprd = mean(s.mad.a,0)
  if m<10**3:
    ax_r.fill_between(arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.7,label='Spread')
  else:
    ax_r.plot(arange(len(sprd)),sprd,alpha=0.7,label='Spread')
  ax_r.set_title('Element-wise error comparison (_a)')
  #ax_r.set_yscale('log')
  ax_r.set_ylim(bottom=mean(sprd)/10)
  ax_r.set_xlim(right=m-1); add_endpoint_xtick(ax_r)
  ax_r.get_xaxis().set_major_locator(MaxNLocator(integer=True))
  ax_r.legend()
  #ax_r.set_position([0.125,0.6, 0.78, 0.34])
  plt.subplots_adjust(hspace=0.55)

  ax_s = plt.subplot(312)
  has_been_computed = not all(s.umisf[-1] == 0)
  ax_s.set_xlabel('Principal component index')
  ax_s.set_ylabel('Time-average magnitude')
  ax_s.set_title('Spectral error comparison (_a)')
  if has_been_computed:
    msft = mean(abs(s.umisf.a),0)
    sprd = mean(s.svals.a,0)
    ax_s.plot(        arange(len(msft)),              msft,'k',lw=2, label='Error')
    ax_s.fill_between(arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.7,label='Spread')
    ax_s.set_yscale('log')
    ax_s.set_ylim(bottom=1e-4*sum(sprd))
    ax_s.set_xlim(right=m-1); add_endpoint_xtick(ax_s)
    ax_s.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    ax_s.legend()
  else:
    not_available_text(ax_s)


  ax_R = plt.subplot(313)
  ax_R.set_ylabel('Num. of occurence')
  ax_R.set_xlabel('RMSE')
  ax_R.set_title('Histogram of RMSE values (_u)')
  ax_R.hist(s.rmse.u[chrono.kk_BI],bins=30,normed=0)


def plot_rank_histogram(stats):
  chrono = stats.setup.t

  has_been_computed = hasattr(stats,'rh') and not all(stats.rh[-1]==0)

  def are_uniform(w):
    """Test inital & final weights, not intermediate (for speed)."""
    (w[0]==1/N).all() and (w[-1]==1/N).all()

  fg = plt.figure(13,figsize=(8,4)).clf()
  set_figpos('3331 mac')
  #
  ax_H = plt.subplot(111)
  ax_H.set_title('(Average of marginal) rank histogram (_u)')
  ax_H.set_ylabel('Freq. of occurence\n (of truth in interval n)')
  ax_H.set_xlabel('ensemble member index (n)')
  ax_H.set_position([0.125,0.15, 0.78, 0.75])
  if has_been_computed:
    w     = stats.w.u
    ranks = stats.rh.u[chrono.kk_BI]
    m     = ranks.shape[1]
    N     = w.shape[1]
    if are_uniform(w):
      # Ensemble rank histogram
      integer_hist(ranks.ravel(),N)
    else:
      # Experimental: weighted rank histogram.
      # Weight ranks by inverse of particle weight. Why? Coz, with correct
      # importance weights, the "expected value" histogram is then flat.
      # Potential improvement: interpolate weights between particles.
      w  = w[chrono.kk_BI]
      K  = len(w)
      w  = np.hstack([w, ones((K,1))/N]) # define weights for rank N+1
      w  = array([ w[arange(K),ranks[arange(K),i]] for i in range(m)])
      w  = w.T.ravel()
      w  = np.maximum(w, 1/N/100) # Artificial cap. Reduces variance, but introduces bias.
      w  = 1/w
      integer_hist(ranks.ravel(),N,weights=w)
  else:
    not_available_text(ax_H)
  
def show_figs(fignums=None):
  """Move all fig windows to top"""
  if fignums == None:
    fignums = plt.get_fignums()
  try:
    fignums = list(fignums)
  except:
    fignums = [fignums]
  for f in fignums:
    plt.figure(f)
    fmw = plt.get_current_fig_manager().window
    fmw.attributes('-topmost',1) # Bring to front, but
    fmw.attributes('-topmost',0) # don't keep in front
  

def set_figpos(loc):
  """
  Place figure on screen, where 'loc' can be either
    NW, E, ...
  or
    4 digits (as str or int) to define grid m,n,i,j.
  """

  #Only works with both:
   #- Patrick's monitor setup (Dell with Mac central-below)
   #- TkAgg backend. (Previously: Qt4Agg)
  if not user_is_patrick or mpl.get_backend() != 'TkAgg':
    return
  fmw = plt.get_current_fig_manager().window

  loc = str(loc)

  # Qt4Agg only:
  #  # Current values 
  #  w_now = fmw.width()
  #  h_now = fmw.height()
  #  x_now = fmw.x()
  #  y_now = fmw.y()
  #  # Constants 
  #  Dell_w = 2560
  #  Dell_h = 1440
  #  Mac_w  = 2560
  #  Mac_h  = 1600
  #  # Why is Mac monitor scaled by 1/2 ?
  #  Mac_w  /= 2
  #  Mac_h  /= 2
  # Append the string 'mac' to place on mac monitor.
  #  if 'mac' in loc:
  #    x0 = Dell_w/4
  #    y0 = Dell_h+44
  #    w0 = Mac_w
  #    h0 = Mac_h-44
  #  else:
  #    x0 = 0
  #    y0 = 0
  #    w0 = Dell_w
  #    h0 = Dell_h

  # TkAgg
  x0 = 0
  y0 = 0
  w0 = 1280
  h0 = 752
  
  # Def place function with offsets
  def place(x,y,w,h):
    #fmw.setGeometry(x0+x,y0+y,w,h) # For Qt4Agg
    geo = str(int(w)) + 'x' + str(int(h)) + \
        '+' + str(int(x)) + '+' + str(int(y))
    fmw.geometry(newGeometry=geo) # For TkAgg

  if not loc[:4].isnumeric():
    if   loc.startswith('NW'): loc = '2211'
    elif loc.startswith('SW'): loc = '2221'
    elif loc.startswith('NE'): loc = '2212'
    elif loc.startswith('SE'): loc = '2222'
    elif loc.startswith('W' ): loc = '1211'
    elif loc.startswith('E' ): loc = '1212'
    elif loc.startswith('S' ): loc = '2121'
    elif loc.startswith('N' ): loc = '2111'

  # Place
  m,n,i,j = [int(x) for x in loc[:4]]
  assert m>=i>0 and n>=j>0
  h0   -= (m-1)*25
  yoff  = 25*(i-1)
  if i>1:
    yoff += 25
  place((j-1)*w0/n, yoff + (i-1)*h0/m, w0/n, h0/m)


