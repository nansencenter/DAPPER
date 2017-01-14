from common import *

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes


# RGB constants
cblue,cgreen,cred,cmagenta,cyellow,ccyan = array(sns.color_palette())
sns_bg = array([0.9176, 0.9176, 0.9490])
#cred   = array([1,0,0])
#cgreen = array([0,1,0])
#cblue  = array([0,0,1])
cwhite = array([1,1,1])
cblack = array([0,0,0])


class LivePlot:
  """
  Live plotting functionality.
  """
  def __init__(self,setup,config,E,stats,xx,yy):
    N,m = E.shape
    dt = setup.t.dt
    ii  = range(m)

    self.stats  = stats
    self.xx     = xx
    self.yy     = yy
    self.setup = setup

    self.is_available = config.liveplotting
    if not self.is_available:
      return
    self.is_on     = False
    self.is_paused = False
    print('Press <Enter> to toggle live plot ON/OFF.')
    print('Press <Space> and then <Enter> to pause.')

    #ens_props = {} # yields rainbow
    ens_props = {'color': 0.6*cwhite} # 0.7*cblue


    #####################
    # Amplitudes
    #####################
    self.fga = plt.figure(21,figsize=(8,8))
    self.fga.clf()
    set_figpos('2311')

    self.ax  = plt.subplot(211)
    self.lmu,= plt.plot(ii,stats.mu[0],'b',lw=2,ls='-',label='Ens.mean')
    self.lx ,= plt.plot(ii,      xx[0],'k',lw=3,ls='-',label='Truth')

    #lE  = plt.plot(ii,E.T,lw=1,*ens_props)
    self.ks  = 3.0
    self.CI  = self.ax.fill_between(ii, \
        stats.mu[0] - self.ks*sqrt(stats.var[0]), \
        stats.mu[0] + self.ks*sqrt(stats.var[0]), \
        alpha=0.4,label=(str(self.ks) + ' sigma'))

    if hasattr(setup.h,'plot'):
      self.yplot = setup.h.plot
      self.obs = setup.h.plot(yy[0])
      self.obs.set_label('Obs')
      self.obs.set_visible('off')

    self.ax.legend()
    self.ax.set_xlabel('State index')

    ax2 = plt.subplot(212)
    msft = abs(stats.umisf[0])
    sprd =     stats.svals[0]
    self.lmf, = ax2.plot(arange(len(msft)),msft,'k',lw=2,label='Error')
    self.lew, = ax2.plot(arange(len(sprd)),sprd,'b',lw=2,label='Spread',alpha=0.9)
    plt.subplots_adjust(hspace=0.3)
    ax2.set_xlabel('Sing. value index')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-5)
    ax2.legend()
    #ax2.set_ylim([1e-3,1e1])


    #####################
    # 3D phase space
    #####################
    self.fg3 = plt.figure(23,figsize=(8,6))
    self.fg3.clf()
    set_figpos('2321')

    ax3      = self.fg3.add_subplot(111,projection='3d')
    self.ax3 = ax3
    self.sx  = ax3.scatter(*xx[0,:3]  ,s=300,c='y',marker=(5, 1))
    self.sE  = ax3.scatter(*E[: ,:3].T,s=10,**ens_props)

    tail_k       = max([2,int(1/dt)])
    self.tail_xx = ones((tail_k,3)) * xx[0,:3] # init
    self.ltx,    = ax3.plot(*self.tail_xx.T,'b',lw=4)

    self.tail_E  = zeros((tail_k,N,3))
    for k in range(tail_k):
      self.tail_E[k] = E[:,:3]
    self.ltE = []
    for n in range(N):
      lEn, = ax3.plot(*self.tail_E[:,n].squeeze().T,**ens_props)
      self.ltE.append(lEn)

    #ax3.axis('off')
    for i in range(3):
      set_ilim(ax3,i,xx,1.7)
    ax3.set_axis_bgcolor('w')


    #####################
    # Diagnostics
    #####################
    self.fgd = plt.figure(24,figsize=(8,6))
    self.fgd.clf()
    set_figpos('2312')

    chrono = setup.t
    self.Kplot = estimate_good_plot_length(xx,chrono)
    pkk = arange(self.Kplot)
    ptt = chrono.tt[pkk]

    self.ax_e = plt.subplot(211)
    self.le,  = self.ax_e.plot(ptt,stats.rmse[pkk],'k',lw=2,alpha=1.0,label='Error')
    self.lv,  = self.ax_e.plot(ptt,stats.rmv [pkk],'b',lw=2,alpha=0.6,label='Spread')
    self.ax_e.set_ylabel('RMS')
    self.ax_e.legend()
    self.ax_e.set_xticklabels([])

    self.ax_i = plt.subplot(212)
    self.ls,  = self.ax_i.plot(ptt,stats.skew[pkk],'g',lw=2,alpha=1.0,label='Skew')
    self.lk,  = self.ax_i.plot(ptt,stats.kurt[pkk],'r',lw=2,alpha=1.0,label='Kurt')
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
      self.setter_mean = setup.f.plot(stats.mu[0])
      self.axu2.set_title('Mean')


    plt.pause(0.01)


  def update(self,E,k,kObs):
    if not self.is_available:
      return

    if self.is_paused:
      ch = getch()
      while ch not in [' ','\r']:
        ch = getch()
      if '\r' in ch:
        self.is_paused = False

    key = poll_input()
    if key is not None:
      if key == '\n':
        self.is_on = not self.is_on
      elif key == ' \n':
        self.is_on = True
        self.is_paused = not self.is_paused
        print("Press <Space> to step. Press <Enter> to resume.")
    if not self.is_on:
      return


    open_figns = plt.get_fignums()

    N,m = E.shape
    ii  = range(m)
    stats = self.stats
    
    #####################
    # Amplitudes
    #####################
    if plt.fignum_exists(self.fga.number):
      plt.figure(self.fga.number)
      self.lmu.set_ydata(stats.mu[k])
      self.lx .set_ydata(self.xx[k])

      #for i,l in enumerate(lE):
        #l.set_ydata(E[i])
      self.CI.remove()
      self.CI  = self.ax.fill_between(ii, \
          stats.mu[k] - self.ks*sqrt(stats.var[k]), \
          stats.mu[k] + self.ks*sqrt(stats.var[k]), alpha=0.4)

      plt.sca(self.ax)
      if hasattr(self,'obs'):
        try:
          self.obs.remove()
        except Exception:
          pass
        if kObs is not None:
          self.obs = self.yplot(self.yy[kObs])

      self.lew.set_ydata(stats.svals[k])
      self.lmf.set_ydata(abs(stats.umisf[k]))

      plt.pause(0.01)

    #####################
    # 3D phase space
    #####################
    if plt.fignum_exists(self.fg3.number):
      plt.figure(self.fg3.number)
      self.sx._offsets3d = juggle_axes(*vec2list2(self.xx[k,:3]),'z')
      self.sE._offsets3d = juggle_axes(*E[:,:3].T,'z')

      self.tail_xx = np.roll(self.tail_xx,1,axis=0)
      self.tail_xx[0] = self.xx[k,:3]
      self.ltx.set_data(self.tail_xx[:,0],self.tail_xx[:,1])
      self.ltx.set_3d_properties(self.tail_xx[:,2])

      self.tail_E = np.roll(self.tail_E,1,axis=0)
      self.tail_E[0] = E[:,:3]
      for n in range(N):
        self.ltE[n].set_data(self.tail_E[:,n,0],self.tail_E[:,n,1])
        self.ltE[n].set_3d_properties(self.tail_E[:,n,2])

      plt.pause(0.01)
      #self.fg3.savefig('figs/l63_' + str(k) + '.png',format='png',dpi=70)
    
    #####################
    # Diagnostics
    #####################
    if plt.fignum_exists(self.fgd.number):
      plt.figure(self.fgd.number)
      pkk = arange(self.Kplot)
      if k > self.Kplot:
        pkk += (k-self.Kplot)
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
      self.setter_mean(stats.mu[k])
      plt.pause(0.01)
    

def update_ylim(data,ax,Min=None,Max=None):
  maxv = minv = -np.inf
  for d in data:
    minv, maxv = np.maximum([minv, maxv], \
        1.1 * array([-1, 1]) * np.percentile(d,[1,99]))
  minv *= -1
  if minv == maxv:
    minv = np.min(d)
    maxv = np.max(d)
  if Max is not None:
    maxv = Max
  if Min is not None:
    minv = Min
  if (minv, maxv) != ax.get_ylim():
    ax.set_ylim(minv,maxv)


def set_ilim(ax,i,data,zoom=1.0):
  Min  = np.min(data[:,i])
  Max  = np.max(data[:,i])
  lims = round2sigfig([Min, Max])
  lims = inflate_ens(lims,1/zoom)
  if i is 0: ax.set_xlim(lims)
  if i is 1: ax.set_ylim(lims)
  if i is 2: ax.set_zlim(lims)


def estimate_good_plot_length(xx,chrono,scale=80):
  if xx.ndim == 2:
    xx = xx.ravel(order='F')
  try:
    K = scale * estimate_corr_length(xx)
  except ValueError:
    K = 0
  K = int(min(max(K,chrono.dkObs),chrono.K))
  T = round2sigfig(chrono.tt[K],2)
  K = find_1st_ind(chrono.tt >= T)
  return K

def get_plot_inds(chrono,Kplot,Tplot,xx):
  if Kplot is None:
    if Tplot: Kplot = find_1st_ind(chrono.tt >= Tplot)
    else:     Kplot = estimate_good_plot_length(xx,chrono)
  plot_kk    = chrono.kk[:Kplot+1]
  plot_kkObs = chrono.kkObs[chrono.kkObs<=Kplot]
  return plot_kk, plot_kkObs


def plot_3D_trajectory(stats,xx,dims=0,Kplot=None,Tplot=None):
  if isinstance(dims,int):
    dims = dims + arange(3)
  assert len(dims)==3

  chrono = stats.setup.t
  kk = get_plot_inds(chrono,Kplot,Tplot,xx)[0]
  T  = chrono.tt[kk[-1]]

  xx = xx      [kk][:,dims]
  mu = stats.mu[kk][:,dims]

  plt.figure(14).clf()
  set_figpos('2311 mac')
  ax3 = plt.subplot(111, projection='3d')

  ax3.plot   (xx[:,0] ,xx[:,1] ,xx[:,2] ,c='k',label='Truth')
  ax3.plot   (mu[:,0] ,mu[:,1] ,mu[:,2] ,c='b',label='DA estim.')
  ax3.scatter(xx[0 ,0],xx[0 ,1],xx[0 ,2],c='g',s=40)
  ax3.scatter(xx[-1,0],xx[-1,1],xx[-1,2],c='r',s=40)
  ax3.set_title('Phase space trajectory up to t={:.2g}'.format(T))
  ax3.set_xlabel('dim ' + str(dims[0]))
  ax3.set_ylabel('dim ' + str(dims[1]))
  ax3.set_zlabel('dim ' + str(dims[2]))
  ax3.legend(frameon=False)
  ax3.set_axis_bgcolor('w')
  # Don't do the following, coz it also needs the white grid,
  # which I can't get working for 3d.
  #for i in 'xyz': eval('ax3.w_' + i + 'axis.set_pane_color(sns_bg)')


def plot_time_series(stats,xx,dim=0,Kplot=None,Tplot=None):
  s      = stats
  chrono = stats.setup.t

  fg = plt.figure(12,figsize=(8,8)).clf()
  set_figpos('1313 mac')

  pkk,pkkObs = get_plot_inds(chrono,Kplot,Tplot,xx[:,dim])
  tt = chrono.tt

  ax_d = plt.subplot(3,1,1)
  ax_d.plot(tt[pkk],xx  [pkk,dim],'k',lw=3,label='Truth')
  ax_d.plot(tt[pkk],s.mu[pkk,dim],lw=2,label='DA estim.',alpha=1.0)
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
  ylim = 1.1*np.max([ylim, np.max(s.rmv[pkk])])
  ax_e.set_ylim(0,ylim)
  ax_e.set_ylabel('RMS')
  ax_e.legend()
  ax_e.set_xlabel('time (t)')

  #cm = mpl.colors.ListedColormap(sns.color_palette("BrBG", 256)) # RdBu_r
  #cm = plt.get_cmap('BrBG')
  fgH = plt.figure(16,figsize=(6,5)).clf()
  set_figpos('2312 mac')
  m = xx.shape[1]
  plt.contourf(arange(m),tt[pkk],xx[pkk],25)
  plt.colorbar()
  ax = plt.gca()
  ax.set_title("Hovmoller diagram (of 'Truth')")
  ax.set_xlabel('Element index (i)')
  ax.set_ylabel('Time (t)')
  add_endpoint_xtick(ax)


def add_endpoint_xtick(ax):
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
  ax.text(0.5,0.5,'[Not available]',fontsize=fs,va='center',ha='center')

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
  m      = s.mu.shape[1]

  fgE = plt.figure(15,figsize=(8,8)).clf()
  set_figpos('2321 mac')
  ax_r = plt.subplot(311)
  ax_r.set_xlabel('Element index (i)')
  ax_r.set_ylabel('Time-average magnitude')
  ax_r.plot(arange(m),mean(abs(s.err),0),'k',lw=2, label='Error')
  sprd = mean(s.mad,0)
  if m<10**3:
    ax_r.fill_between(arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.7,label='Spread')
  else:
    ax_r.plot(arange(len(sprd)),sprd,alpha=0.7,label='Spread')
  ax_r.set_title('Element-wise error comparison')
  #ax_r.set_yscale('log')
  ax_r.set_ylim(bottom=mean(sprd)/10)
  ax_r.set_xlim(right=m-1); add_endpoint_xtick(ax_r)
  ax_r.legend()
  #ax_r.set_position([0.125,0.6, 0.78, 0.34])
  plt.subplots_adjust(hspace=0.55)

  ax_s = plt.subplot(312)
  has_been_computed = not all(s.umisf[-1] == 0)
  ax_s.set_xlabel('Principal component index')
  ax_s.set_ylabel('Time-average magnitude')
  ax_s.set_title('Spectral error comparison')
  if has_been_computed:
    msft = mean(abs(s.umisf),0)
    sprd = mean(s.svals,0)
    ax_s.plot(        arange(len(msft)),              msft,'k',lw=2, label='Error')
    ax_s.fill_between(arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.7,label='Spread')
    ax_s.set_yscale('log')
    ax_s.set_ylim(bottom=1e-4*sum(sprd))
    ax_s.set_xlim(right=m-1); add_endpoint_xtick(ax_s)
    ax_s.legend()
  else:
    not_available_text(ax_s)


  ax_R = plt.subplot(313)
  ax_R.set_ylabel('Num. of occurence')
  ax_R.set_xlabel('RMSE')
  ax_R.set_title('Histogram of RMSE values')
  ax_R.hist(s.rmse[chrono.kkBI],alpha=1.0,bins=30,normed=0)


def plot_rank_histogram(stats):
  chrono = stats.setup.t

  has_been_computed = not all(stats.rh[-1] == 0)

  def are_uniform(w):
    """Test inital & final weights, not intermediate (for speed)."""
    (w[0]==1/N).all() and (w[-1]==1/N).all()

  fg = plt.figure(13,figsize=(8,4)).clf()
  set_figpos('2322 mac')
  #
  ax_H = plt.subplot(111)
  ax_H.set_title('(Average of marginal) rank histogram')
  ax_H.set_ylabel('Freq. of occurence\n (of truth in interval n)')
  ax_H.set_xlabel('ensemble member index (n)')
  ax_H.set_position([0.125,0.15, 0.78, 0.75])
  if has_been_computed:
    ranks = stats.rh[chrono.kkBI]
    m     = ranks.shape[1]
    N     = stats.w.shape[1]
    if are_uniform(stats.w):
      # Ensemble rank histogram
      integer_hist(ranks.ravel(),N,alpha=1.0)
    else:
      # Experimental: weighted rank histogram.
      # Weight ranks by inverse of particle weight. Why? Coz, with correct
      # importance weights, the "expected value" histogram is then flat.
      # Potential improvement: interpolate weights between particles.
      KBI= len(chrono.kkBI)
      w  = stats.w[chrono.kkBI]
      w  = np.hstack([w, ones((KBI,1))/N]) # define weights for rank N+1
      w  = array([ w[arange(KBI),ranks[arange(KBI),i]] for i in range(m)])
      w  = w.T.ravel()
      w  = np.maximum(w, 1/N/100) # Artificial cap. Reduces variance, but introduces bias.
      w  = 1/w
      integer_hist(ranks.ravel(),N,weights=w,alpha=1.0)
  else:
    not_available_text(ax_H)
  

  

def set_figpos(loc):
  """
  Place figure on screen, where 'loc' can be either
    NW, E, ...
  or
    4 digits (as str or int) to define grid m,n,i,j.
  Append the string 'mac' to place on mac monitor.
  Only works with both:
   - Patrick's monitor setup (Dell with Mac central-below)
   - Qt4Agg backend.
  """
  if 'Qt4Agg' is not matplotlib.get_backend():
    return
  fmw = plt.get_current_fig_manager().window

  # Current values
  #w_now = fmw.width()
  #h_now = fmw.height()
  #x_now = fmw.x()
  #y_now = fmw.y()

  # Constants
  Dell_w = 2560
  Dell_h = 1440
  Mac_w  = 2560
  Mac_h  = 1600
  # Why is Mac monitor scaled by 1/2 ?
  Mac_w  /= 2
  Mac_h  /= 2
  sysbar = 44
  winbar = 44 # coz window bars not computed by X11 forwarding ?

  loc = str(loc)
  if 'mac' in loc:
    x0 = Dell_w/4
    y0 = Dell_h+sysbar
    w0 = Mac_w
    h0 = Mac_h-sysbar
  else:
    x0 = 0
    y0 = 0
    w0 = Dell_w
    h0 = Dell_h
  
  # Def place function with offsets
  def place(x,y,w,h):
    fmw.setGeometry(x0+x,y0+y,w,h)

  if not loc[:4].isnumeric():
    if loc.startswith('NW'):
      loc = '2211'
    elif loc.startswith('SW'):
      loc = '2221'
    elif loc.startswith('NE'):
      loc = '2211'
    elif loc.startswith('SE'):
      loc = '2221'
    elif loc.startswith('W'):
      loc = '1211'
    elif loc.startswith('E'):
      loc = '1212'
    elif loc.startswith('S'):
      loc = '2121'
    elif loc.startswith('N'):
      loc = '2111'

  # Place
  m,n,i,j = [int(x) for x in loc[:4]]
  assert m>=i>0 and n>=j>0
  fudge = winbar/m
  place((j-1)*w0/n, (i-1)*h0/m+fudge/m, w0/n, h0/m-fudge)



