# TODO: Do something about figure/subplot/graphic handles
# TODO: obs plotting

from common import *

from mpl_toolkits.mplot3d.art3d import juggle_axes


class LivePlot:
  """
  Live plotting functionality.
  """
  def __init__(self,setup,cfg,E,stats,xx,yy):
    N,m = E.shape
    dt = setup.t.dt
    ii  = range(m)

    self.stats  = stats
    self.xx     = xx
    self.yy     = yy
    self.setup = setup

    self.is_available = cfg.liveplotting
    if not self.is_available:
      return
    self.is_on     = False
    self.is_paused = False
    print('Press <Enter> to toggle live plot ON/OFF.')
    print('Press <Space> then <Enter> to pause.')

    #ens_props = {} # yields rainbow
    ens_props = {'color': 0.6*cwhite} # 0.7*cblue


    #####################
    # Amplitudes
    #####################
    self.fga = plt.figure(21,figsize=(8,8))
    self.fga.clf()
    set_figpos('E (mac)')

    self.ax  = plt.subplot(211)
    self.lmu,= plt.plot(ii,stats.mu[0],'b',lw=2,ls='-',label='Ens.mean')
    self.lx ,= plt.plot(ii,xx[0      ],'k',lw=3,ls='-',label='Truth')

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
    self.lmf, = ax2.plot(1+arange(m),abs(stats.umisf[0]),           'k',lw=2,label='Error')
    sprd = stats.svals[0]
    self.lew, = ax2.plot(1+arange(len(sprd)),sprd,'b',lw=2,label='Spread',alpha=0.9)
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
    set_figpos('NE (mac)')

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


    #####################
    # Diagnostics
    #####################
    self.fgd = plt.figure(24,figsize=(8,6))
    self.fgd.clf()
    set_figpos('SW (mac)')

    chrono = setup.t
    self.Kplot = estimate_good_plot_length(xx.ravel(order='F'),chrono)
    self.Kplot /= 4
    pkk = arange(self.Kplot).astype(int)
    ptt = chrono.tt[pkk]

    self.ax_e = plt.subplot(211)
    self.le,  = self.ax_e.plot(ptt,stats.rmse[pkk],'k',lw=2,alpha=1.0,label='Error')
    self.lv,  = self.ax_e.plot(ptt,stats.rmv[pkk],'b',lw=2,alpha=0.6,label='Spread')
    self.ax_e.set_ylabel('RMS')
    self.ax_e.legend()
    self.ax_e.set_xticklabels([])

    self.ax_i = plt.subplot(212)
    self.ls,  = self.ax_i.plot(ptt,stats.skew[pkk],'g',lw=2,alpha=1.0,label='Skew')
    self.lk,  = self.ax_i.plot(ptt,stats.kurt[pkk],'r',lw=2,alpha=1.0,label='Kurt')
    self.ax_i.legend()
    self.ax_i.set_xlabel('time (t)')


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

    

def update_ylim(data,ax,Min=None,Max=None):
  maxv = minv = -np.inf
  for d in data:
    minv, maxv = np.maximum([minv, maxv], \
        1.1 * array([-1, 1]) * np.percentile(d,[1,99]))
  minv *= -1
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


def estimate_good_plot_length(xx,chrono):
  L = estimate_corr_length(xx)
  if L is not 0:
    K = round2sigfig(100*L,2)
  else:
    K = chrono.K/10
  K = max(K,chrono.dtObs)
  return K

def get_plot_inds(chrono,Kplot,Tplot,xx):
  if Kplot is None:
    if Tplot:
      Kplot = find_1st_ind(chrono.tt >= Tplot)
    else:
      Kplot = estimate_good_plot_length(xx,chrono)
  kkObs      = chrono.kkObs
  plot_kk    = chrono.kk[chrono.kk<=Kplot]
  plot_kkObs = kkObs[kkObs<=Kplot]
  return plot_kk, plot_kkObs


def plot_3D_trajectory(xx,stats,chrono,\
    Kplot=None,Tplot=None):
  """ xx: array with (exactly) 3 columns."""
  s = stats

  plt.figure(14).clf()
  set_figpos('2311 mac')

  kk = get_plot_inds(chrono,Kplot,Tplot,xx.ravel(order='F'))[0]

  ax3 = plt.subplot(111, projection='3d')
  ax3.plot   (xx[kk    ,0],xx[kk    ,1],xx[kk    ,2],c='k',label='Truth'   )
  ax3.plot   (s.mu[kk  ,0],s.mu[kk  ,1],s.mu[kk  ,2],c='b',label='Ens.mean')
  ax3.scatter(xx[0     ,0],xx[0     ,1],xx[0     ,2],s=40 ,c='g'           )
  ax3.scatter(xx[kk[-1],0],xx[kk[-1],1],xx[kk[-1],2],s=40 ,c='r'           )
  ax3.legend()


def plot_time_series(xx,stats,chrono, \
    dim=0,Kplot=None,Tplot=None):
  s  = stats

  fg = plt.figure(12,figsize=(8,8)).clf()
  set_figpos('1313 mac')

  pkk,pkkObs = get_plot_inds(chrono,Kplot,Tplot,xx[:,dim])
  tt = chrono.tt

  ax_d = plt.subplot(3,1,1)
  ax_d.plot(tt[pkk],xx[pkk  ,dim],'k',lw=3,label='Truth')
  ax_d.plot(tt[pkk],s.mu[pkk,dim],'b',lw=2,label='DA estim.',alpha=0.6)
  #ax_d.set_ylabel('$x_{' + str(dim) + '}$',usetex=True,size=20)
  ax_d.set_ylabel('$x_{' + str(dim) + '}$',size=20)
  ax_d.legend()
  ax_d.set_xticklabels([])

  has_been_computed = not all(s.trHK[:] == 0)
  if has_been_computed:
    ax_K = plt.subplot(3,1,2)
    ax_K.plot(tt[pkkObs], s.trHK[:len(pkkObs)],'k',lw=2)
    ylim = 1.1 * np.percentile(s.trHK[:len(pkkObs)],99.6)
    ax_K.set_ylim(0,ylim)
    ax_K.set_ylabel('trace(H K)')
    ax_K.set_xticklabels([])

  ax_e = plt.subplot(3,1,3)
  ax_e.plot(        tt[pkk], s.rmse[pkk],'k',lw=2,label='Error')
  ax_e.fill_between(tt[pkk], s.rmv[pkk],alpha=0.4,label='Spread') 
  ylim = np.percentile(s.rmse[pkk],99)
  ylim = 1.1*np.max([ylim, np.max(s.rmv[pkk])])
  ax_e.set_ylim(0,ylim)
  ax_e.set_ylabel('RMS')
  ax_e.legend()
  ax_e.set_xlabel('time (t)')


  fgH = plt.figure(16,figsize=(6,5)).clf()
  set_figpos('2312 mac')
  m = xx.shape[1]
  plt.contourf(1+arange(m),tt[pkk],xx[pkk])
  plt.colorbar()
  ax = plt.gca()
  ax.set_title('Hovmoller diagram')
  ax.set_xlabel('Dimension index (i)')
  ax.set_ylabel('Time (t)')



def integer_hist(E,N,centrd=False,**kwargs):
  """Histogram for integers."""
  ax = plt.gca()
  rnge = (-0.5,N+0.5) if centrd else (0,N+1)
  ax.hist(E,bins=N+1,range=rnge,normed=1,**kwargs)
  ax.set_xlim(rnge)


def plot_ens_stats(xx,stats,chrono,cfg,dims=None):
  if not hasattr(cfg,'N'):
    return
  m = xx.shape[1]
  if not dims:
    dims = arange(m)
    d_text = '(averaged over all dims)'
  else:
    d_text = '(dims: ' + str(dims) + ')'

  fgE = plt.figure(15,figsize=(8,6)).clf()
  set_figpos('2321 mac')
  ax_r = plt.subplot(211)
  ax_r.set_xlabel('Component index')
  ax_r.set_ylabel('Time-average magnitude')
  ax_r.plot(1+arange(m),mean(abs(stats.err),0),'k',lw=2, label='Error')
  sprd = mean(stats.mad,0)
  ax_r.fill_between(1+arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.4,label='Spread')
  ax_r.set_title('Element-wise error comparison')
  #ax_r.set_yscale('log')
  ax_r.set_ylim(bottom=mean(sprd)/10)
  ax_r.legend()
  ax_r.set_position([0.125,0.6, 0.78, 0.34])

  ax_s = plt.subplot(212)
  has_been_computed = not all(stats.umisf[-1] == 0)
  if has_been_computed:
    ax_s.set_xlabel('Sing. value index')
    ax_s.set_ylabel('Time-average magnitude')
    ax_s.plot(1+arange(m),mean(abs(stats.umisf),0),'k',lw=2, label='Error')
    sprd = mean(stats.svals,0)
    ax_s.fill_between(1+arange(len(sprd)),[0]*len(sprd),sprd,alpha=0.4,label='Spread')
    ax_s.set_title('Spectral error comparison')
    ax_s.set_yscale('log')
    ax_s.set_ylim(bottom=1e-4*sum(sprd))
    ax_s.legend()


  fg = plt.figure(13,figsize=(8,6)).clf()
  set_figpos('2322 mac')
  #
  has_been_computed = not all(stats.rh[-1] == 0)
  if has_been_computed:
    ax_H = plt.subplot(211)
    ax_H.set_title('Rank histogram ' + d_text)
    ax_H.set_ylabel('Freq. of occurence\n (of truth in interval n)')
    ax_H.set_xlabel('ensemble member index (n)')
    ax_H.set_position([0.125,0.6, 0.78, 0.34])
    integer_hist(stats.rh[chrono.kkBI].ravel(),cfg.N,alpha=0.5)

    ax_R = plt.subplot(212)
    #ax_R.set_title('RMSE histogram')
    ax_R.set_ylabel('Num. of occurence')
    ax_R.set_xlabel('RMSE value')
    ax_R.hist(stats.rmse[chrono.kkBI],alpha=0.5,bins=30,normed=0)
  

  

@atmost_2d
def plt_vbars(xx,y1y2=None,*kargs,**kwargs):
  if y1y2 == None:
    y1y2 = plt.ylim()
  yy = np.tile(asmatrix(y1y2).ravel().T,(1,len(xx)))
  xx = np.tile(xx,(2,1))
  plt.plot(xx,yy,*kargs,**kwargs)

@atmost_2d
def plt_hbars(yy,x1x2=None,*kargs,**kwargs):
  if x1x2 == None:
    x1x2 = plt.xlim()
  xx = np.tile(asmatrix(x1x2).ravel().T,(1,len(yy)))
  yy = np.tile(yy,(2,1))
  plt.plot(xx,yy,*kargs,**kwargs)





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
  place((j-1)*w0/n,(i-1)*(h0-winbar)/m,w0/n,h0/m)



