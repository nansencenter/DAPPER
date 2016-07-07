# TODO: Do something about figure/subplot/graphic handles
# TODO: obs plotting

from common import *

from mpl_toolkits.mplot3d.art3d import juggle_axes

class LivePlot:
  def __init__(self,params,E,stats,xx,yy):
    N,m = E.shape
    dt = params.t.dt
    ii  = range(1,m+1)

    self.stats  = stats
    self.xx     = xx
    self.yy     = yy

    self.is_on  = False
    print('Press <Enter> to toggle live plot')

    ens_props = {} # yields rainbow
    #ens_props = {'color': 0.6*cwhite} # 0.7*cblue

    self.fg  = plt.figure(21,figsize=(8,8))
    self.fg.clf()
    set_figpos('E (mac)')

    self.ax  = plt.subplot(211)
    self.lmu,= plt.plot(ii,stats.mu[0,:],'b',lw=2,ls='-',label='Ens mean'  )
    self.lx ,= plt.plot(ii,xx[0      ,:],'k',lw=3,ls='-',label='Truth mean')

    #lE  = plt.plot(ii,E.T,lw=1,*ens_props)
    self.ks  = 3.0
    self.cE  = self.ax.fill_between(ii, \
        stats.mu[0,:] - self.ks*sqrt(stats.var[0,:]), \
        stats.mu[0,:] + self.ks*sqrt(stats.var[0,:]), \
        alpha=0.4,label=(str(self.ks) + ' sigma'))

    self.ax.legend()
    self.ax.set_xlabel('State index')

    ax2 = plt.subplot(212)
    A   = anom(E)[0]
    # TODO: Very wasteful (e.g. LA: m=1000)
    self.eE, = ax2.plot(sqrt(np.maximum(0,eigh(A @ A.T)[0][-min(N-1,m):])))
    ax2.set_xlabel('Sing. value index')
    ax2.set_yscale('log')
    ax2.set_ylim([1e-3,1e1])

    
    fg3 = plt.figure(23,figsize=(8,6))
    fg3.clf()
    set_figpos('NE (mac)')

    ax3      = fg3.add_subplot(111,projection='3d')
    self.sx  = ax3.scatter(*xx[0,:3]  ,s=30,c='k'      )
    self.sE  = ax3.scatter(*E[: ,:3].T,s=10,**ens_props)

    tail_k       = max([2,int(1/dt)])
    self.tail_xx = ones((tail_k,3)) * xx[0,:3] # init
    self.ltx,    = ax3.plot(*self.tail_xx.T)

    self.tail_E  = zeros((tail_k,N,3))
    for k in range(tail_k):
      self.tail_E[k,:,:] = E[:,:3]
    self.ltE = []
    for n in range(N):
      lEn, = ax3.plot(*self.tail_E[:,n,:].squeeze().T,**ens_props)
      self.ltE.append(lEn)

    #ax3.axis('off')
    ax3.set_xlim([-20,20])
    ax3.set_ylim([-20,20])
    ax3.set_zlim([-20,20])

    plt.pause(0.01)


  def update(self,E,k):

    if heardEnter():
      self.is_on = not self.is_on
    if not self.is_on:
      return

    N,m = E.shape
    ii  = range(1,m+1)
    stats = self.stats
    
    plt.figure(21)
    self.lmu.set_ydata(stats.mu[k,:])
    self.lx.set_ydata(self.xx[k,:])

    #for i,l in enumerate(lE):
      #l.set_ydata(E[i,:])
    self.cE.remove()
    self.cE  = self.ax.fill_between(ii, \
        stats.mu[k,:] - self.ks*sqrt(stats.var[k,:]), \
        stats.mu[k,:] + self.ks*sqrt(stats.var[k,:]), alpha=0.4)

    plt.pause(0.01)

    A = anom(E)[0]
    self.eE.set_ydata(sqrt(np.maximum(0,eigh(A @ A.T)[0][-min(N-1,m):])))

    fg3 = plt.figure(23)
    self.sx._offsets3d = juggle_axes(*vec2list2(self.xx[k,:3]),'z')
    self.sE._offsets3d = juggle_axes(*E[:,:3].T,'z')

    self.tail_xx = np.roll(self.tail_xx,1,axis=0)
    self.tail_xx[0,:] = self.xx[k,:3]
    self.ltx.set_data(self.tail_xx[:,0],self.tail_xx[:,1])
    self.ltx.set_3d_properties(self.tail_xx[:,2])

    self.tail_E = np.roll(self.tail_E,1,axis=0)
    self.tail_E[0,:,:] = E[:,:3]
    for n in range(N):
      self.ltE[n].set_data(self.tail_E[:,n,0],self.tail_E[:,n,1])
      self.ltE[n].set_3d_properties(self.tail_E[:,n,2])

    plt.pause(0.01)
    #input("Press Enter to continue...")
    #fg3.savefig('figs/l63_' + str(k) + '.png',format='png',dpi=100)
    


def estimate_good_plot_length(xx,chrono):
  L = estimate_corr_length(xx)
  if L is not 0:
    K = round2sigfig(100*L,2)
  else:
    K = chrono.K/10
  K = max(K,chrono.dtObs)
  return K



def plot_3D_trajectory(xx,stats,chrono,Kplot=None):
  """ xx: array with (exactly) 3 columns."""
  s = stats

  fig = plt.figure(14)
  set_figpos('SE')
  ax3 = fig.add_subplot(111, projection='3d')

  if Kplot is None:
    Kplot = estimate_good_plot_length(xx.ravel(order='F'),chrono)
  pkk = chrono.kk[chrono.kk<=Kplot]

  ax3.plot   (xx[pkk    ,0],xx[pkk    ,1],xx[pkk    ,2],c='k',label='Truth'   )
  ax3.plot   (s.mu[pkk  ,0],s.mu[pkk  ,1],s.mu[pkk  ,2],c='b',label='Ens.mean')
  ax3.scatter(xx[0      ,0],xx[0      ,1],xx[0      ,2],s=40 ,c='g'           )
  ax3.scatter(xx[pkk[-1],0],xx[pkk[-1],1],xx[pkk[-1],2],s=40 ,c='r'           )
  ax3.legend()


def plot_diagnostics_dashboard(xx,stats,chrono,N,dim=0,Kplot=None):
  s  = stats

  plt.figure(12,figsize=(8,8))
  set_figpos('NE')
  ax = plt.subplot(4,1,1)

  if Kplot is None:
    Kplot = estimate_good_plot_length(xx[:,dim],chrono)
  tt     = chrono.tt
  kkObs  = chrono.kkObs
  kk     = chrono.kk
  pkk    = kk<=Kplot
  pkkObs = kkObs[kkObs<=Kplot]

  plt.plot(tt[pkk],xx[pkk  ,dim],'k',lw=2,        label='Truth')
  plt.plot(tt[pkk],s.mu[pkk,dim],'g',lw=1,ls='--',label='Ens. mean')
  ax.set_ylabel('$x_{' + str(dim) + '}$',usetex=True,size=20)
  #plt.plot(kkObs,yy[:,dim],'k*')
  ax.legend()

  ax = plt.subplot(4,1,2)
  plt.plot(tt[pkk], s.rmse[pkk],'k',lw=2)
  plt.fill_between(tt[pkk], s.rmsv[pkk],alpha=0.4)
  ax.set_ylim(0,np.percentile(s.rmse[pkk],99))
  ax.set_ylabel('RMS Err and Var')

  ax = plt.subplot(4,1,3)
  plt.plot(tt[pkkObs], s.trHK[:len(pkkObs)],'k',lw=2)
  ax.set_ylim(0,np.percentile(s.trHK[:len(pkkObs)],99.6))
  ax.set_ylabel('trace(K)')
  ax.set_xlabel('time (t)')

  ax = plt.subplot(4,1,4)
  integer_hist(s.rh.ravel(),N,alpha=0.5)


def integer_hist(E,N,**kwargs):
  """Histogram for integers."""
  ax = plt.gca()
  plt.hist(E,bins=N+1, range=(-0.5,N+0.5),normed=1,**kwargs)
  ax.set_xlim(np.min(E)-0.5,np.max(E)+0.5)



def set_figpos(case):
  if 'Qt4Agg' is not matplotlib.get_backend():
    return
  fmw = plt.get_current_fig_manager().window
  w = fmw.width()
  h = fmw.height()
  x = fmw.x()
  y = fmw.y()
  if 'mac' in case:
    if case.startswith('NE'):
      fmw.setGeometry(640, 45, 640, 365)
    elif case.startswith('SE'):
      fmw.setGeometry(640, 431, 640, 365)
    elif case.startswith('E '):
      fmw.setGeometry(640, 45, 640, 751)
    else:
      sys.exit('Position not defined')
  else:
    if case.startswith('NE'):
      Cx = 1922
      Cy = -720
      w = 700
      h = 900
      fmw.setGeometry(Cx-w,Cy-h,w,h) # x,y,w,h
    elif case.startswith('SE'):
      Cx = 1922
      Cy = 0
      h = 500
      w = 700
      fmw.setGeometry(Cx-w,Cy-h,w,h)
    elif case.startswith('E '):
      fmw.setGeometry(642, -1418, 716, 1418)
    else:
      sys.exit('Position not defined')
  

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

