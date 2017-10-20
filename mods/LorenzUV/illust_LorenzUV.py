# Quick illustration.
# Sorry for the mess.

from common import *
from matplotlib import cm
plt.style.use('AdInf/paper.mplstyle')

# Setup
sd0 = seed(4)
from mods.LorenzUV.wilks05 import LUV
nX, J = LUV.nX, LUV.J

dt = 0.005
t0 = np.nan
K  = int(6/dt)

step_1 = with_rk4(LUV.dxdt,autonom=True)
step_K = make_recursive(step_1,with_prog=1)

x0 = 0.01*randn(LUV.m)
x0 = step_K(x0,int(2/dt),t0,dt)[-1] # BurnIn
xx = step_K(x0,K        ,t0,dt)

# Grab parts of state vector
ii = arange(nX+1)
jj = arange(nX*J+1)
circX = np.mod(ii  ,nX)
circY = np.mod(jj,nX*J) + nX
iX = np.hstack([0, 0.5+arange(8), 8])
def Xi(xx):
  interp = (xx[0]+xx[-1])/2
  return np.hstack([interp, xx, interp])

# Animate linear
plt.figure(1)
lhX   = plt.plot(arange(nX+1)    ,xx[-1][circX],'b',lw=3)[0]
lhY   = plt.plot(arange(nX*J+1)/J,xx[-1][circY],'g',lw=2)[0]
for k in progbar(range(K),'Plotting'):
  lhX.set_ydata(xx[k][circX])
  lhY.set_ydata(xx[k][circY])
  plt.pause(0.001)

# Overlay linear
fg = plt.figure(2)
fg.clear()
ax = fg.gca()
tY = arange(nX+1)
lY = ['\n1'] + ['\n'+str((i+1)*J) for i in circX]
tX = iX[1:-1]
lX = np.array([str(i+1) for i in range(nX)])
ax.set_xticks(tY)
ax.set_xticklabels(lY)
for t, l in zip(tX,lX):
  ax.text(t-0.05,-5.8,l,fontsize=mpl.rcParams['xtick.labelsize'])
ax.grid(color='k',alpha=0.6,lw=0.4,axis='both',which='major')
L = 30 # Num of lines to plot
for p in range(L):
  k = int(2e5*dt) + p*2
  c = cm.viridis(1-p/L)
  a = 0.8-0.2*p/L
  plt.plot(iX  ,Xi(xx[k][:nX]),color=c,lw=2  ,alpha=a)[0]
  plt.plot(jj/J,xx[k][circY]  ,color=c,lw=0.7,alpha=a)[0]
ax.set_ylim(-5,13)

# Convert to circular coordinates
# Should have used instead: projection='polar' 
def tX(zz):
  xx  = (40 + 3*zz)*cos(2*pi*ii/nX)
  yy  = (40 + 3*zz)*sin(2*pi*ii/nX)
  return xx,yy
def tY(zz):
  xx  = (80 + 15*zz)*cos(2*pi*jj/nX/J)
  yy  = (80 + 15*zz)*sin(2*pi*jj/nX/J)
  return xx,yy


# Animate circ
plt.figure(3)
lhX   = plt.plot(*tX(xx[-1][circX]),'b',lw=3)[0]
lhY   = plt.plot(*tY(xx[-1][circY]),'g',lw=1)[0]
for k in progbar(range(K),'Plotting'):
  dataX = tX(xx[k][circX])
  dataY = tY(xx[k][circY])
  lhX.set_xdata(dataX[0])
  lhX.set_ydata(dataX[1])
  lhY.set_xdata(dataY[0])
  lhY.set_ydata(dataY[1])
  plt.pause(0.001)


# Overlay circ
from matplotlib import cm
fg = plt.figure(4)
fg.clear()
plt.plot(*tX(4.52*np.ones_like(circX)),color='k',lw=1)[0]
plt.plot(*tY(0.15*np.ones_like(circY)),color='k',lw=1)[0]
ax = fg.axes[0]
ax.set_axis_off()
ax.set_facecolor('white')
ax.set_aspect('equal')
L = 40 # Num of lines to plot
for p in range(L):
  k = 143 + p*3
  c = cm.viridis(1-p/L)
  a = 0.8-0.2*p/L
  plt.plot(*tX(xx[k][circX]),color=c,lw=2,alpha=a)[0]
  plt.plot(*tY(xx[k][circY]),color=c,lw=1,alpha=a)[0]




