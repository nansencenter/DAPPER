from common import *
sd0 = seed(4)

plt.style.use('AdInf/paper.mplstyle')


from mods.LorenzXY.core import nX,J,m
from mods.LorenzXY.defaults import setup
K = 1000
setup.t.BurnIn = 2
setup.t.K      = K
xx,yy = simulate(setup)

ii = arange(nX+1)
jj = arange(nX*J+1)
circX = np.mod(ii  ,nX)
circY = np.mod(jj,nX*J) + nX

# Animate linear
#plt.figure(1)
#lhX   = plt.plot(arange(nX+1)    ,xx[-1][circX],'b',lw=3)[0]
#lhY   = plt.plot(arange(nX*J+1)/J,xx[-1][circY],'g',lw=2)[0]
#for k in progbar(range(K),'Plotting'):
  #lhX.set_ydata(xx[k][circX])
  #lhY.set_ydata(xx[k][circY])
  #plt.pause(0.001)

# Overlay linear
from matplotlib import cm
fg = plt.figure(2)
fg.clear()
ax = fg.gca()
#ax.set_axis_off()
ax.set_xticks(arange(nX+1))
ax.set_xlabel('Y-index')
ax.set_xticklabels(['1'] + [str((i+1)*J) for i in circX])
ax2 = ax.twiny()
ax2.set_xlabel('X-index')
ax2.set_xticklabels([str(i+1) for i in circX])
#ax2.set_xticks(arange(nX+1))
L = 40 # Num of lines to plot
for p in range(L):
  k = 640 + p*1
  c = cm.viridis(1-p/L)
  a = 0.8-0.2*p/L
  plt.plot(ii  ,xx[k][circX],color=c,lw=2,alpha=a)[0]
  plt.plot(jj/J,xx[k][circY],color=c,lw=1,alpha=a)[0]
ax.set_ylim(-5,13)

# Double tick marks
# Basic
#ax.set_xticklabels([(str(i) + '/\n' + str(i*J)) for i in circX])
# Advanced
#labels = ['|1\n|1']
#for i in circX[1:-1]:
  #xi = i+1
  #yi = i*J
  #s = u"\u035F"*int(1+log10(yi+0.0001)) + '|' + str(xi) + '\n' + str(yi) + '|'+u"\u035E"
  #labels.append(s)
#labels.append(u"\u035F"*2+'|1\n256|'+u"\u035E")
#ax.set_xticklabels(labels,family='monospace')


# Convert to circular coordinates
def tX(zz):
  xx  = (40 + 3*zz)*cos(2*pi*ii/nX)
  yy  = (40 + 3*zz)*sin(2*pi*ii/nX)
  return xx,yy
def tY(zz):
  xx  = (80 + 15*zz)*cos(2*pi*jj/nX/J)
  yy  = (80 + 15*zz)*sin(2*pi*jj/nX/J)
  return xx,yy


# Animate circ
#plt.figure(3)
#lhX   = plt.plot(*tX(xx[-1][circX]),'b',lw=3)[0]
#lhY   = plt.plot(*tY(xx[-1][circY]),'g',lw=1)[0]
#for k in progbar(range(K),'Plotting'):
  #dataX = tX(xx[k][circX])
  #dataY = tY(xx[k][circY])
  #lhX.set_xdata(dataX[0])
  #lhX.set_ydata(dataX[1])
  #lhY.set_xdata(dataY[0])
  #lhY.set_ydata(dataY[1])
  #plt.pause(0.001)


# Overlay circ
from matplotlib import cm
fg = plt.figure(4)
fg.clear()
plt.plot(*tX(4.52*np.ones_like(circX)),color='k',lw=1)[0]
plt.plot(*tY(0.15*np.ones_like(circY)),color='k',lw=1)[0]
ax = fg.axes[0]
ax.set_axis_off()
ax.set_axis_bgcolor('white')
ax.set_aspect('equal')
L = 40 # Num of lines to plot
for p in range(L):
  k = 200 + p*1
  c = cm.viridis(1-p/L)
  a = 0.8-0.2*p/L
  plt.plot(*tX(xx[k][circX]),color=c,lw=2,alpha=a)[0]
  plt.plot(*tY(xx[k][circY]),color=c,lw=1,alpha=a)[0]




