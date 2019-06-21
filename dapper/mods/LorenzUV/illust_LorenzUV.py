# Quick illustration.
# Sorry for the mess.

from dapper import *
from matplotlib import cm

# Setup
sd0 = seed(4)
# from dapper.mods.LorenzUV.wilks05 import LUV
from dapper.mods.LorenzUV.lorenz95 import LUV
nU, J = LUV.nU, LUV.J

dt = 0.005
t0 = np.nan
K  = int(10/dt)

step_1 = with_rk4(LUV.dxdt,autonom=True)
step_K = with_recursion(step_1,prog=1)

x0 = 0.01*randn(LUV.M)
x0 = step_K(x0,int(2/dt),t0,dt)[-1] # BurnIn
xx = step_K(x0,K        ,t0,dt)

# Grab parts of state vector
ii = arange(nU+1)
jj = arange(nU*J+1)
circU = np.mod(ii  ,nU)
circV = np.mod(jj,nU*J) + nU
iU = np.hstack([0, 0.5+arange(nU), nU])
def Ui(xx):
  interp = (xx[0]+xx[-1])/2
  return np.hstack([interp, xx, interp])

# Overlay linear
fg = plt.figure(2)
fg.clear()
ax = fg.gca()
L = 20 # Num of lines to plot
start = int(3e5*dt)
step  = 3
for i,Ny in enumerate(range(L)):
  k = start + Ny*step
  c = cm.viridis(1-Ny/L)
  a = 0.8-0.2*Ny/L
  plt.plot(iU  ,Ui(xx[k][:nU]),color=c,lw=2  ,alpha=a)[0]
  if i%2==0:
    plt.plot(jj/J,xx[k][circV]  ,color=c,lw=0.7,alpha=a)[0]
# Make ticks, ticklabels, grid
ax.set_xticks([])
ym,yM = -4,10
ax.set_ylim(ym,yM)
ax.set_xlim(0,nU)
dY = 4 # SET TO: 1 for wilks05, 4 for lorenz95
# U-vars: major
tU = iU[1:-1]
lU = np.array([str(i+1) for i in range(nU)])
tU = ccat(tU[0],tU[dY-1::dY])
lU = ccat(lU[0],lU[dY-1::dY])
for t, l in zip(tU,lU):
  ax.text(t,ym-.6,l,fontsize=mpl.rcParams['xtick.labelsize'],horizontalalignment='center')
  ax.vlines(t, ym, -3.78, 'k',lw=mpl.rcParams['xtick.major.width'])
# V-vars: minor
tV = arange(nU+1)
lV = ['1'] + [str((i+1)*J) for i in circU]
for i, (t, l) in enumerate(zip(tV,lV)):
  if i%dY==0:
    ax.text(t,-5.0,l,fontsize=9,horizontalalignment='center')
    ax.vlines(t,ym,yM,lw=0.3)
  ax.vlines(t, ym, -3.9, 'k',lw=mpl.rcParams['xtick.minor.width'])
ax.grid(color='k',alpha=0.6,lw=0.4,axis='y',which='major')






# # Convert to circular coordinates
# # Should have used instead: projection='polar' 
# def tU(zz):
#   xx  = (40 + 3*zz)*cos(2*pi*ii/nU)
#   yy  = (40 + 3*zz)*sin(2*pi*ii/nU)
#   return xx,yy
# def tV(zz):
#   xx  = (80 + 15*zz)*cos(2*pi*jj/nU/J)
#   yy  = (80 + 15*zz)*sin(2*pi*jj/nU/J)
#   return xx,yy
# 
# 
# # Animate circ
# plt.figure(3)
# lhU   = plt.plot(*tU(xx[-1][circU]),'b',lw=3)[0]
# lhV   = plt.plot(*tV(xx[-1][circV]),'g',lw=1)[0]
# for k in progbar(range(K),'Plotting'):
#   dataU = tU(xx[k][circU])
#   dataV = tV(xx[k][circV])
#   lhU.set_xdata(dataU[0])
#   lhU.set_ydata(dataU[1])
#   lhV.set_xdata(dataV[0])
#   lhV.set_ydata(dataV[1])
#   plt.pause(0.001)
# 
# 
# # Overlay circ
# from matplotlib import cm
# fg = plt.figure(4)
# fg.clear()
# plt.plot(*tU(4.52*np.ones_like(circU)),color='k',lw=1)[0]
# plt.plot(*tV(0.15*np.ones_like(circV)),color='k',lw=1)[0]
# ax = fg.axes[0]
# ax.set_axis_off()
# ax.set_facecolor('white')
# ax.set_aspect('equal')
# L = 40 # Num of lines to plot
# for Ny in range(L):
#   k = 143 + Ny*3
#   c = cm.viridis(1-Ny/L)
#   a = 0.8-0.2*Ny/L
#   plt.plot(*tU(xx[k][circU]),color=c,lw=2,alpha=a)[0]
#   plt.plot(*tV(xx[k][circV]),color=c,lw=1,alpha=a)[0]




