# Plot scattergram of "unresolved tendency" 
# and the parameterization that emulate it.
# We plot the diff:
#   model_step/dt - true_step/dt    (1)
# Whereas Wilks plots
#   model_dxdt    - true_step/dt    (2) 
# Another option is:
#   model_dxdt    - true_dxdt       (3)
# Thus, for us (eqn 1), the model integration scheme matters.
# Also, Wilks uses
#  dt=0.001 for truth
#  dt=0.005 for model.

from common import *
plt.style.use('AdInf/paper.mplstyle')

###########################
# Setup
###########################
import mods.LorenzXY.core as LXY

# Default parameters are from Wilks 2005.
# Set these for Lorenz'95 settings.
#LXY.nX = 36
#LXY.J  = 10
#LXY.F  = 10
#LXY.check_parameters = False

from mods.LorenzXY.core import *
from mods.LorenzXY.wilks05_full import plot_state

K  = 4000
dt = 0.005
t0 = np.nan

seed(30) # 3 5 7 13 15 30
x0 = randn(ndim())

true_step  = with_rk4(dxdt      ,autonom=True)
model_step = with_rk4(dxdt_trunc,autonom=True)

###########################
# Compute truth trajectory
###########################
true_K = make_recursive(true_step,with_prog=1)
x0 = true_K(x0,int(2/dt),t0,dt)[-1] # BurnIn
xx = true_K(x0,K        ,t0,dt)

# Plot truth evolution
# setter = plot_state(xx[0])
# ax = plt.gca()
# for k in progbar(range(K),'plot'):
#   if not k%4:
#     setter(xx[k])
#     ax.set_title("t = {:<5.2f}".format(dt*k))
#     plt.pause(0.01)

###########################
# Compute unresovled scales
###########################
gg = zeros((K,nX)) # "Unresolved tendency"
for k,x in enumerate(xx[:-1]):
  X = x[:nX]
  Z = model_step(X,t0,dt)
  D = Z - xx[k+1,:nX]
  gg[k] = 1/dt*D

# Automated regression for deterministic parameterizations
pc = {}
for order in [0,1,2,3,4]:
  pc[order] = np.polyfit(xx[:-1,:nX].ravel(), gg.ravel(), deg=order)

###########################
# Scatter plot
###########################
xx = xx[:-1,:nX]
dk = int(8/dt/50) # step size
xx = xx[::dk].ravel()
gg = gg[::dk].ravel()

fig, ax = plt.subplots()
ax.scatter(xx,gg, facecolors='none', edgecolors=blend_rgb('k',0.5),s=40)
#ax.plot(xx,gg,'o',color=[0.7]*3)
ax.set_ylabel('Unresolved tendency ($q_{k,i}/\Delta t$)')
ax.set_xlabel('Resolved variable ($X_{k,i}$)')
# Wilks'2005
ax.set_xlim(-10,17)
ax.set_ylim(-10,20)
# Lorenz'95
#ax.set_xlim(-8,12)
#ax.set_ylim(-3,6)

# Plot pc
#uu = linspace(-10,17,201)
#ax.plot(uu,np.poly1d(pc[0])(uu),'g',lw=4.0)
#ax.plot(uu,np.poly1d(pc[1])(uu),'r',lw=4.0)
#ax.plot(uu,np.poly1d(pc[4])(uu),'b',lw=4.0)

###########################
# Wilks2005 Parameterization annotations
###########################
p0 = lambda x: 3.82+0.00*x
p1 = lambda x: 0.74+0.82*x                                        # lin.reg(gg,xx)
p3 = lambda x: .341+1.30*x -.0136*x**2 -.00235*x**3               # Arnold'2013
p4 = lambda x: .262+1.45*x -.0121*x**2 -.00713*x**3 +.000296*x**4 # Wilks'2005
uu = linspace(-10,17,201)
plt.plot(uu,p0(uu),'g',lw=4.0)
plt.plot(uu,p1(uu),'r',lw=4.0)
plt.plot(uu,p4(uu),'b',lw=4.0)
#plt.plot(uu,p3(uu),'y',lw=3.0)

def an(T,xy,xyT,HA='left'):
  ah = ax.annotate(T,
    xy    =xy ,   xycoords='data',
    xytext=xyT, textcoords='data',
    fontsize=16,
    horizontalalignment=HA,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3",lw=2)
    )
  return ah

s4 = '$0.262$\n$+1.45X$\n$-0.0121X^2$\n$-0.00713X^3$\n$+0.000296X^4$'
an('$3.82$'      ,(10  ,3.82),(10,-2) ,'center')
an('$0.74+0.82X$',(-7.4,-5.4),(1 ,-6))
an(s4            ,(7   ,8)   ,(0 ,10) ,'right')
