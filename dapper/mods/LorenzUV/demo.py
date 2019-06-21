# Quick illustration.
# Sorry for the mess.

from dapper import *
from matplotlib import cm

from dapper.mods.LorenzUV.lorenz95 import LUV
nU, J = LUV.nU, LUV.J

dt = 0.005
K  = int(4/dt)

step_1 = with_rk4(LUV.dxdt,autonom=True)
step_K = with_recursion(step_1,prog=1)

x0 = zeros(LUV.M); x0[0] = 1
xx = step_K(x0,K,np.nan,dt)

# Grab parts of state vector
ii, wrap = setup_wrapping(nU  , periodic=1)
jj, wrap = setup_wrapping(nU*J, periodic=1)

# Animate linear
plt.figure(1)
lhU = plt.plot(ii  ,wrap(xx[-1,:nU]),'b',lw=3)[0]
lhV = plt.plot(jj/J,wrap(xx[-1,nU:]),'g',lw=2)[0]
for k in progbar(range(K),'Plotting'):
  lhU.set_ydata(wrap(xx[k,:nU]))
  lhV.set_ydata(wrap(xx[k,nU:]))
  plt.pause(0.001)








