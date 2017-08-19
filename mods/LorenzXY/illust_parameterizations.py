
from common import *
from mods.LorenzXY.defaults import setup
from mods.LorenzXY.core import *

K  = 2000
dt = 0.005
t0 = np.nan

#seed(5)
x0 = randn(m)

step = lambda x0,t0,dt: rk4(lambda t,x: dxdt(x),x0,np.nan,dt)
step_k = make_recursive(step,with_prog=1)

x0 = step_k(x0,int(2/dt),t0,dt)[-1]
xx = step_k(x0,K        ,t0,dt)

# setter = setup.f.plot(xx[0])
# ax = plt.gca()
# for k in progbar(range(K),'plot'):
#   if not k%4:
#     setter(xx[k])
#     ax.set_title("t = {:<5.2f}".format(dt*k))
#     plt.pause(0.01)
