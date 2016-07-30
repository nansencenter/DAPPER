

from common import *

from mods.Lorenz_2s.fundamentals import step, nX,J,m, mu0,P0

t = Chronology(0.001,dkObs=10,T=4**2,BurnIn=2)

def xplot(x):
  lhX = plt.plot(arange(1,nX+1),x[:nX])[0]
  lhY = plt.plot(arange(1,nX*J+1)/J,x[nX:])[0]
  return lhX


f = {
    'm': m,
    'model': lambda x,t,dt: step(x,t,dt),
    'noise': 0,
    'plot' : xplot
    }

X0 = GaussRV(mu0, 0.01*P0)

p = nX
obsInds = range(nX)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

H = zeros((p,m))
for i,j in enumerate(obsInds):
  H[i,j] = 1.0

h = {
    'm': p,
    'model': hmod,
    'noise': GaussRV(C=1*eye(p)),
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

params = OSSE(f,h,t,X0,**other)



# Example:

# from common import *
# 
# from mods.Lorenz_2s.defaults import params
# 
# f,h,chrono,X0 = params.f, params.h, params.t, params.X0
# 
# 
# 
# # truth
# xx = zeros((chrono.K+1,f.m))
# xx[0,:] = X0.sample(1)
# 
# fg = plt.figure(1)
# fg.clf()
# circ = np.mod(range(9),8)
# lhX, = plt.plot(arange(9),xx[0,circ],'b',lw=2)
# lhY, = plt.plot(arange(8*32)/32,xx[0,8:],'g')
# ax = plt.gca()
# ax.set_ylim(-5,15)
# 
# for k,_,t,dt in progbar(chrono.forecast_range):
#   xx[k,:] = f.model(xx[k-1,:],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
# 
#   lhX.set_ydata(xx[k,circ])
#   lhY.set_ydata(xx[k,8:])
#   plt.pause(0.001)

