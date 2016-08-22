

from common import *

from mods.LorenzXY.core import step, nX,J,m, mu0,P0, dfdx

t = Chronology(0.001,dkObs=10,T=4**2,BurnIn=2)

def xplot(x):
  lhX = plt.plot(arange(1,nX+1),x[:nX])[0]
  lhY = plt.plot(arange(1,nX*J+1)/J,x[nX:])[0]
  return lhX


f = {
    'm': m,
    'model': lambda x,t,dt: step(x,t,dt),
    'TLM'  : dfdx,
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

setup = OSSE(f,h,t,X0,**other)



# Example:

#  from common import *
#  
#  np.random.seed(2)
#  
#  from mods.LorenzXY.defaults import setup
#  
#  setup.t = Chronology(0.01,dkObs=1,T=4**1,BurnIn=2)
#  
#  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0
#  
#  
#  # truth
#  xx = zeros((chrono.K+1,f.m))
#  xx[0] = X0.sample(1)
#  
#  fg = plt.figure(1)
#  fg.clf()
#  circ = np.mod(range(9),8)
#  lhX, = plt.plot(arange(9),xx[0,circ],'b',lw=2)
#  lhY, = plt.plot(arange(8*32)/32,xx[0,8:],'g')
#  ax = plt.gca()
#  ax.set_ylim(-5,15)
#  
#  for k,_,t,dt in progbar(chrono.forecast_range):
#    xx[k] = f.model(xx[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
#  
#    lhX.set_ydata(xx[k,circ])
#    lhY.set_ydata(xx[k,8:])
#    plt.pause(0.001)
#  
#  #################
#  # Test Jacobians
#  #################
#  xt = xx[-1]
#  e  = 1e-3
#  dt = 1e-7
#  
#  J_fd = zeros((f.m,f.m))
#  for j in range(f.m):
#    ej     = zeros(f.m)
#    ej[j]  = e
#    J_fd[:,j] = f.model(xt + ej, np.nan, dt) - f.model(xt, np.nan, dt)
#  J_fd /= e
#  
#  J_an = f.TLM(xt,np.nan,dt)
#  
#  J_fd -= eye(f.m)
#  J_an -= eye(f.m)
#  
#  J_fd /= dt
#  J_an /= dt
