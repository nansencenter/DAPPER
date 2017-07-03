

from common import *

from mods.LorenzXY.core import nX,J,m,dxdt,dfdx

#t = Chronology(dt=0.001,dkObs=10,T=4**3,BurnIn=6)
t = Chronology(dt=0.005,dkObs=10,T=4**3,BurnIn=6)

def plot_state(x):
  circX = np.mod(arange(nX+1)  ,nX)
  circY = np.mod(arange(nX*J+1),nX*J) + nX
  lhX   = plt.plot(arange(nX+1)    ,x[circX],'b',lw=3)[0]
  lhY   = plt.plot(arange(nX*J+1)/J,x[circY],'g',lw=2)[0]
  ax    = plt.gca()
  ax.set_xticks(arange(nX+1))
  ax.set_xticklabels([(str(i) + '/\n' + str(i*J)) for i in circX])
  ax.set_ylim(-5,15)
  def setter(x):
    lhX.set_ydata(x[circX])
    lhY.set_ydata(x[circY])
  return setter


f = {
    'm'    : m,
    'model': lambda x0,t0,dt: rk4(lambda t,x: dxdt(x),x0,np.nan,dt),
    'jacob': dfdx,
    'noise': 0,
    'plot' : plot_state
    }

X0  = GaussRV(C=0.01*eye(m))

p = nX
jj= arange(p)
h = partial_direct_obs_setup(m,jj)
h['noise'] = 0.1

 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)


####################
# Suggested tuning
####################
# Not at all tuned:
#from mods.LorenzXY.defaults import setup
#setup.t.dt = 0.005
#cfgs += EnKF('Sqrt',N=100,rot=True,infl=1.01)
# infl = 1.15 for p = m
