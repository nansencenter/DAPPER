

from common import *

from mods.LorenzXY.core import nX,J,m,dxdt,dfdx

t = Chronology(dt=0.001,dkObs=10,T=4**3,BurnIn=6)

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

from mods.Lorenz95.core import typical_init_params
mu0 = zeros(nX*(J+1))
P0  = eye(m)
mu0[:nX],P0[:nX,:nX] = typical_init_params(nX)
X0  = GaussRV(mu0, 0.01*P0)

# TODO: utilize partial_direct_obs...
p = nX
obsInds = range(p)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

H = zeros((p,m))
for i,j in enumerate(obsInds):
  H[i,j] = 1.0

h = {
    'm'    : p,
    'model': hmod,
    'jacob': lambda x,t: H,
    'noise': GaussRV(C=0.1*eye(p)),
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)


####################
# Suggested tuning
####################

# Not optimized at all:
#from mods.LorenzXY.defaults import setup
#setup.t.dt = 0.005
#config           = EnKF('Sqrt',N=100,rot=True)
#config.infl      = 1.01 # for p = nX
#config.infl      = 1.15 # for p = m
