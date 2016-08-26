

from common import *

from mods.LorenzXY.core import nX,J,m, dxdt_trunc, dxdt_det, dxdt_bad
from mods.LorenzXY.defaults import setup as setup_full

T = 4**3
setup_full.t.T = T
t = Chronology(dt=0.005,dtObs=0.01,T=T,BurnIn=6)

f = {
    'm'    : nX,
    'model': lambda x0,t0,dt: rk4(lambda t,x: dxdt_det(x),x0,np.nan,dt),
    'noise': 0,
    }

from mods.Lorenz95.core import typical_init_params
mu0,P0 = typical_init_params(nX)
X0 = GaussRV(mu0, 0.01*P0)

p = nX
obsInds = range(p)
@atmost_2d
def hmod(E,t):
  return E[:,obsInds]

h = {
    'm'    : p,
    'model': hmod,
    'noise': GaussRV(C=0.1*eye(p)),
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)
