

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

X0  = GaussRV(C=0.01*eye(nX))

p = nX
jj= arange(p)
h = partial_direct_obs_setup(nX,jj)
h['noise'] = 0.1
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)
