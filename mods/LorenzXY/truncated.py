# Use truncated models and larger dt.

from common import *

from mods.LorenzXY.core import nX,dxdt_detp






from mods.LorenzXY.defaults import t
t = t.copy()
t.dt = 0.05

f = {
    'm'    : nX,
    'model': with_rk4(dxdt_detp),
    'noise': 0,
    }



X0 = GaussRV(C=0.01*eye(nX))

h = partial_direct_obs_setup(nX,arange(nX))
h['noise'] = 0.1
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)
