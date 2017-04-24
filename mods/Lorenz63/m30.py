# Moderate dtObs and non-0 Q.


from common import *

from mods.Lorenz63.core import step, dfdx
from aux.utils import Id_op, Id_mat

m = 3
p = m

t = Chronology(0.01,dkObs=15,T=4**4,BurnIn=4)

f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': GaussRV(C=2,m=m)
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=0.5,mu=mu0)

h = {
    'm'    : p,
    'model': Id_op(),
    'jacob': Id_mat(m),
    'noise': GaussRV(C=2,m=p)
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
