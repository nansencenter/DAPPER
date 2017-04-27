# For testing initial implementation of iEnKF


from common import *

from mods.Lorenz63.core import step
from aux.utils import Id_op, Id_mat

m = 3
p = m

t = Chronology(0.01,0.05,T=4**3,4)

f = {
    'm'    : m,
    'model': step,
    'noise': 0
    }


mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

h = {
    'm'    : p,
    'model': Id_op(),
    'noise': 2,
    }


other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################

#config = EnKF ('Sqrt',N=10)         # rmse_a = 0.205
#config = iEnKF('Sqrt',N=10,iMax=10) # rmse_a = 0.185

