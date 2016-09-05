# Reproduce results from
# table1 of sakov et al "iEnKF" (2012)

from common import *

from mods.Lorenz63.core import step, dfdx

m = 3
p = m

#T = 4**6
T = 4**4
t = Chronology(0.01,dkObs=25,T=T,BurnIn=4)

m = 3
f = {
    'm'    : m,
    'model': lambda x,t,dt: step(x,t,dt),
    'TLM'  : dfdx,
    'noise': 0
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

h = {
    'm'    : p,
    'model': lambda x,t: x,
    'TLM'  : lambda x,t: eye(3),
    'noise': GaussRV(C=2,m=p)
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################

#cfg = DAM(EnKF,'Sqrt',N=3 ,infl=1.30)
#cfg = DAM(EnKF ,'Sqrt',N=10,infl=1.02,rot=True)          # 0.63 (sak: 0.65)
#cfg = DAM(iEnKF,'Sqrt',N=10,infl=1.02,rot=True,iMax=10)  # 0.31
#cfg = DAM(PartFilt, N=800, NER=0.1)                      # 0.275 (with N=4000)
#cfg = DAM(ExtKF, infl = 1.05); setup.t.dkObs = 10 # reduce non-linearity
