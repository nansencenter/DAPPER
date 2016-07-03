# Reproduce results from
# table1 of sakov et al "DEnKF" (2008)
#mod = 'LA from Evensen''09';

from common import *

############################
# Set system params
############################
from mods.LA.funamentals import Fmat

m = 1000;
p = 40;
obsInds = arange(ceil(m/p/2),m,ceil(m/p)).astype(int)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1)


Fm = Fmat(m,-1,1,tseq.dt)
f  = lambda x,t,dt: asarray(F*x)
Q   = 1e-9 * eye(m) # approx 0
Q12 = sla.cholesky(Q,lower=True)

h   = lambda x,t: x[obsInds,:]
R   = 0.1 * eye(p)
R12 = sla.cholesky(R,lower=True)
Ri  = nla.inv(R)

# TODO:
#P0      = genCov(m,m/8);
#P012= sla.sqrtm(P0)


############################
# set DA method params
############################

# rmse_a =
N =
infl =
AMethod =
