# Reproduce results from fig2
# of raanes'2014 "extending sqrt method to model noise"
# which is based on Evensen'2009 EnKF paper.

from common import *

from mods.LA.funamentals import Fmat

m = 100;
p = 4;
obsInds = arange(ceil(m/p/2),m,ceil(m/p)).astype(int)

tseq = Chronology(dt=1,dkObs=5,T=300,BurnIn=-1)

Fm = Fmat(m,-1,1,tseq.dt)
#f  = lambda x,t,dt: asarray(F*x)
f = {
    'm': m,
    'model': lambda x,t,dt: (dt/tseq.dt) x @ Fm
    'noise': 0
    }

# TODO:
#P0      = genCov(m,m/8);
#P012= sla.sqrtm(P0)
X0 = GaussRV() #XXX

#% Initial noise
wnum  = 25;
#F.RV0 = @(N) X0pat(m,wnum,N);

#% Forward noise
wnumQ     = 25;
NQ        = 2*wnumQ+1;
A         = X0pat(m,wnumQ,NQ);
A         = 1/10 * anom(A)[0] / sqrt(NQ);
Q         = A.T @ A
% F.noise.chol = tsqrt(A*A',2*wnumQ);
F.noise.chol = A';




p = m
h = {
    'm': p,
    'model': lambda x,t: x[:,obsInds],
    'noise': GaussRV(C=0.01*eye(p))
    }
 
other = {'name': os.path.basename(__file__)}

params = OSSE(f,h,tseq,X0,**other)

####################
# Suggested tuning
####################
