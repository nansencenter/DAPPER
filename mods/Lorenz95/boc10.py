# From Fig. 1 of Bocquet 2010 "Beyond Gaussian Statistical Modeling
# in Geophysical Data Assimilation".
from common import *

from mods.Lorenz95 import core

t = Chronology(0.05,dkObs=1,T=4**3,BurnIn=20)

m = 10
f = {
    'm'    : m,
    'model': core.step,
    'noise': 0
    }

X0 = GaussRV(m=m, C=0.001)

obsInds = arange(0,m,2)
p       = len(obsInds)
@atmost_2d
def hmod(E,t): return E[:,obsInds]
H = direct_obs_matrix(m,obsInds)
h = {
    'm'    : p,
    'model': hmod,
    'jacob': lambda x,t: H,
    'noise': GaussRV(C=1.5*eye(p)),
    'plot' : lambda y: plt.plot(obsInds,y,'g*',ms=8)[0]
    }
 
other = {'name': os.path.relpath(__file__,'mods/')}

setup = OSSE(f,h,t,X0,**other)

####################
# Suggested tuning
####################
#cfgs.add(EnKF,'Sqrt',N=24,rot=True,infl=1.02)   # rmse_a = 0.32
#cfgs.add(PartFilt,N=100,NER=0.2,reg=1.3)        # rmse_a = 0.35
#cfgs.add(OptPF,   N=100,NER=0.2,reg=1.0,Qs=0.3) # rmse_a = 0.37
#cfgs.add(PartFilt,N=800,NER=0.2,reg=0.8)        # rmse_a = 0.25

# Note: contrary to the article, we use, in the EnKF,
# - inflation instead of additive noise ?
# - Sqrt      instead of perturbed obs
# - random orthogonal rotations.
# The PartFilt is also perhaps better tuned?
# This explains why the above benchmarks are superior to article.

#cfgs.add(PFD     ,N=30, NER=0.4,reg=0.6,xN=1000,Qs=1.0) # 0.48
#cfgs.add(PFD     ,N=50, NER=0.3,reg=0.8,xN=100 ,Qs=1.1) # 0.43
#cfgs.add(PFD     ,N=100,NER=0.3,reg=0.5,xN=100 ,Qs=1.0) # 0.38
#cfgs.add(PFD     ,N=300,NER=0.3,reg=0.3,xN=100 ,Qs=0.8) # 0.29
# PFD worse than PartFilt (bootstrap) with N>100. Potential causes:
# - Tuning
# - 'reg' is better (less bias coz 'no-uniq-jitter') than 'Qs'
