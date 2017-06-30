# Same as boc10, except that here ndim=40 (i.e. Fig. 5 of paper)
from mods.Lorenz95.boc10 import *

m = 40
f['m'] = m

X0 = GaussRV(m=m, C=0.001)

jj = arange(0,m,2)
h = partial_direct_obs_setup(m,jj)
h['noise'] = 1.5
 
other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,t,X0,**other)

####################
# Suggested tuning
####################
#                                                         # rmse_a 
#cfgs += EnKF_N(      N=24,      rot=True ,infl=1.01)     # 0.38
#cfgs += iEnKS('Sqrt',N=19,Lag=2,rot=False,infl=1.04)     # 0.39
#cfgs += iEnKS('-N'  ,N=19,Lag=2,rot=False,nu=1.5)        # 0.39

# NB: Particle filter scores very sensitive to rare events => Need T>=2000.
#cfgs += PartFilt(N=3000,NER=0.20,reg=1.2)                # 0.77
#cfgs += PartFilt(N=5000,NER=0.10,reg=1.1)                # 0.72
#cfgs += PartFilt(N=10000,NER=0.05,reg=0.8)               # 0.45

#cfgs += PFxN(    N=100, xN=1000,NER=0.9,Qs=1.0)          # 0.87
#cfgs += PFxN(    N=100, xN=1000,NER=0.3,Qs=0.9)          # 0.72 Diverges
#cfgs += PFxN(    N=1000,xN=100, NER=0.9,Qs=0.6)          # 0.51
