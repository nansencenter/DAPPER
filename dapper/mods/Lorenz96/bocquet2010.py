"""From Fig. 1 of `bib.bocquet2010beyond`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96 import step

tseq = modelling.Chronology(0.05, dko=1, T=4**3, BurnIn=20)

Nx = 10
Dyn = {
    'M': Nx,
    'model': step,
    'noise': 0,
}

X0 = modelling.GaussRV(M=Nx, C=0.001)

jj = np.arange(0, Nx, 2)
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 1.5

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

####################
# Suggested tuning
####################
# Why are these benchmarks superior to those in the article?
# We use, in the EnKF,
# - inflation instead of additive noise ?
# - Sqrt      instead of perturbed obs
# - random orthogonal rotations.
# The particle filters are also probably better tuned:
# - jitter covariance proportional to ensemble (weighted) cov
# - no jitter on unique particles after resampling
#
# For a better "picture" of the relative performances,
# see benchmarks in presentation from SIAM_SEAS.
# Note: They are slightly unrealiable (short runs).

#                                                   rmse.a:
# xps += EnKF_N(N=8,rot=True,xN=1.3)                # 0.31

# xps += PartFilt(N=50 ,NER=0.3 ,reg=1.7)           # 1.0
# xps += PartFilt(N=100,NER=0.2 ,reg=1.3)           # 0.36
# xps += PartFilt(N=800,NER=0.2 ,reg=0.8)           # 0.25

# xps += OptPF(   N=50 ,NER=0.25,reg=1.4,Qs=0.4)    # 0.61
# xps += OptPF(   N=100,NER=0.2 ,reg=1.0,Qs=0.3)    # 0.37
# xps += OptPF(   N=800,NER=0.2 ,reg=0.6,Qs=0.1)    # 0.25

# xps += PFa(     N=50 ,alpha=0.4,NER=0.5,reg=1.0)  # 0.45
# xps += PFa(     N=100,alpha=0.3,NER=0.4,reg=1.0)  # 0.38

# xps += PFxN     (N=30, NER=0.4, Qs=1.0,xN=1000)   # 0.48
# xps += PFxN     (N=50, NER=0.3, Qs=1.1,xN=100 )   # 0.43
# xps += PFxN     (N=100,NER=0.2, Qs=1.0,xN=100 )   # 0.32
# xps += PFxN     (N=400,NER=0.2, Qs=0.8,xN=100 )   # 0.27
# xps += PFxN     (N=800,NER=0.2, Qs=0.6,xN=100 )   # 0.25

# xps += PFxN_EnKF(N=25 ,NER=0.4 ,Qs=1.5,xN=100)    # 0.49
# xps += PFxN_EnKF(N=50 ,NER=0.25,Qs=1.5,xN=100)    # 0.36
# xps += PFxN_EnKF(N=100,NER=0.20,Qs=1.0,xN=100)    # 0.32
# xps += PFxN_EnKF(N=300,NER=0.10,Qs=1.0,xN=100)    # 0.28
