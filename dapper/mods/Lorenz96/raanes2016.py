"""Reproduce `bib.raanes2015rts`."""

import dapper.mods as modelling
from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM

HMM = _HMM.copy()
HMM.tseq = modelling.Chronology(0.01, dko=15, T=4**5, BurnIn=20)

# from dapper.mods.Lorenz96.raanes2016 import HMM
# xps += EnKS ('Sqrt',N=25,infl=1.08,rot=False,Lag=12)
# xps += EnRTS('Sqrt',N=25,infl=1.08,rot=False,DeCorr=0.99)
# ...
# print_averages(xps,avrgs,[],['rmse.u'])
