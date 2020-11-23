"""Reproduce raanes'2016 ("EnRTS and EnKS")"""

import dapper as dpr
from dapper.mods.Lorenz96.sakov2008 import HMM

HMM.t = dpr.Chronology(0.01, dkObs=15, T=4**5, BurnIn=20)

# from dapper.mods.Lorenz96.raanes2016 import HMM
# xps += EnKS ('Sqrt',N=25,infl=1.08,rot=False,Lag=12)
# xps += EnRTS('Sqrt',N=25,infl=1.08,rot=False,cntr=0.99)
# ...
# print_averages(xps,avrgs,[],['rmse.u'])
