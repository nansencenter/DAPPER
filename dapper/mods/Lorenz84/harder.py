"""Harder settings than in `dapper.mods.Lorenz84.pajonk2012`.

This was adjudged by noting that with their settings,
the average val. of `trHK` is 0.013.

Here we increase `dko` to make the DA problem more difficult.
"""

import dapper.mods as modelling
from dapper.mods.Lorenz84.pajonk2012 import HMM as _HMM

HMM = _HMM.copy()
HMM.tseq = modelling.Chronology(0.05, dko=10, T=4**5, BurnIn=20)


####################
# Suggested tuning
####################
# xps += ExtKF(infl=8)
# xps += EnKF ('Sqrt',N=10,infl=1.05)
# xps += EnKF ('Sqrt',N=100,rot=True,infl=1.01)
# xps += EnKF_N (N=4)
# xps += PartFilt(N=100, NER=0.4) # add reg!
# xps += PartFilt(N=1000, NER=0.1) # add reg!
