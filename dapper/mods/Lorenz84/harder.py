# The settings of Pajonk are too easy:
# the average value of trHK is 0.013.
# Here we increase dkObs to make the DA problem more difficult.

from dapper import *
from dapper.mods.Lorenz84.pajonk2012 import HMM

HMM.t = Chronology(0.05,dkObs=10,T=4**5,BurnIn=20)


####################
# Suggested tuning
####################
# cfgs += ExtKF(infl=8)
# cfgs += EnKF ('Sqrt',N=10,infl=1.05)
# cfgs += EnKF ('Sqrt',N=100,rot=True,infl=1.01)
# cfgs += EnKF_N (N=4)
# cfgs += PartFilt(N=100, NER=0.4) # add reg!
# cfgs += PartFilt(N=1000, NER=0.1) # add reg!
