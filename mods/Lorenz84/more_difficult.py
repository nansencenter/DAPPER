# The settings in of Pajonk are too easy:
# the average value of trHK is 0.013.
# Here we increase dkObs to make the DA problem more difficult.

from common import *
from mods.Lorenz84.pajonk2012 import setup

setup.t  = Chronology(0.05,dkObs=10,T=4**5,BurnIn=20)

setup.X0 = GaussRV(C=0.01, mu=array([0.9, 0.44, 0.3]))

setup.name = os.path.relpath(__file__,'mods/')


####################
# Suggested tuning
####################
#cfgs.add(ExtKF,infl=8)
#cfgs.add(EnKF ,'Sqrt',N=10,infl=1.05)
#cfgs.add(EnKF ,'Sqrt',N=100,rot=True,infl=1.01)
#cfgs.add(EnKF_N ,N=4)
#cfgs.add(PartFilt, N=100, NER=0.4)
#cfgs.add(PartFilt, N=1000, NER=0.1)
