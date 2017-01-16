
from common import *

from mods.Lorenz95.sak08 import setup

setup.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)
setup.name = os.path.relpath(__file__,'mods/')


# Reproduce raanes'2015 ("EnRTS and EnKS")
#from mods.Lorenz95.m33 import setup
#config = DAC(EnKS ,'Sqrt',N=25,infl=1.08,rot=False,tLag=2.0)
#config = DAC(EnRTS,'Sqrt',N=25,infl=1.08,rot=False,cntr=0.99)
