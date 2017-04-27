# Reproduce raanes'2016 ("EnRTS and EnKS")

from common import *

from mods.Lorenz95.sak08 import setup

setup.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)
setup.name = os.path.relpath(__file__,'mods/')


#from mods.Lorenz95.raanes2016 import setup
#cfgs += EnKS ('Sqrt',N=25,infl=1.08,rot=False,tLag=2.0)
#cfgs += EnRTS('Sqrt',N=25,infl=1.08,rot=False,cntr=0.99)
#...
#print_averages(cfgs,avrgs,[],['rmse_u'])
