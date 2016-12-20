
from common import *

from mods.Lorenz95.sak08 import setup

setup.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)
setup.name = os.path.relpath(__file__,'mods/')


# Reproduce raanes'2014 ("EnRTS and EnKS")
#from mods.Lorenz95.m33 import setup
#config.N         = 25
#config.infl      = 1.08
#config.upd_a   = 'Sqrt'
#config.rot       = False
#config.tLag      = 2.0
#config.da_driver = EnKS
#
#config.cntr      = 0.99
#config.da_driver = EnRTS
