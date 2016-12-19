
from common import *

from mods.Lorenz95.sak08 import setup

setup.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)
setup.name = os.path.relpath(__file__,'mods/')


# Reproduce raanes'2014 ("EnRTS and EnKS")
#from mods.Lorenz95.m33 import setup
#cfg.N         = 25
#cfg.infl      = 1.08
#cfg.upd_a   = 'Sqrt'
#cfg.rot       = False
#cfg.tLag      = 2.0
#cfg.base_da = EnKS
#
#cfg.cntr      = 0.99
#cfg.base_da = EnRTS
