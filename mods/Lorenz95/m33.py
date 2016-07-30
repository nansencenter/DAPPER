
from common import *

from mods.Lorenz95.sak08 import params

params.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)
params.name = os.path.relpath(__file__,'mods/')


# Reproduce raanes'2014 ("EnRTS and EnKS")
#from mods.Lorenz95.m33 import params
#cfg.N         = 25
#cfg.infl      = 1.08
#cfg.AMethod   = 'Sqrt'
#cfg.rot       = False
#cfg.tLag      = 2.0
#cfg.da_method = EnKS

