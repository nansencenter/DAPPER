# For testing initial implementation of iEnKF


from common import *

from mods.Lorenz63.sak12 import *

t = Chronology(0.01,0.05,T=4**3,4)

other = {'name': os.path.relpath(__file__,'mods/')}
setup = TwinSetup(f,h,t,X0,**other)

####################
# Suggested tuning
####################

#cfgs += EnKF ('Sqrt',N=10)         # rmse_a = 0.205
#cfgs += iEnKF('Sqrt',N=10,iMax=10) # rmse_a = 0.185

