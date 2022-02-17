import dapper.da_methods as da

from dapper.mods.LA.evensen2009 import HMM

xx, yy = HMM.simulate()

xp = da.EnKF("PertObs", N=2)

# Feda: based on the docstring of `liveplotters` of HiddenMarkovModel,
#       xp.assimilate(...) should plot stuff if HMM has LivePlotters attached to it.
#       this seems not to happen.
# xp.assimilate(HMM, xx, yy,   liveplots=["spatial1d"])
xp.assimilate(HMM, xx, yy)
