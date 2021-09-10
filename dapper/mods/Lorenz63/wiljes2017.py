"""Settings from `bib.wiljes2016second`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63.sakov2012 import HMM as _HMM
from dapper.mods.Lorenz63.sakov2012 import Nx

HMM = _HMM.copy()
HMM.tseq = modelling.Chronology(0.01, dko=12, T=4**5, BurnIn=4)

jj = np.array([0])
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 8
HMM.Obs = modelling.Operator(**Obs)

####################
# Suggested tuning
####################
# Reproduce benchmarks for NETF and ESRF (here EnKF-N) from left pane of Fig 1.
# from dapper.mods.Lorenz63.wiljes2017 import HMM # rmse.a reported by DAPPER / PAPER:
# ------------------------------------------------------------------------------
# HMM.tseq.Ko = 10**2
# xps += OptInterp()                                                # 5.4    / N/A
# xps += Var3D(xB=0.3)                                              # 3.0    / N/A
# xps += EnKF_N(N=5)                                                # 2.68   / N/A
# xps += EnKF_N(N=30,rot=True)                                      # 2.52   / 2.5
# xps += LNETF(N=40,rot=True,infl=1.02,Rs=1.0,loc_rad='NA')         # 2.61   / ~2.2
# xps += PartFilt(N=35 ,reg=1.4,NER=0.3)                            # 2.05   / 1.4 *
# *: tuning settings not given
#
# - The relevance of the experimental settings is questionable,
#   since the EnKF/NETF are barely able to beat 3D-Var
#   (with a little tuning beyond infl, 3D-Var would probably
#   beat these more sophisticated methods, so what then is
#   the interest in NETF beating the EnKF in this setting?).
# - Also, note that the EnKF manages "fine" with N=5,
#   while the paper only tests N>15.
# - Finally, note that there is a mismatch between DAPPER's NETF score
#   and the one reported in the paper. However, DAPPER's implementation
#   is not entirely wrong, since it has some skill.
#   Indeed, it is questionable whether it wrong at all.
