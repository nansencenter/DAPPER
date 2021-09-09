"""Settings as in `bib.bocquet2016localization`."""

import numpy as np

from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM


# Shift localization indices to adjust for time (i.e. in smoothing)
def loc_shift(ii, dt):
    shift = int(np.round(6.0*dt))  # Taken from Fig 4 of bocquet2016localization
    # NB: don't use builtin round; it returns integers -- except for round(0.0) !!!
    ii_new = ii + shift
    ii_new = np.remainder(ii_new, HMM.Nx)  # periodicity
    assert HMM.Nx == HMM.Obs.M, "This func assumes the obs operator is identity."
    return ii_new


HMM = _HMM.copy()
HMM.Obs.loc_shift = loc_shift


####################
# Suggested tuning
####################
# Reproduce data point dt=0.4 from figure 5                               # rmse.a:
# HMM.tseq.dko = 8
# xps += iEnKS('-N'  , N=20)                                              # 0.40
# xps += iLEnKS('Sqrt',N=10,loc_rad=12/1.82,infl=1.07)                    # 0.42
# xps += iLEnKS('-N' , N=10,loc_rad=12/1.82)                              # 0.45

# Reproduce data point L=10 from figure 6.a
# xps += iLEnKS('Sqrt' ,infl=1.02, N=10,nIter=4,Lag=10,loc_rad=12/1.82)   # 0.17
# xps += iLEnKS('-N' ,             N=10,nIter=4,Lag=10,loc_rad=12/1.82)   # 0.18
