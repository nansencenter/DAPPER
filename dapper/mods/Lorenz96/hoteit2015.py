"""Reproduce results from `bib.hoteit2015mitigating`."""

from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM

HMM = _HMM.copy()
HMM.tseq.T = 365
HMM.tseq.BurnIn = 4
# Further settings used in paper:
# Experiments repeated 10 times.
# Sometimes used: dko=4.
# Sometimes used: obs_inds = arange(Nx)[::2].
# Used localization: as in Whitaker/Hamill'2002: GC.


####################
# Suggested tuning
####################

# DAPPER only has localization implemented for
# the ETKF (LETKF) and the serial EAKF (SL_EAKF).
# Thus, direction comparison to paper (which always uses localization) is difficult.
# Still, these LETKF scores can be compared with Fig. 2 of the paper.
# They indicate that the LETKF is a little better than any scheme in the paper.
# However, the localization implementation is probably not fully equivalent
# (also, since optimal R seems to be around 6, I think Hoteit et al may have
# forgotten the sqrt(10/3) factor from Whitaker/Hamill).
#                                                               # Expected rmse.a:
# xps += LETKF(        N=10,rot=True,infl=1.02,loc_rad=4)       # 0.21
# xps += LETKF(        N=10,rot=True,infl=1.04,loc_rad=6)       # 0.20
# xps += LETKF(        N=10,rot=True,infl=1.10,loc_rad=10)      # 0.22
# xps += LETKF(        N=10,         infl=1.20,loc_rad=15)      # 0.29


# Without localization, DAPPER can also compare ESOPS to many other schemes:
# xps += EnKF ('Sqrt'           , N=28, infl=1.02,rot=True)     # 0.18
# xps += EnKF ('Serial'         , N=28, infl=1.02,rot=True)     # 0.18
# xps += EnKF ('Serial ESOPS'   , N=28, infl=1.02)              # 0.18
# xps += EnKF ('Serial Stoch'   , N=28, infl=1.08)              # 0.24
# xps += EnKF ('Serial Var1'    , N=28, infl=1.08)              # 0.24
# xps += EnKF ('PertObs'        , N=28, infl=1.08)              # 0.24
# As can be seen, ESOPS does well for a medium-large ensemble,
# getting RMSE scores on the level of the ETKF (i.e. 'Sqrt').
# For a small ensemble, however, ESOPS does not do quiet as well as the ETKF
# (and requires higher inflation to avoid divergence):
# xps += EnKF ('Sqrt'           , N=17, infl=1.03)              # 0.217
# xps += EnKF ('Serial'         , N=17, infl=1.06)              # 0.225
# xps += EnKF ('Serial ESOPS'   , N=17, infl=1.08)              # 0.242
