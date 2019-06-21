# As in
# J. Mandel, E. Bergou, S. Gürol, S. Gratton, and I. Kasanický (2016)
# "Hybrid Levenberg–Marquardt and weak-constraint ensemble Kalman smoother method"

from dapper import *
from dapper.mods.Lorenz63.sak12 import HMM

# HMM.t = Chronology(0.01,KObs=10**5,BurnIn=500), with dkObs in [5:55].
# But it's pretty safe to shorten the BurnIn and KObs.

HMM.Obs = Operator(**{
    'M'    : 3,
    'model': lambda x,t: x**3,
    'noise': 8
    })

# It is unclear whether the model error (cov Q) is used
# just for the DA method, or also for the truth.
Q = lambda dkObs: GaussRV(M=HMM.Nx, C=0.01/(dkObs*HMM.t.dt))


####################
# Suggested tuning
####################
# Compare EnKF scores with those in figure 5 of paper.
# Note: We tune mult. inflation.
#       The paper only seems to use Q, probably yielding divergence.

# RMSE_a as reported by:                                      DAPPER  paper
# --------------------------------------------------------------------------
# from dapper.mods.Lorenz63.mandel2016 import HMM, Q
#
# t.dkObs = 55
# HMM.Dyn.noise = Q(t.dkObs)
# cfgs += EnKF  ('PertObs', N=100, infl=1.05)                # 0.20    [not tested]
# cfgs += EnKF  ('PertObs', N=10 , infl=1.30)                # nan     3.4

# t.dkObs = 35
# HMM.Dyn.noise = Q(t.dkObs)
# cfgs += EnKF  ('PertObs', N=100, infl=1.03)                # 0.20    [not tested]
# cfgs += EnKF  ('PertObs', N=10 , infl=1.20)                # 0.64    3.6

# t.dkObs = 5
# HMM.Dyn.noise = Q(t.dkObs)
# cfgs += EnKF  ('PertObs', N=100, infl=1.03)                # 0.05    [not tested]
# cfgs += EnKF  ('PertObs', N=10 , infl=1.20)                # 0.09    0.3


# Concerning the paper's first experiment (also Lorenz'63), reported in fig 2:
# In contrast to the above experiment, this one is not a filtering problem,
# but an initial condition (IC) problem (Q=0 is assumed).
# However, the IC they estimate is [1,1,1], which is OFF the attractor,
# and takes T=1 to get to the attractor.
# Meanwhile they're observing this mellow "run-way",
# which should give very precise information about the IC.
# There is no comparison to the EnKF or any other standard/baseline method.
