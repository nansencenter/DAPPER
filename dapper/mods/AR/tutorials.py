"""Settings of "T5 - Multivariate Kalman filter" of the DA-tutorials.

Only the 1st component is observed, and that rather poorly,
so the others (which drive it) must be inferred through the cross-covariances.

NB: The components are badly scaled (stationary std. devs. span 3 orders of
magnitude), so field-averaged stats (e.g. `rmse`) are dominated by `x_0`.
Inspect component-wise stats (e.g. `xp.stats.err.a`) for a fuller picture.

Difference from the tutorial: there, `X0` is near-zero, and so the experiment
is dominated by the transitory regime. Here `X0` is the stationary distribution,
making time-averaged stats immediately representative.
"""

import numpy as np

import dapper.mods as modelling
from dapper.mods.AR import Mmat, linear_step, stationary_cov

D = 4
M = Mmat(D, phi=0.9, c=1.0)
Q = 0.01**2 * np.diag(1 + np.arange(D))

tseq = modelling.Chronology(dt=1, dko=1, Ko=2000, Tplot=100, BurnIn=50)

Dyn = modelling.Operator(
    M=D, model=linear_step(M), linear=lambda x, t, dt: M, noise=modelling.GaussRV(C=Q)
)

Obs = modelling.Operator(**modelling.partial_Id_Obs(D, [0]), noise=30**2)

X0 = modelling.GaussRV(mu=np.zeros(D), C=stationary_cov(M, Q))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

####################
# Suggested tuning
####################
# Analytic expected (steady-state) values, per component (see `tests/test_AR.py`):
#                      x_0      x_1     x_2     x_3
# Climatology std   : 25.79    2.827   0.3273  0.0459
# ExtKF std (f)     : 10.31    1.795   0.2853  0.0454
# ExtKF std (a)     :  9.753   1.719   0.2797  0.0452
