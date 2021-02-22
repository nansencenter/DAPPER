"""Illustrate usage of DAPPER to make parameter estimation.

Implementation of using augmented state vector to estimate the forcing term
of Lorenz96.

DAPPER mods do not consider parameter estimation by default.
Hence, it is necessary to wrap the original time-stepping function
to unwrap state vector and parameters from the augmented state vector.

The initial distribution of the Force must set to be exactly 8.

Experiment set-up is partially from `bib.bocquet2013joint` with shorter
time and just EnKF.

"""

import numpy as np
from matplotlib import pyplot as plt

import dapper as dpr
import dapper.da_methods as da
import dapper.mods as modelling
import dapper.mods.Lorenz96 as core


######################################
# Wrap dxdt for parameter estimation
######################################
def dxdt_pe(x):
    """Separate parameter from x"""
    # declare returned array
    X = np.zeros_like(x)

    # ensure the consistency of Force shape
    shape = list(x.shape)
    shape[-1] = 1
    core.Force = np.zeros(shape)

    # unravel input into parameter and state
    core.Force[..., 0] = x[..., -1]
    X[..., :-1] = core.dxdt(x[..., :-1])

    return X


##############################
# Hidden Markov Model
##############################
# time setup
t = modelling.Chronology(0.05, dkObs=1, T=5e2, BurnIn=25)

# dim of state vector
Nx = 40
# number of parameters
P = 1

# Set-up Dyn
Dyn = {
    'M': Nx + P,
    'model': modelling.with_rk4(dxdt_pe, autonom=True),
    'noise': 0,
}

# Set the (dummy) initial distribution
X0 = modelling.GaussRV(M=Nx+P, C=1)

# set identity observation operator
jj = np.arange(Nx)
Obs = modelling.partial_Id_Obs(Nx+P, jj)
Obs['noise'] = 1.


HMM = modelling.HiddenMarkovModel(Dyn, Obs, t, X0)

##############################
# Twin experiment
##############################
# Generate the same random numbers every time
dpr.set_seed(3000)

# Simulate synthetic truth (xx) and noisy obs (yy)
# Here we assume the correct Force is 8
# Covariance must set to 0 to ensure the Force = 8 exactly
x0 = np.append(np.random.rand(Nx), 8.)
HMM.X0 = modelling.GaussRV(M=Nx+P, C=0., mu = x0)
xx, yy = HMM.simulate()

# Specify a DA method configuration ("xp" for "experiment")
# Specify covariance of initial perturbation and set the initial parameter.
x0[-1] = 7.
HMM.X0 = modelling.GaussRV(M=Nx+P, C=[1]*Nx+[0.1]*P, mu = x0)
xp = da.EnKF_N(N=20)

# Assimilate yy, knowing the HMM; xx is used to assess the performance
xp.assimilate(HMM, xx, yy)

# plot time series of analysis of parameter F
plt.figure(1)
plt.clf()
plt.axhline(8., color='r', label='truth')
plt.plot(xp.stats.mu.a[:, -1], label='analysis')
plt.xticks(np.arange(0, 1e4+1, 2e3), np.arange(0, 501, 100))
plt.xlabel('time')
plt.legend()
plt.show()

# Average the time series of various statistics
xp.stats.average_in_time()

# Print some averages
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))
