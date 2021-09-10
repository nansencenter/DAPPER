"""Settings as in `bib.grudzien2020numerical`."""

import dapper.mods as modelling
from dapper.mods.Lorenz96 import Tplot, x0
from dapper.mods.Lorenz96s import steppers


def HMMs(stepper="Tay2", resolution="Low", R=1):
    """Define the various HMMs used."""
    # Use small version of L96. Has 4 non-stable Lyapunov exponents.
    Nx = 10

    # Time sequence
    # Grudzien'2020 uses the below chronology with Ko=25000, BurnIn=5000.
    t = modelling.Chronology(dt=0.005, dto=.1, T=30, Tplot=Tplot, BurnIn=10)
    if resolution == "High":
        t.dt = 0.001
    elif stepper != "Tay2":
        t.dt = 0.01

    # Dynamical operator
    Dyn = {'M': Nx, 'model': steppers(stepper)}

    # (Random) initial condition
    X0 = modelling.GaussRV(mu=x0(Nx), C=0.001)

    # Observation operator
    jj = range(Nx)  # obs_inds
    Obs = modelling.partial_Id_Obs(Nx, jj)
    Obs['noise'] = R

    return modelling.HiddenMarkovModel(Dyn, Obs, t, X0)
